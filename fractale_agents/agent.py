import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional

import fractale.utils as utils
from fastmcp import Client
from fractale.agents.base import AgentBase
from fractale.logger import logger


class BaseSubAgent(AgentBase):
    """
    Common base for autonomous sub-agents with Reactive Event support.
    """

    def __init__(self):
        super().__init__()

        # Accumulates events in the background between turns (like a buffer or queue)
        self.events = []
        self.subscriptions = set()

        # Create the client with the transport. This is created in the parent init, and we add
        # the handler to it.
        self.mcp_client = Client(self.transport, message_handler=self.handle_notification)

    async def handle_notification(self, notification: Any):
        """
        Receiving Side: receive the mcp server notifications.
        """
        # Convert Pydantic Union/RootModel into a plain dictionary
        # With fallback for raw dict
        try:
            raw = notification.model_dump()
        except Exception:
            raw = notification
        if not isinstance(raw, dict):
            print(f"Returning early on event: found invalid data: {raw}")
            return

        method = raw.get("method")
        params = raw.get("params", {})

        if method == "notifications/message":
            payload = params.get("data")
            if isinstance(payload, dict) and "subscription_id" in payload:
                provider = payload.get("provider", "unknown")
                event_data = payload.get("data")
                self.events.append({"provider": provider, "data": event_data})
                logger.info(f"📩 Received Event from {provider}")

    def add_subscription(self, result):
        """
        A subscription can be added by an agent or the calling sub-agent tool
        """
        # Content might be a string (json) or a parsed dict depending on backend
        content = result.data if hasattr(result, "data") else result.content
        if isinstance(content, str):
            content = utils.extract_code_block(content)
        sub_id = content.get("subscription_id")
        status = content.get("status")
        if sub_id:
            self.subscriptions.add(sub_id)
            logger.info(f"📡 Proactive Sub Active: {sub_id} ({status})")

    async def close_subscriptions(self):
        """
        Make call to stop events subscriptions from running.
        """
        for sub_id in self.subscriptions:
            try:
                await self.call_tool({"name": "unsubscribe", "args": {"subscription_id": sub_id}})
            except:
                pass

    def parse_safe_content(self, result, required=False):
        """
        Shared function for try/except to parse content.
        """
        try:
            return utils.clean_output(result)
        except:
            if required:
                return None
            return result

    async def execute_loop(
        self,
        system_prompt: str,
        goal: str,
        context: str,
        max_turns: int = 100,
        subscriptions: Optional[List[Dict[str, Any]]] = None,
        process_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        The main autonomous loop. Wraps execution in an MCP client context
        to allow for reactive background notifications.
        """
        from fractale.agents.base import backend

        # TODO vsoch: what to do if agent stops calls, we still have events?
        async with self.mcp_client:

            # Handle subscriptions (these are explicitly defined by us, the developers)
            # we are doing this for now because we KNOW these events are wanted.
            if subscriptions:
                for sub in subscriptions:

                    # Don't assume the server HAS the tool, but it should.
                    try:
                        # This is an explicit call to subscribe (the agent can make it too)
                        result = await self.call_tool(
                            {
                                "name": "subscribe",
                                "args": {
                                    "provider_name": sub["provider"],
                                    "params": sub.get("params") or {},
                                },
                            }
                        )
                        self.add_subscription(result)
                    except Exception as e:
                        logger.error(f"⚠️ Failed proactive subscription: {e}")

            # base_instructions persists tool results across turns
            turn = 0
            current_prompt = f"{system_prompt}\n\n### USER GOAL\n{goal}\n\n### CONTEXT\n{context}"

            while turn < max_turns:
                turn += 1

                # This is where we are going to grab events that have come in from notifications (mcp json-rpc)
                # since the last turn. In doing so, we clear the buffer.
                if self.events:
                    events_to_process = list(self.events)
                    self.events.clear()

                    # Let's avoid any possible duplicates, assume receiving is imperfect
                    seen = set()
                    event_log = []
                    for event in events_to_process:
                        event_data = json.dumps(e["data"])
                        if event_data not in seen:
                            seen.add(event_data)
                            event_log.append(f"{event['provider']} {event_data}")
                    event_log = "\n".join(event_log)

                    # Prepend the reactive news to the turn prompt.
                    # TODO vsoch: check how long the outputs are here, we might want to update event
                    # providers to be more succinct.
                    current_prompt = (
                        "### NEW EVENTS\n"
                        "The following events occurred in the environment since your last action:\n"
                        f"{event_log}\n\n"
                        f"{current_prompt}"
                    )

                logger.info(f"🧠 [{self.__class__.__name__}] Turn {turn}/{max_turns}")

                # Add more verbosity
                logger.panel(
                    title=f"🧠[{self.__class__.__name__}] Prompt",
                    message=current_prompt,
                    color="green",
                )
                response_text, tool_calls = backend.generate_response(
                    prompt=current_prompt,
                    use_tools=True,
                    memory=True,
                )

                # Handle empty output
                if not response_text and not tool_calls:
                    current_prompt = "\n\nYour last response was empty. Provide your next tool call or final response."
                    continue

                # Tool execution
                if tool_calls:
                    current_prompt = ""
                    for call in tool_calls:
                        result = await backend.call_tool(call)

                        # Catch autonomous subscriptions if the agent discovers the tool
                        if call["name"] == "subscribe":
                            self.add_subscription(result)
                        content = self.parse_safe_content(result.content)
                        current_prompt += f"\n\nTool '{call['name']}' returned:\n{content}"

                    # Reset tool calls
                    tool_calls = []

                    # Go back to top of loop
                    continue

                # Json Parsing
                clean_json = self.parse_safe_content(result.content, required=True)
                if not clean_json:
                    current_prompt = (
                        "Please provide your final status/decision in a JSON markdown code block."
                    )
                    continue

                data = json.loads(clean_json)

                # A reason can be provided in data
                if "reason" in data and data["reason"]:
                    logger.panel(
                        title=f"🧠[{self.__class__.__name__}] Thinking",
                        message=data["reason"],
                        color="blue",
                    )

                if process_callback:
                    # The instruction MUST return back json response:
                    # {"action": "stop", "instruction": "...."}
                    instruction = await process_callback(data)
                    if instruction:
                        if instruction.get("action") == "stop":
                            logger.info(
                                f"🛑 [{self.__class__.__name__}] Force stopped by callback."
                            )
                            data.update(
                                {
                                    "turns_taken": turn,
                                    "goal": goal,
                                    "reason": "Interrupted by caller",
                                }
                            )
                            return data
                        elif "instruction" in instruction:
                            current_prompt = instruction["instruction"]
                            continue

                # Normal class-specific termination checks
                if "action" in data and data.get("action") == "stop":
                    logger.info(f"✅ [{self.__class__.__name__}] Goal reached.")
                    data["turns_taken"] = turn
                    await self.close_subscriptions()
                    return data

        await self.close_subscriptions()
        return {
            "status": "limit_reached",
            "message": f"Exceeded {max_turns} turns.",
            "goal": goal,
            "turns_taken": turn,
        }
