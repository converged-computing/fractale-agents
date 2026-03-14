import json
from typing import Any, Awaitable, Callable, Dict, Optional

import fractale.utils as utils
from fractale.logger import logger


class BaseSubAgent:
    """
    Common base for autonomous sub-agents managing internal turn-based loops.
    """

    async def execute_loop(
        self,
        system_prompt: str,
        goal: str,
        context: str,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:

        from fractale.agents.base import backend

        current_prompt = f"{system_prompt}\n\n### USER GOAL\n{goal}\n\n### CONTEXT\n{context}"
        turn = 0

        while turn < max_turns:
            turn += 1
            logger.info(f"🧠 [{self.__class__.__name__}] Turn {turn}/{max_turns}")
            logger.panel(current_prompt, title="Agent Prompt", color="green")

            response_text, tool_calls = backend.generate_response(
                prompt=current_prompt,
                use_tools=True,
                memory=True,
            )

            # Handle Empty Responses
            if not response_text and not tool_calls:
                current_prompt = "Your last response was empty. Please provide your next tool call or final response."
                continue

            # Case 1: Execute Tool Calls
            if tool_calls:
                current_prompt = ""
                for call in tool_calls:
                    tool_result = await backend.call_tool(call)
                    # We can't always clean the content. If not, use raw content
                    try:
                        safe_content = utils.clean_output(tool_result.content)
                    except:
                        safe_content = tool_result.content
                    current_prompt += f"\nTool '{call['name']}' returned:\n{safe_content}"
                    # Reset tool calls
                    tool_calls = []
                continue

            # Case 2: Parse JSON
            try:
                clean_json = utils.extract_code_block(response_text)
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

                # Are we processing custom callback?
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
                                    "turns_taken": turns,
                                    "reason": "Interrupted by caller",
                                    "goal": goal,
                                }
                            )
                            return data

                        elif "instruction" in instruction:
                            # Override the agent's next prompt based on caller feedback
                            current_prompt = instruction["instruction"]
                            continue

                # Normal class-specific termination checks
                if "action" in data and data.get("action") == "stop":
                    logger.info(f"✅ [{self.__class__.__name__}] Goal reached.")
                    data["turns_taken"] = turn
                    return data

            except (json.JSONDecodeError, KeyError):
                current_prompt = "Your response did not contain valid JSON. Please provide the required JSON structure."

        # Limit reached
        return {
            "status": "limit_reached",
            "message": f"Exceeded {max_turns} turns.",
            "goal": goal,
        }
