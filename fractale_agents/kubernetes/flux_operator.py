import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import fractale_agents.utils as utils
from fractale_agents.logger import logger

BUILD_PROMPT = """
You are an autonomous build sub-agent with expertise for building containers to run using Flux Framework. Your goal is to iteratively achieve the target provided by the user.

### YOUR OPERATING LOOP
1. DISCOVER: Look at the tools and prompts available to you.
2. ANALYZE: Container requirements provided by the user and constraints here.
3. ACT: Derive a Dockerfile to build. Call the appropriate tools (e.g., docker_build).
4. ENSURE: If data is needed for the run and can go in the container, it is in the WORKDIR.
5. VALIDATE: Ensure the build is successful (return code 0). If it fails, analyze the logs and FIX the Dockerfile.
6. PUSH: If the user requests it, push the container to the associated registry.

### CONSTRAINTS
- You MUST NOT do a multi-stage build.
- You MUST put all relevant executables on the PATH.
- You MUST return a final JSON object when the build is finished or unrecoverable.
- Final JSON format: {"status": "success|failure", "summary": "...", "container": "<URI>"}
"""

OPTIMIZE_SYSTEM_PROMPT = """
You are an autonomous optimization sub-agent with expertise for deploying Flux Framework and Flux Operator MiniCluster CRDs to Kubernetes. Your goal is to iteratively achieve the target provided by the user.

### YOUR OPERATING LOOP
1. DISCOVER: Look at the tools available to you.
2. ANALYZE: Check previous results (via database or logs) to understand the current state.
3. ACT: Decide on a configuration tweak (e.g., changing nodes, cores, or environment variables). Deploy using tools (e.g., kubectl_apply).
4. VALIDATE: After the job finishes, evaluate the Figure of Merit (FOM) from logs.
5. DECIDE: Either "retry" with a new configuration or "stop" because the goal is met or impossible.

### CONSTRAINTS
- You MUST save intermediate data and FOMs to the database using available storage tools.
- You MUST be precise with tool arguments.
- When you are finished, you MUST return a final JSON object:
  {"decision": "stop", "summary": "...", "final_fom": <value>}
"""


class BaseSubAgent:
    """
    Common base for autonomous sub-agents managing internal turn-based loops.
    """

    async def execute_loop(
        self, system_prompt: str, goal: str, context: str, max_turns: int
    ) -> Dict[str, Any]:
        from fractale.agents.base import backend

        current_prompt = f"{system_prompt}\n\n### USER GOAL\n{goal}\n\n### CONTEXT\n{context}"
        turn = 0

        while turn < max_turns:
            turn += 1
            logger.info(f"🧠 [{self.__class__.name}] Turn {turn}/{max_turns}")

            # 1. Ask the LLM
            response_text, tool_calls = backend.generate_response(
                prompt=current_prompt,
                use_tools=True,
                memory=True,
            )

            # 2. Handle Empty Responses
            if not response_text and not tool_calls:
                current_prompt = "Your last response was empty. Please provide your next tool call or final response."
                continue

            # 3. ACT: Execute Tool Calls
            if tool_calls:
                current_prompt = ""
                for call in tool_calls:
                    tool_result = await backend.call_tool(call)

                    # Neutralize noisy logs/tracebacks before feeding back
                    safe_content = clean_output(tool_result.content)
                    current_prompt += f"\nTool '{call['name']}' returned:\n{safe_content}"
                continue

            # 4. PARSE: Look for terminal JSON
            try:
                clean_json = utils.extract_code_block(response_text)
                if not clean_json:
                    current_prompt = (
                        "Please provide your final status/decision in a JSON markdown code block."
                    )
                    continue

                data = json.loads(clean_json)

                # Check for class-specific termination keys
                if "status" in data or data.get("decision") == "stop":
                    logger.info(f"✅ [{self.__class__.name}] Goal reached.")
                    data["turns_taken"] = turn
                    return data

            except (json.JSONDecodeError, KeyError):
                current_prompt = "Your response did not contain valid JSON. Please provide the required JSON structure."

        return {
            "status": "limit_reached",
            "message": f"Exceeded {max_turns} turns.",
            "goal": goal,
        }


class FluxBuildAgent(BaseSubAgent):
    """
    Agent optimized for iterative container builds and fixes.
    """

    name = "flux-build"
    description = (
        "An autonomous specialist that builds, fixes, and pushes container images "
        "designed to run Flux Framework workloads. It handles Dockerfile generation, "
        "build logs analysis, and registry management."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The container requirements (e.g., 'Build lammps with kokkos').",
            },
            "push": {
                "type": "boolean",
                "default": False,
                "description": "Whether to push the image to a registry after a successful build.",
            },
            "max_turns": {
                "type": "integer",
                "default": 20,
                "description": "Maximum attempts to fix build errors.",
            },
        },
        "required": ["goal"],
        "annotations": {"fractale.type": "agent"},
    }

    output_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "success or failure"},
            "summary": {"type": "string", "description": "Explanation of the build results"},
            "container": {"type": "string", "description": "The final container URI"},
            "turns_taken": {"type": "integer"},
        },
        "required": ["status"],
    }

    async def __call__(self, goal: str, push: bool = False, max_turns: int = 20) -> Dict[str, Any]:
        context = f"Push to registry requested: {push}"
        return await self.execute_loop(BUILD_PROMPT, goal, context, max_turns)


class FluxOperatorAgent(BaseSubAgent):
    """
    Agent optimized to deploy and optimize workloads via the Flux Operator.
    """

    name = "flux-operator"
    description = (
        "An autonomous specialist for deploying and optimizing Flux MiniClusters in Kubernetes. "
        "It can adjust MiniCluster configurations, evaluate performance metrics (FOM), "
        "and iteratively retry deployments to meet optimization goals."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The optimization target (e.g., 'Minimize lammps runtime').",
            },
            "task_context": {
                "type": "string",
                "description": "Initial configuration or previous results.",
            },
            "max_turns": {
                "type": "integer",
                "default": 30,
                "description": "Max turns for the optimization loop.",
            },
        },
        "required": ["goal"],
        "annotations": {"fractale.type": "agent"},
    }

    output_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "fom": {
                "type": ["number", "string", "null"],
                "description": "Final figure of merit achieved",
            },
            "turns_taken": {"type": "integer"},
            "goal": {"type": "string"},
        },
        "required": ["status"],
    }

    async def __call__(
        self, goal: str, task_context: str = "", max_turns: int = 30
    ) -> Dict[str, Any]:
        result = await self.execute_loop(OPTIMIZE_SYSTEM_PROMPT, goal, task_context, max_turns)

        # Standardize 'decision: stop' to 'status: completed' for the output schema
        if result.get("decision") == "stop":
            result["status"] = "completed"
        return result


def clean_output(data: Any) -> str:
    """
    Neutralizes characters that trigger malformed JSON responses
    without removing technical content (tracebacks).
    """
    text = str(data)
    text = text.replace("{", "❴").replace("}", "❵")
    text = text.replace("[", "❲").replace("]", "❳")
    text = text.replace('"', "'")
    text = text.replace("\\", "/")
    lines = text.splitlines()
    return "\n".join([f"| {line}" for line in lines])
