import json
from typing import Any, Awaitable, Callable, Dict, Optional

import fractale_agents.utils as utils
from fractale_agents.agent import BaseSubAgent
from fractale_agents.logger import logger

transform_prompt = f"""### PERSONA
You are an autonomous build sub-agent with expertise in transforming job specifications.

### REQUIREMENTS & CONSTRAINTS
- You MUST not make up directives that do not exist.
- You MUST preserve as many options as possible from the original.
- If there is a directive that does not translate, you MUST leave it out and add a comment about the performance implications of the omission.
- If you have a tool available, you MUST use it to validate the conversion.
- If you do not have a tool available, you MUST provide a "reason" the script is valid.

### INSTRUCTIONS
1. Analyze the original script provided in the CONTEXT.
2. Write a new script that converts from %s to %s.
3. When you have your finished job specification, return it in a JSON markdown code block with the key "jobspec".
4. If the input script is not a workload manager batch file, you MUST return the JSON "jobspec" value as "noop".
"""


class JobspecTransformAgent(BaseSubAgent):
    """
    Agent optimized to transform workload manager job specifications (e.g., Slurm to Flux).
    """

    name = "transform"
    description = (
        "An autonomous expert agent that converts job specifications (batch scripts) "
        "between different workload managers (e.g., SLURM to Flux). It can validate "
        "conversions and iteratively fix errors based on previous attempts."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "The original batch script or job specification to convert.",
            },
            "from_manager": {
                "type": "string",
                "description": "The name of the source workload manager (e.g., 'slurm').",
            },
            "to_manager": {
                "type": "string",
                "description": "The name of the target workload manager (e.g., 'flux').",
            },
            "fmt": {
                "type": "string",
                "default": "batch",
                "description": "Target format: 'batch' or 'jobspec' (canonical JSON representation).",
            },
            "error": {
                "type": ["string", "null"],
                "description": "Error message from a previous failed validation attempt (optional).",
            },
            "previous_jobspec": {
                "type": ["string", "null"],
                "description": "The previously generated jobspec that caused the error (optional).",
            },
            "max_turns": {
                "type": "integer",
                "default": 100,
                "description": "Max turns for the optimization/debugging loop.",
            },
        },
        "required": ["script", "from_manager", "to_manager"],
        "annotations": {"fractale.type": "agent"},
    }

    output_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "jobspec": {
                "type": "string",
                "description": "The transformed job specification, or 'noop' if the input was invalid.",
            },
            "turns_taken": {"type": "integer"},
            "message": {"type": "string"},
        },
        "required": ["status", "jobspec"],
    }

    async def __call__(
        self,
        script: str,
        from_manager: str,
        to_manager: str,
        fmt: str = "batch",
        error: Optional[str] = None,
        previous_jobspec: Optional[str] = None,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Executes the job specification transformation loop.
        """

        # 1. Construct the dynamic Goal
        if error is not None:
            goal = (
                f"You previously attempted to convert a job specification and it did not validate. "
                f"Analyze the error and fix it.\n\n### ERROR\n{error}"
            )
        else:
            goal = (
                f"Convert the provided job specification from '{from_manager}' to '{to_manager}'. "
                f"The desired output format is a '{fmt}' script."
            )

        # 2. Construct the System Prompt (Persona, Requirements, Instructions)
        system_prompt = transform_prompt % (from_manager, to_manager)

        # 3. Construct the Context payload
        context = f"### ORIGINAL SCRIPT\n{script}\n"

        if previous_jobspec is not None:
            context += f"\n### PREVIOUS ATTEMPT\n{previous_jobspec}\n"

        # 4. Execute the loop inherited from BaseSubAgent
        result = await self.execute_loop(
            system_prompt=system_prompt,
            goal=goal,
            context=context,
            max_turns=max_turns,
            process_callback=process_callback,
        )

        # Fallback safeguard to ensure output matches schema if the agent didn't provide it
        if "jobspec" not in result:
            result["jobspec"] = "noop"

        return result
