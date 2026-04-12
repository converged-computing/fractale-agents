import json
from typing import Any, Awaitable, Callable, Dict, Optional

import fractale_agents.utils as utils
from fractale_agents.agent import BaseSubAgent
from fractale_agents.logger import logger

from .common import shared_constraints

transform_prompt = f"""### PERSONA
You are an autonomous sub-agent with expertise in transforming job specifications.

{shared_constraints}
- If there is a directive that does not translate, you MUST leave it out and add a comment about the performance implications of the omission.
- You MUST preserve as many options as possible from the original.
- If you have a tool available, you MUST use it to validate the conversion.
- If you do not have a tool available, you MUST provide a "reason" the script is valid.

### INSTRUCTIONS
1. Analyze the original script provided in the CONTEXT.
2. Write a new script that converts from %s to %s.
3. Return the finished job specification in a JSON markdown code block with the key "jobspec".
4. Scrutinize your result and provide a list of "issues" one per item, in a "missing" return value.
  An issue might be:
    UNKNOWN_TO_ME: There is a parameter I am not familiar with that I cannot figure out (and provide detail)
    NO_ANALOGOUS: The from parameter has no analogous translation (and provide it)
    MISSING: You are not aware of any possible conversion (and details)
5. If you can predict the performance or other implications of the conversion, include them in "implications"
6. If the input script is not a workload manager batch file, you MUST return the JSON "jobspec" value as "noop".

- When you make each decision (response or tool call) you MUST return a JSON object with your reason/thinking:
  {"reason": "..."}
- When you are finished, you MUST return a final JSON object:
  {"action": "stop", "summary": "...", "issues": ["NO_ANALOGOUS: the parameter..."], "implications": "...", "jobspec": "#!/bin/bash ..."}
"""


class JobTransformAgent(BaseSubAgent):
    """
    Agent optimized to transform workload manager job specifications (e.g., Slurm to Flux).
    """

    name = "job-transform"
    description = (
        "An expert agent that converts job specifications (batch scripts) "
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
        from_manager = from_manger or "the detected workload manager"

        # System prompt and goal
        system_prompt = transform_prompt % (from_manager, to_manager)
        if error is not None:
            goal = (
                f"You previously attempted to convert a job specification and it did not validate. "
                f"Analyze the error and fix it.\n\n### ERROR\n{error}"
            )
        else:
            goal = (
                f"Convert the provided job specification from {from_manager} to {to_manager}. "
                f"The desired output format is a '{fmt}' script."
            )

        context = f"### ORIGINAL SCRIPT\n{script}\n"
        if previous_jobspec is not None:
            context += f"\n### PREVIOUS ATTEMPT\n{previous_jobspec}\n"

        # Execute the loop inherited from BaseSubAgent
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
