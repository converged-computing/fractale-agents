import json
from typing import Any, Awaitable, Callable, Dict, Optional

import fractale_agents.utils as utils
from fractale_agents.agent import BaseSubAgent
from fractale_agents.logger import logger

from .common import shared_constraints

generate_prompt = f"""### PERSONA
You are an autonomous sub-agent with expertise in submitting jobs to different workload managers.

### YOUR OPERATING LOOP
1. DISCOVER: Look at the tools and prompts available to you.
2. ANALYZE: Container request for a workload provided by the user and the constraints here.
3. ACT: Identify the right submission tool, and CALL it providing a command that represents the user requirements.
4. CHECK: Use an information or status tool to check job status, and get the job identifier.

{shared_constraints}
- If there is a requirement that cannot be represented, you MUST about the performance implications of the omission under "issues"
- When you make each decision (response or tool call) you MUST include a "reason"
- When you are finished, you MUST return ONE final JSON object:
{{"action": "stop", "status": "success|failure|other", "summary": "...", "command": "<command>", "issues": "<issues>", "job_id": "<job_id>", "reason": "..."}}
"""


class JobGenerateAgent(BaseSubAgent):
    """
    Agent optimized to generate and submit workload manager job specifications
    based on high-level user requirements (prompts) that can include the gamut
    from "run this exact command" to "here is my abstract description of a workload."
    """

    name = "job-generate"
    description = (
        "An expert agent that generates job submissions for workload managers (e.g., Slurm/Flux/Kubernetes)"
        "It analyzes requirements, identifies the correct submission, and either returns the jobid when "
        "submit and running, or waits for successful completion and returns on that."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "requirement": {
                "type": "string",
                "description": "The user requirement.",
            },
            "wait_for_completion": {
                "type": "boolean",
                "description": "Monitor until job is determined complete.",
                "default": False,
            },
            "max_turns": {
                "type": "integer",
                "default": 100,
                "description": "Max turns for the discovery and monitoring loop.",
            },
        },
        "required": ["requirement"],
        "annotations": {"fractale.type": "agent"},
    }

    output_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "failure", "other"],
                "description": "The final status of the submission and execution.",
            },
            "summary": {
                "type": "string",
                "description": "A summary of the actions taken and results.",
            },
            "command": {
                "type": "string",
                "description": "The exact command or script used for submission.",
            },
            "issues": {
                "type": "string",
                "description": "Any performance implications or requirements that could not be met.",
            },
            "job_id": {
                "type": "string",
                "description": "The identifier assigned by the workload manager.",
            },
        },
        "required": ["status", "summary", "command"],
    }

    async def __call__(
        self,
        requirement: str,
        wait_for_completion: bool = False,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Executes the job generation and submission loop.
        """
        system_prompt = generate_prompt
        goal = f"Generate and submit a job."
        context = f"The following requirements are provided: '{requirement}'. "
        if wait_for_completion:
            goal += f"Monitor the job until it reaches a terminal state (success/failure)."

        # Execute the loop inherited from BaseSubAgent
        # The agent is expected to use tools (like sbatch, flux submit, etc.)
        # and return the JSON structure specified in its instructions.
        return await self.execute_loop(
            system_prompt=system_prompt,
            goal=goal,
            context=context,
            max_turns=max_turns,
            process_callback=process_callback,
        )
