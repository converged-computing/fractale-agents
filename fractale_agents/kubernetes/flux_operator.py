import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import fractale_agents.utils as utils
from fractale_agents.agent import BaseSubAgent
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

### MiniCluster
To install the ARM-based MiniCluster CRD, apply the file: https://raw.githubusercontent.com/flux-framework/flux-operator/refs/heads/main/examples/dist/flux-operator-arm.yaml
To install the AMD MiniCluster CRD, https://raw.githubusercontent.com/flux-framework/flux-operator/refs/heads/main/examples/dist/flux-operator.yaml
The flux.container.image MUST match the operating system. Choose from:
  ghcr.io/converged-computing/flux-view-rocky:arm-9
  ghcr.io/converged-computing/flux-view-rocky:arm-8
  ghcr.io/converged-computing/flux-view-rocky:tag-9
  ghcr.io/converged-computing/flux-view-rocky:tag-8
  ghcr.io/converged-computing/flux-view-ubuntu:tag-noble
  ghcr.io/converged-computing/flux-view-ubuntu:tag-jammy
  ghcr.io/converged-computing/flux-view-ubuntu:tag-focal
  ghcr.io/converged-computing/flux-view-ubuntu:arm-jammy
  ghcr.io/converged-computing/flux-view-ubuntu:arm-focal

If you are generating multiple MiniCluster, name them ordinally in increasing order.
If you are getting logs for a MiniCluster, be mindful that the MiniCluster lead broker pod must be Completed to indicate the work is done.
For arm nodes, you MUST use an arm flux view image, and set flux arch to arm. Your command MUST be a single line to give to flux submit - no custom or multi-line scripts.
You should NOT delete and re-create the operator. You should NOT check the operator logs given the pod is Running.

### CONSTRAINTS
- If you request a specific node (e.g., for an autoscaler) you MUST only add a nodeSelector and no other annotations.
- You MUST save intermediate data and FOMs in your memory or using available storage tools.
- You MUST be precise with tool arguments.
- You MUST NOT include any flux command in your MiniCluster command. The operator wraps in a flux submit.
- You MUST wait for pods to initialize or be ready by sleeping and you must NOT delete preemptively.
- You must only install the Flux operator once and you MUST NOT delete it and reinstall.
- When you make each decision (response or tool call) you MUST return a JSON object with your reason/thinking:
  {"reason": "..."}
- When you are finished, you MUST return a final JSON object:
  {"action": "stop", "summary": "...", "final_fom": [<value1>,<value2>,<value3>]}
"""


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
                "default": 100,
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

    async def __call__(
        self,
        goal: str,
        push: bool = False,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        context = f"Push to registry requested: {push}"

        # Call the inherited execute_loop from BaseSubAgent
        return await self.execute_loop(
            system_prompt=BUILD_PROMPT,
            goal=goal,
            context=context,
            max_turns=max_turns,
            process_callback=process_callback,
        )


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
                "default": 100,
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
        self,
        goal: str,
        task_context: str = "",
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:

        # Call the inherited execute_loop from BaseSubAgent
        result = await self.execute_loop(
            system_prompt=OPTIMIZE_SYSTEM_PROMPT,
            goal=goal,
            context=task_context,
            max_turns=max_turns,
            process_callback=process_callback,
        )
        print(result)
        if result.get("action") == "stop":
            result["status"] = "completed"
        return result
