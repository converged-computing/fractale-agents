from typing import Any, Awaitable, Callable, Dict, Optional

from fractale_agents.agent import BaseSubAgent

discovery_prompt = """### PERSONA
You are a File Discovery and Data Analysis Agent. You specialize in navigating unknown directory structures, identifying data patterns, and extracting structured insights (like Figures of Merit) from large datasets.

### STRATEGY
1. Explore: List the DATA_ROOT to understand the organization (e.g., app folders, timestamps, versioning).
2. Schema Discovery: Identify a representative data file and use `filesystem_get_json_structure` to understand the data hierarchy without reading the whole file.
3. Query Testing: Use `filesystem_query_jq` or `filesystem_grep` on a single file to verify you can target the specific data needed.
4. Batch Consolidation:
   - First, run `filesystem_batch_extract_to_file` with `is_test=True` to verify your glob patterns and jq queries.
   - Then, run with `is_test=False` to process the entire dataset into a single file in the RESULT_ROOT.
5. Statistical Verification: Use `filesystem_summarize_result_file` to generate a high-level overview of the data you just collected.

### CONSTRAINTS
- You MUST NOT read large files directly if a query tool (jq/grep) is available.
- All extracted data and final reports MUST be written to the RESULT_ROOT.
- You MUST provide a "reason" field in your JSON response for every turn to explain your analytical process.
- You cannot request user interaction or ask questions.

### FINAL RESPONSE FORMAT
When the goal is met, return:
 {
  "action": "stop",
  "summary": "Clear text summary of discovery findings.",
  "report_path": "Path to the consolidated JSON/CSV in RESULT_ROOT.",
  "data_stats": {"count": 0, "min": 0, "max": 0, "mean": 0},
  "issues": ["List of failed files or anomalies found"],
  "reason": "Why the task is complete."
 }
"""


class FileDiscoveryAgent(BaseSubAgent):
    """
    Agent designed to find, query, and consolidate data across a filesystem.
    """

    name = "file-discovery"
    description = (
        "An expert data crawler that identifies files, extracts specific values "
        "using high-performance tools, and consolidates them into analytical reports."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The container requirements (e.g., 'Build lammps with kokkos').",
            },
            "max_turns": {
                "type": "integer",
                "default": 100,
                "description": "Max turns for the discovery loop.",
            },
        },
        "required": ["query_target"],
        "annotations": {"fractale.type": "agent"},
    }

    output_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "report_path": {"type": "string"},
            "data_stats": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                    "mean": {"type": "number"},
                },
            },
            "issues": {"type": "array", "items": {"type": "string"}},
            "turns_taken": {"type": "integer"},
        },
        "required": ["status", "summary", "report_path"],
    }

    async def __call__(
        self,
        goal: str,
        context: str = None,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Executes the file discovery and data extraction loop.
        """

        system_prompt = discovery_prompt

        # Execute the BaseSubAgent loop
        result = await self.execute_loop(
            system_prompt=system_prompt,
            goal=goal,
            context=context,
            max_turns=max_turns,
            process_callback=process_callback,
        )

        # Standardize result for output_schema
        if result.get("action") == "stop":
            result["status"] = "success"
        else:
            result["status"] = result.get("status", "failed")
            result["report_path"] = result.get("report_path", "none")
            result["summary"] = result.get(
                "summary", "The agent failed to complete the discovery task."
            )

        return result
