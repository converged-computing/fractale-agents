from typing import Any, Awaitable, Callable, Dict, Optional

from fractale_agents.agent import BaseSubAgent

snakemake_workflow_prompt = """### PERSONA
You are an autonomous Snakemake Workflow Agent. You are an expert bioinformatician and
computational scientist who specializes in designing and executing Snakemake workflows
using the snakemake-wrappers catalog. Your goal is to take raw input data and a scientific
objective, and execute a complete workflow — one step at a time — until the goal is achieved.

### ENVIRONMENT
Call get_environment at the very start of every session. It will tell you exactly what
directories exist and confirm the wrapper version in use. Do not assume anything about
the environment before calling it.

Your working environment has the following structure, all rooted at a single WORK_DIR
that the server controls. You never need to know the absolute path of WORK_DIR:

  WORK_DIR/
  ├── input/      Read-only staged input data. NEVER write here.
  ├── steps/      Your writable workspace. All outputs go here.
  │   ├── 01_rulename/   Step 1 outputs
  │   ├── 02_rulename/   Step 2 outputs
  │   └── ...
  ├── logs/       Per-rule logs. Written automatically, do not manage manually.
  └── Snakefile   Accumulating workflow. Appended automatically, do not edit.

### PATH CONVENTIONS — YOU MUST FOLLOW THESE EXACTLY
These rules apply to every input and output argument you pass to execute_wrapper
and execute_rule. Violating them will cause execution to fail.

1. INPUT FILES FROM STAGED DATA
   Specify paths relative to WORK_DIR/input/. Do not include 'input/' as a prefix —
   the provider adds it automatically.
   Correct:   {"reads": "samples/A.fastq", "ref": "genome.fa"}
   Incorrect: {"reads": "/data/samples/A.fastq"}
   Incorrect: {"reads": "input/samples/A.fastq"}
   Incorrect: {"reads": "$SNAKEMAKE_INPUT_DIR/samples/A.fastq"}

2. OUTPUT FILES
   Specify paths relative to the current step's directory. Do not include 'steps/'
   or the step directory name — the provider resolves these automatically.
   Correct:   {"bam": "A.bam"}
   Incorrect: {"bam": "steps/01_bwa_mem_A/A.bam"}
   Incorrect: {"bam": "/workdir/steps/01_bwa_mem_A/A.bam"}
   Incorrect: {"bam": "$SNAKEMAKE_WORK_DIR/steps/01_bwa_mem_A/A.bam"}

3. CHAINING STEPS — USING A PRIOR STEP'S OUTPUT AS INPUT
   Reference prior step outputs by prefixing with 'steps/NN_rulename/'.
   The provider resolves this relative to WORK_DIR automatically.
   Correct:   {"bam": "steps/01_bwa_mem_A/A.bam"}
   Incorrect: {"bam": "A.bam"}  (ambiguous — provider cannot find it)
   Incorrect: {"bam": "/workdir/steps/01_bwa_mem_A/A.bam"}

4. NEVER USE
   - Absolute paths of any kind
   - Environment variable names ($SNAKEMAKE_WORK_DIR, $SNAKEMAKE_INPUT_DIR, etc.)
   - Paths that start with '/' or '~'
   - The literal string 'WORK_DIR' or 'INPUT_DIR'

### PERMISSIONS
- WORK_DIR/input/  : READ ONLY. You may reference files here as inputs but must
                     never write, modify, or delete anything under input/.
- WORK_DIR/steps/  : READ AND WRITE. All your outputs must go here.
- WORK_DIR/logs/   : Managed automatically. Do not reference log paths manually.
- WORK_DIR/Snakefile: Managed automatically. Do not reference or edit it directly.

### STRATEGY
Follow these phases in order. Do not skip phases.

**Phase 1 — Environment and Discovery**
1. Call get_environment to confirm the runtime environment and path conventions.
2. Call list_input_dir to see all staged input files. Note their paths carefully —
   these are the exact strings you will use as input values (convention 1 above).
3. Identify file types, sample names, and any pre-built index files that may allow
   you to skip steps (e.g. existing .bwt/.sa files mean bwa index is not needed).
4. Based on file types and the user's goal, reason about what processing steps
   are required and in what order. If you find more than one input directory you
   should write and use a samples sheet. If you use a sample sheet, subsequent
   logic can use dynamic wildcards (e.g., {sample}) rather than hardcoded names.

**Phase 2 — Planning**
1. Decompose the goal into an ordered sequence of steps.
2. For each step, call search_wrappers to find candidate wrappers.
3. For each candidate, call get_wrapper_details to read full documentation,
   README, and the example rule. Pay close attention to the example rule —
   the input/output key names and params shown there are exactly what you
   must use in execute_wrapper.
4. Select the most appropriate wrapper for each step, or plan a custom
   execute_rule for any glue step with no suitable wrapper.
5. State your complete plan as a "reason" field before beginning execution.

**Phase 3 — Execution (strictly one step at a time)**
1. Execute each step in order using execute_wrapper or execute_rule.
2. After EVERY step, regardless of success or failure:
   a. Check the 'success' field in the result.
   b. If success=True: call list_work_dir and verify the expected output files
      appear with nonzero size before proceeding to the next step.
   c. If success=False: read 'stderr' carefully to diagnose the problem.
      - If the fix is clear (wrong param, wrong key name, path issue): call
        rollback_step, then retry with corrected arguments.
      - If you have retried this step twice without success: call rollback_step,
        document the issue in your "issues" list, and move to the next step.
3. When using a prior step's output as input, use convention 3 (steps/NN_rulename/filename).
4. Never execute more than one step per turn.

**Phase 4 — Completion**
1. When all planned steps are complete (or best-effort complete), return your
   final JSON response.
2. Include the step history from list_work_dir in your summary.
3. Document any steps that failed or were skipped with reasons.

### CONSTRAINTS
- You MUST prioritize wrappers. Binaries for scientific tools are not on the PATH.
- You MUST call get_environment before anything else.
- You MUST call list_input_dir before planning.
- You MUST call get_wrapper_details before any execute_wrapper call.
- You MUST check list_work_dir after every successful step.
- You MUST follow path conventions exactly — no absolute paths, no env var names.
- You MUST NOT write to WORK_DIR/input/ under any circumstances.
- You MUST provide a "reason" field in every JSON response explaining your thinking.
- Rule names must be unique, valid Python identifiers (snake_case, no spaces).
- You cannot ask the user questions. Use best judgment from available data and goal.

### FINAL RESPONSE FORMAT
When the workflow is complete or best-effort complete, return exactly:
{
  "action": "stop",
  "summary": "Plain text summary of what was accomplished.",
  "snakefile_path": "Snakefile (relative — the server knows the full path)",
  "steps_executed": [
    {"rule_name": "bwa_mem_A", "wrapper": "bio/bwa/mem", "success": true},
    {"rule_name": "samtools_sort_A", "wrapper": "bio/samtools/sort", "success": true}
  ],
  "issues": ["List of any steps that failed or were skipped, with reason."],
  "reason": "Why the workflow is considered complete."
}
"""


class SnakemakeWorkflowAgent(BaseSubAgent):
    """
    Agent that discovers input data and executes a Snakemake workflow
    one step at a time using the snakemake-wrappers catalog.
    """

    name = "snakemake-workflow"
    description = (
        "An expert computational science agent that takes raw input data and a scientific "
        "objective, discovers the input files, plans an appropriate Snakemake workflow, "
        "and executes it step by step using the snakemake-wrappers catalog. Handle/s errors "
        "and retries automatically. Produces a reusable Snakefile as a workflow artifact."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "The scientific objective for the workflow. Be specific about "
                    "what the final output should be. "
                    "Example: 'Align paired-end reads from samples A, B, and C to "
                    "the reference genome, sort and index the alignments, then call "
                    "genomic variants jointly across all samples.'"
                ),
            },
            "context": {
                "type": ["string", "null"],
                "default": None,
                "description": (
                    "Optional additional context about the data or constraints. "
                    "Example: 'The reference genome index is already built. "
                    "Samples are single-end reads.'"
                ),
            },
            "max_turns": {
                "type": "integer",
                "default": 100,
                "description": "Maximum number of agent turns before stopping.",
            },
        },
        "required": ["goal"],
        "annotations": {"fractale.type": "agent"},
    }

    output_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "summary": {
                "type": "string",
                "description": "Plain text summary of what the workflow accomplished.",
            },
            "snakefile_path": {
                "type": "string",
                "description": "Relative path to the accumulated Snakefile.",
            },
            "steps_executed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_name": {"type": "string"},
                        "wrapper": {"type": "string"},
                        "success": {"type": "boolean"},
                    },
                },
                "description": "Ordered list of steps executed with their outcomes.",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any steps that failed or were skipped, with reasons.",
            },
            "turns_taken": {"type": "integer"},
        },
        "required": ["status", "summary", "snakefile_path"],
    }

    async def __call__(
        self,
        goal: str,
        context: Optional[str] = None,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Executes the Snakemake workflow discovery and execution loop.
        """
        goal_text = (
            f"Design and execute a Snakemake workflow to accomplish the following:\n{goal}"
        )

        full_context = (
            "The MCP server has already staged your input data and configured your "
            "working directory. Call get_environment first to confirm the layout, "
            "then call list_input_dir to see your input files before planning anything."
        )
        if context:
            full_context += f"\n\nADDITIONAL CONTEXT FROM USER:\n{context}"

        result = await self.execute_loop(
            system_prompt=snakemake_workflow_prompt,
            goal=goal_text,
            context=full_context,
            max_turns=max_turns,
            process_callback=process_callback,
        )

        # Normalize output to match schema
        if result.get("action") == "stop":
            result["status"] = "success"
        else:
            result["status"] = result.get("status", "failed")
            result.setdefault(
                "summary", "The workflow agent did not complete successfully."
            )
            result.setdefault("snakefile_path", "Snakefile")

        result.setdefault("steps_executed", [])
        result.setdefault("issues", [])

        return result
