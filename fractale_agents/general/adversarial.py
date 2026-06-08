import json
from typing import Any, Awaitable, Callable, Dict, Optional

import fractale_agents.utils as utils
from fractale_agents.agent import BaseSubAgent
from fractale_agents.logger import logger

adversarial_prompt = """You are a meticulous, skeptical reviewer of automated code-analysis output. A first agent was given a Python script and a labeling taxonomy, and it produced a structured JSON extraction. Your job is to critically re-examine that extraction against the actual script, correct any errors, and return an improved result that conforms to the SAME schema. You are adversarial in the sense that you actively hunt for mistakes the first agent may have made — but you are evidence-driven, not destructive: every change you make must be justified by something in the script, and leaving a correct field unchanged is the expected outcome for most fields.

You are given three inputs.

ORIGINAL PROMPT (defines the taxonomy and the exact output schema the result must follow):
<original_prompt>
{original_prompt}
</original_prompt>

PREVIOUS RESULT (the first agent's extraction, which you will evaluate and revise):
<previous_result>
{previous_result}
</previous_result>

SCRIPT (the ground-truth source code being analyzed):
<script>
{script}
</script>

REVIEW CRITERIA

Prioritize the fields that cannot be checked mechanically, since these are where the first agent is most likely wrong and where your review adds the most value:
    - "description" / "step_description": Are they accurate, specific, and grounded in what the code actually does? Flag vague, generic, or hallucinated descriptions (e.g., claiming training when the code only loads data).
    - "domain": Is the inferred science domain consistent with the primary libraries and operations?
    - "operation.type" (or the equivalent classification field): Is each step classified correctly (e.g., a call to `.fit` is training; `read_csv` is data_io; a plotting call is visualization)? Re-derive the label from the code, do not trust the existing label.
    - Step segmentation: Are the steps a faithful decomposition of the script? Look for (a) missing steps that the code clearly performs, (b) spurious steps with no basis in the code, (c) steps that should be split or merged, and (d) incorrect ordering relative to execution flow.
    - "models": Are the named model architectures / algorithms actually present (e.g., a class subclassing nn.Module, an sklearn estimator), or invented?

You may also correct mechanical fields ("dependencies", "parameters", "frameworks", "primary_function"/"executed_file", "inputs"/"outputs") when the script plainly contradicts them, but treat these as secondary — they are independently validated elsewhere.

REVISION RULES

Be conservative and precise:
    - Change a field ONLY when the script gives clear evidence the previous value is wrong, imprecise, or missing. Do NOT invent problems, do NOT rewrite correct fields for style, and do NOT manufacture changes to appear thorough. "No change needed" is a valid and common verdict for a field.
    - When you change something, ground it in the script: cite the relevant construct (function name, class, call, or a short code reference).
    - If you suspect a problem but cannot confirm it from the script alone (e.g., a parameter or model defined in an imported module you cannot see), do NOT change the field — record it under "unresolved_concerns" with your reasoning instead.
    - Preserve the schema EXACTLY as defined in the ORIGINAL PROMPT: same keys, same nesting, same value types, same allowed enum values. Re-emit the COMPLETE result (all steps and fields), not just the parts you changed.
    - Preserve "step_id" ordering and numbering unless a segmentation fix requires renumbering, in which case renumber consistently.

OUTPUT PROTOCOL

Return EXACTLY ONE JSON object and nothing else — no markdown fences, no commentary. It carries the corrected extraction under "result" (conforming to the original schema) and your audit under "evaluation".

{{
    "action": "stop",
    "status": "success|failure|other",
    "summary": "...",
    "issues": "<issues or null>",
    "job_id": "<job_id or null>",
    "reason": "...",
    "result": {{
        ... the COMPLETE revised extraction, conforming exactly to the schema in the ORIGINAL PROMPT ...
    }},
    "evaluation": {{
        "verdict": "accepted_as_is" | "revised" | "rejected",
        "num_changes": <int>,
        "changes": [
            {{
                "location": "<path to the field, e.g. steps[2].operation.type>",
                "action": "fix" | "add" | "remove",
                "from": "<previous value or null>",
                "to": "<new value or null>",
                "evidence": "<the construct in the script that justifies the change>",
                "confidence": "high" | "medium" | "low"
            }}
        ],
        "unresolved_concerns": [ "<suspected but unconfirmable issue + reasoning>" ]
    }}
}}

CONTROL FIELD GUIDANCE
- "verdict": "accepted_as_is" if you made no changes; "revised" if you corrected one or more fields; "rejected" if the previous result was so wrong it had to be substantially rebuilt.
- "num_changes": the length of "changes".
- "status": "success" if you completed the review; "failure" if you could not (e.g., the script or previous result was unreadable); "other" for partial/ambiguous reviews.
- "summary": one or two sentences on what you found and changed.
- "result": if status is "failure", this may be the unchanged previous result or null.
"""


class AdversarialAgent(BaseSubAgent):
    """
    Adversarial reviewer agent. Given the original analysis prompt, a previous
    extraction result, and the source script, it critically re-examines the
    result against the script, corrects errors that the script clearly
    contradicts, and returns a revised result plus a structured audit trail.
    """

    name = "adversarial"
    description = (
        "An expert reviewer that takes a prior script-analysis result, the prompt "
        "that produced it, and the original script, then critiques and revises the "
        "result — focusing on the semantic fields (descriptions, domain, operation "
        "type, step segmentation, models) that cannot be checked mechanically — and "
        "returns a corrected result with an auditable changelog."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "The original source script being analyzed (ground truth).",
            },
            "previous_prompt": {
                "type": "string",
                "description": "The prompt given to the first agent; defines the taxonomy and output schema.",
            },
            "previous_result": {
                "type": "object",
                "description": "The first agent's extraction result (the taxonomy object) to evaluate and revise.",
            },
            "max_turns": {
                "type": "integer",
                "default": 100,
                "description": "Max turns for the review loop.",
            },
        },
        "required": ["script", "previous_prompt", "previous_result"],
        "annotations": {"fractale.type": "agent"},
    }
    output_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "failure", "other"],
                "description": "The final status of the review.",
            },
            "summary": {
                "type": "string",
                "description": "A summary of what was found and changed.",
            },
            "issues": {
                "type": "string",
                "description": "Any problems encountered during review (e.g., unreadable inputs).",
            },
            "result": {
                "type": "object",
                "description": "The complete revised extraction, conforming to the original schema.",
            },
            "evaluation": {
                "type": "object",
                "description": "Audit trail: verdict, num_changes, structured changes, unresolved_concerns.",
            },
        },
        "required": ["status", "summary", "result", "evaluation"],
    }

    async def __call__(
        self,
        script: str,
        previous_prompt: str,
        previous_result: Any,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Executes the adversarial review loop. `previous_result` may be a dict
        (the bare taxonomy object) or an already-serialized JSON string.
        """
        prev_result_str = (
            previous_result
            if isinstance(previous_result, str)
            else json.dumps(previous_result, indent=2)
        )

        # Render the template: fills the three input slots and converts the
        # {{ }} JSON braces to single braces. Braces inside the substituted
        # values (script code, result JSON) are NOT interpreted by .format().
        system_prompt = adversarial_prompt.format(
            original_prompt=previous_prompt,
            previous_result=prev_result_str,
            script=script,
        )

        goal = "Critically review the previous extraction against the script, correct only what the script contradicts, and return the revised result with an audit trail."
        context = "All three inputs (original prompt, previous result, and script) are provided in the system prompt."

        return await self.execute_loop(
            system_prompt=system_prompt,
            goal=goal,
            context=context,
            max_turns=max_turns,
            process_callback=process_callback,
        )
