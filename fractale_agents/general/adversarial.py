import json
from typing import Any, Awaitable, Callable, Dict, Optional

from fractale_agents.agent import BaseSubAgent

adversarial_prompt = """You are a meticulous, skeptical reviewer of the output of an expert analysis agent. A first ("expert") agent was given some input and a labeling task, and it produced a structured result. Your job is to critically re-examine that result against the actual input, correct anything the input clearly contradicts, and return an improved result that conforms to the SAME schema the expert agent was asked to follow. You are adversarial in that you actively hunt for the expert agent's mistakes — but you are evidence-driven, not destructive: every change must be justified by the input, and leaving a correct field unchanged is the expected outcome for most fields.

You are provided the following material.

In the TASK CONTEXT:
    - ORIGINAL PROMPT: the prompt the expert agent was given. It defines the task, the labeling taxonomy, and the EXACT output schema the result must follow.
    - PREVIOUS RESULT: the expert agent's output, which you will evaluate and revise.

In the TASK GOAL:
    - The specific task the agent was asked to do. This is your GROUND TRUTH — re-derive your judgments from it directly.

REVIEW CRITERIA (general)

Prioritize the fields that require interpretation and cannot be checked mechanically, since these are where the expert agent is most likely wrong and where your review adds the most value. Review fields to be accurate, specific, and grounded in truth. Flag vague, generic, or hallucinated text. Look for missing or incomplete items, or incorrect ordering or intent. You may also correct concrete fields (identifiers, parameters, references, paths) when the input plainly contradicts them, but treat these as secondary — they are typically validated by other means.

REVISION RULES

Be conservative and precise:
    - Change a field ONLY when the input gives clear evidence the previous value is wrong, imprecise, or missing. Do NOT invent problems, do NOT rewrite correct fields for style, and do NOT manufacture changes to appear thorough. "No change needed" is a valid and common verdict.
    - When you change something, ground it in the input: cite the specific construct, line, phrase, or element that justifies the change.
    - If you suspect a problem but cannot confirm it from the provided input alone (e.g., a value defined in an external resource you cannot see), do NOT change the field — record it under "unresolved_concerns" with your reasoning instead.
    - Preserve the schema EXACTLY as defined in the ORIGINAL PROMPT: same keys, same nesting, same value types, same allowed enum values. Re-emit the COMPLETE result (every field), not just the parts you changed.
    - Preserve item ordering and identifiers unless a segmentation fix requires renumbering, in which case renumber consistently.

OUTPUT PROTOCOL

Return EXACTLY ONE JSON object and nothing else — no markdown fences, no commentary. It carries the corrected result under "result" (conforming to the original schema) and your audit under "evaluation".

{
    "action": "stop",
    "status": "success|failure|other",
    "summary": "...",
    "issues": "<issues or null>",
    "reason": "...",
    "result": {
        ... the COMPLETE revised result, conforming exactly to the schema in the ORIGINAL PROMPT ...
    },
    "evaluation": {
        "verdict": "accepted_as_is" | "revised" | "rejected",
        "num_changes": <int>,
        "changes": [
            {
                "location": "<path to the field, e.g. steps[2].operation.type>",
                "action": "fix" | "add" | "remove",
                "from": "<previous value or null>",
                "to": "<new value or null>",
                "evidence": "<the construct/phrase in the input that justifies the change>",
                "confidence": "high" | "medium" | "low"
            }
        ],
        "unresolved_concerns": [ "<suspected but unconfirmable issue + reasoning>" ]
    }
}

CONTROL FIELD GUIDANCE
- "verdict": "accepted_as_is" if you made no changes; "revised" if you corrected one or more fields; "rejected" if the previous result was so wrong it had to be substantially rebuilt.
- "num_changes": the length of "changes".
- "status": "success" if you completed the review; "failure" if you could not (e.g., the input or previous result was unreadable); "other" for partial/ambiguous reviews.
- "summary": one or two sentences on what you found and changed.
- "result": if status is "failure", this may be the unchanged previous result or null.
"""


class AdversarialAgent(BaseSubAgent):
    """
    General adversarial reviewer. Given any expert agent's prompt and its
    previous result (via inputs), plus the specific input artifact under review
    (via the goal), it critiques and revises the result and returns a corrected
    result with an auditable changelog. It is artifact- and taxonomy-agnostic:
    the schema is whatever the ORIGINAL PROMPT defines.
    """

    name = "adversarial"
    description = (
        "A general adversarial reviewer that evaluates and revises the output of any "
        "expert agent. Given the expert agent's prompt and previous result, plus the "
        "specific input that was analyzed (supplied as the goal), it corrects fields the "
        "input contradicts — focusing on interpretive fields (free text, classifications, "
        "decomposition, named entities) that cannot be checked mechanically — and returns "
        "the revised result with a structured changelog."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The task goal, including the specific input artifact the expert agent analyzed (e.g. the script). This is the reviewer's ground truth.",
            },
            "previous_prompt": {
                "type": "string",
                "description": "The prompt the expert agent was given; defines the taxonomy and the exact output schema.",
            },
            "previous_result": {
                "type": "object",
                "description": "The expert agent's output to evaluate and revise (the result object). A pre-serialized JSON string is also accepted.",
            },
            "context": {
                "type": "string",
                "default": "",
                "description": "Optional additional context to pass through to the reviewer.",
            },
            "max_turns": {
                "type": "integer",
                "default": 100,
                "description": "Max turns for the review loop.",
            },
        },
        "required": ["goal", "previous_prompt", "previous_result"],
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
                "description": "The complete revised result, conforming to the original schema.",
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
        goal: str,
        previous_prompt: str,
        previous_result: Any,
        context: str = "",
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Executes the adversarial review loop.

        The reviewer persona/instructions live in the static `adversarial_prompt`.
        The expert agent's prompt and previous result are passed as task context;
        the specific input artifact under review is supplied by the caller in `goal`.
        `previous_result` may be a dict (the bare result object) or a JSON string.
        """
        prev_result_str = (
            previous_result
            if isinstance(previous_result, str)
            else json.dumps(previous_result, indent=2)
        )

        review_context = (
            "You are reviewing the output of a prior expert agent.\n\n"
            "=== ORIGINAL PROMPT (defines the task and the exact output schema) ===\n"
            f"{previous_prompt}\n\n"
            "=== PREVIOUS RESULT (evaluate and revise this) ===\n"
            f"{prev_result_str}\n"
        )
        if context:
            review_context += f"\n=== ADDITIONAL CONTEXT ===\n{context}\n"

        return await self.execute_loop(
            system_prompt=adversarial_prompt,
            goal=goal,
            context=review_context,
            max_turns=max_turns,
            process_callback=process_callback,
        )
