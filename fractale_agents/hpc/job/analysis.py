import json
from typing import Any, Awaitable, Callable, Dict, Optional

import fractale_agents.utils as utils
from fractale_agents.agent import BaseSubAgent
from fractale_agents.logger import logger

job_prompt = """You are an expert High-Performance Computing (HPC) engineer. Your task is to analyze a provided job script and extract its characteristics into a strictly formatted JSON object based on a specific multi-dimensional labeling taxonomy. You must identify and extract the specific applications utilized within the script (e.g., the exact workflow manager, container engine, scientific application, or programming language) and map them to the corresponding fields in the JSON schema. Break down the scripts into one or more steps, where each step performs a specific task or set of operations. For each step extract the following information.

TAXONOMY AND DEFINITION

Job Description
"description" (A short textual description of the operations executed by the full job)
"domain" (Infer the science domain of the application based on the primary software and specifications, e.g., "math", "ai", "data science", "chemistry", "physics", "biology")

Step Identification
"step_id" (incremental id based on the order of the step in the job pipeline)
Orchestration (Select exactly one type and identify the tool)
    - "workflow_managed": The script is generated or managed by tools like nextflow, cylc, fireworks, snakemake, etc. Look for specific tool headers or variables. Include the specific workflow manager in tool_detected.
    - "job_array": The script utilizes array variables (e.g., $PBS_ARRAY_INDEX). tool_detected should be null.
    - "standalone": A standard, standalone job submission without arrays or workflow managers. tool_detected should be null.
Execution Environment (Select exactly one type and identify the tool)
    - "containerized": The workload runs inside a container. Look for commands like apptainer exec, singularity run, enroot start, docker run, podman, etc. Include the specific container engine in tool_detected.
    - "native": The workload runs directly on the host OS or via standard environment modules (e.g., module load). tool_detected should be null.
Workload Class (Select exactly one primary type, identify the software, and define sub-intents where applicable)
"type"
    - "simulation": Heavy scientific computations (e.g., VASP, GROMACS, OpenFOAM, Gaussian, LAMMPS).
    - "ai_ml" : Deep learning training, inference, or tuning frameworks (e.g., PyTorch, TensorFlow, JAX, Deepspeed).
    - "data_analytics": Data processing, statistical runtimes, or math systems (e.g., large R scripts, Pandas/Spark tracks, MATLAB).
    - "utility": Structural or housekeeping commands (e.g., data transfers via `rsync`/`cp`, file archiving via `tar`, or setup/cleanup bash routines).

"primary_software" (The the specific scientific application (e.g., VASP, GROMACS, OpenFOAM, Gaussian, LAMMPS), or  programming language (e.g., python, R, java, matlab) used)
"target_file" (Return the executed file if the "primary_software" is a programming language, leave empty otherwise)
"parameters" (The list of parameters passed to the executable and relevant environment variables with values)
"step_description" (A short textual description of the operations executed by the step)

EXTRACTION RULES

Output ONLY valid JSON. Do not include markdown formatting like ```json, greetings, or explanations.
If a specific tool, framework, or model is not found, use null for strings or an empty array [] for lists.
Be case-insensitive when searching, but preserve the standard capitalization in the output.

OUTPUT PROTOCOL

You MUST return exactly ONE JSON object and nothing else. This single object is the control/stop signal for the orchestrator, and it carries the full extraction taxonomy nested inside its "result" field. Do not emit the extraction object separately — it lives only inside "result".

{{
    "action": "stop",
    "status": "success|failure|other",
    "summary": "...",
    "command": "<command>",
    "issues": "<issues>",
    "job_id": "<job_id>",
    "reason": "...",
    "result": {{
        "description": "string",
        "domain": "string or list of string",
        "steps":
        [
            {{
                "step_id": "int",
                "orchestration": {{
                    "type": "workflow_managed" | "job_array" | "standalone",
                    "tool_detected": "string or null"
                }},
                "execution_environment": {{
                    "type": "containerized" | "native",
                    "tool_detected": "string or null"
                }},
                "workload_class": {{
                    "type": "simulation" | "ai_ml" | "data_analytics" | "utility",
                    "primary_software": "string (e.g., 'GROMACS', 'python', 'apptainer') or null",
                    "executed_file": "string or null",
                    "parameters": "list of string or null",
                    "step_description": "string"
                }}
            }}
        ]
    }}
}}

CONTROL FIELD GUIDANCE
- "action": always "stop".
- "status": "success" if the job script was parsed and the taxonomy extracted; "failure" if it could not be analyzed; "other" for partial or ambiguous results.
- "summary": a brief natural-language summary of what the job does and what was extracted.
- "command": the submission/invocation command for the job (or the script path), if applicable.
- "issues": any problems encountered during analysis (e.g., unreadable sections, missing context), or null.
- "job_id": the identifier of the job being analyzed, if available.
- "reason": justification for the chosen status.
- "result": the full extraction taxonomy object described above. If status is "failure", "result" may be null.
"""

script_prompt = """You are an expert High-Performance Computing (HPC) engineer and code analyst. Your task is to analyze a provided script (e.g., a Python, R, Julia, MATLAB, or shell script) that is executed within an HPC job and extract its characteristics into a strictly formatted JSON object based on a specific multi-dimensional labeling taxonomy. You must identify the programming language, the specific modules/libraries imported, and the scientific applications, models, or algorithms used within the script, and map them to the corresponding fields in the JSON schema. Break down the script into one or more steps, where each step performs a specific task or set of operations (e.g., data loading, preprocessing, model definition, training, inference, analysis, visualization). For each step extract the following information.

TAXONOMY AND DEFINITIONS

Script Description
"description" (A short textual description of the operations executed by the full script)
"domain" (Infer the science domain based on the primary libraries and operations, e.g., "math", "ai", "data science", "chemistry", "physics", "biology")
"language" (The programming language of the script, e.g., "python", "R", "julia", "matlab", "bash")
"dependencies" (The complete list of modules, libraries, or packages imported/loaded by the script, e.g., ["numpy", "pandas", "torch", "sklearn"])

Step Identification
"step_id" (incremental id based on the order of the step in the script)

Operation (Select exactly one type and identify the primary driver)
    - "data_io": Reading or writing data/files (e.g., pd.read_csv, np.load, open(), torch.save, writing figures/checkpoints).
    - "preprocessing": Cleaning, transforming, normalizing, encoding, or feature engineering of data.
    - "modeling": Defining a model architecture or instantiating an estimator/algorithm (e.g., nn.Module definitions, sklearn estimator setup, regression specification).
    - "training": Fitting or optimizing a model (e.g., model.fit, training loops, optimizer steps).
    - "inference": Generating predictions or running a trained model (e.g., model.predict, forward passes at eval time).
    - "analysis": Statistical analysis, numerical computation, or metric calculation (e.g., scipy.stats, computing accuracy/RMSE, aggregations).
    - "visualization": Producing plots, charts, or figures (e.g., matplotlib, seaborn, ggplot).
    - "utility": Setup, configuration, logging, argument parsing, environment setup, or housekeeping.
    "primary_function" (The main function, method, or call that drives the step, e.g., "model.fit", "pd.read_csv", or null)

Computation (Identify the scientific/ML content of the step where applicable)
    "models" (The specific model architectures or algorithms used, e.g., ["ResNet50"], ["RandomForestClassifier"], ["linear regression"], or null)
    "frameworks" (The specific libraries/frameworks driving the computation in this step, e.g., ["torch"], ["sklearn"], ["scipy"], or null)
    "parameters" (The list of hyperparameters, function arguments, and relevant variables with values, e.g., ["epochs=50", "lr=0.001", "n_estimators=100"], or null)

Data (Identify the data flow of the step)
    "inputs" (The files, datasets, or variables consumed by the step, or null)
    "outputs" (The files, artifacts, or variables produced by the step, e.g., model checkpoints, output CSVs, figures, or null)

"step_description" (A short textual description of the operations executed by the step)

EXTRACTION RULES

Analyze the script and produce valid JSON conforming to the schema below.
If a specific tool, framework, model, or value is not found, use null for strings or an empty array [] for lists.
Be case-insensitive when searching, but preserve the standard capitalization in the output (e.g., "PyTorch", "NumPy", "GROMACS").
Group consecutive lines that serve the same logical purpose into a single step rather than emitting one step per line.

EXTRACTION OUTPUT JSON SCHEMA

{{
    "description": "string",
    "domain": "string or list of string",
    "language": "string",
    "dependencies": ["list of string"],
    "steps":
    [
        {{
            "step_id": "int",
            "operation": {{
                "type": "data_io" | "preprocessing" | "modeling" | "training" | "inference" | "analysis" | "visualization" | "utility",
                "primary_function": "string or null"
            }},
            "computation": {{
                "models": "list of string or null",
                "frameworks": "list of string or null",
                "parameters": "list of string or null"
            }},
            "data": {{
                "inputs": "list of string or null",
                "outputs": "list of string or null"
            }},
            "step_description": "string"
        }}
    ]
}}

COMPLETION PROTOCOL

The extraction JSON above is your analysis deliverable. After producing it, you MUST return ONE final JSON object — as the last thing you emit — to signal the orchestrator that the task is complete:

{{"action": "stop", "status": "success|failure|other", "summary": "...", "command": "<command>", "issues": "<issues>", "result": "<result>", "reason": "..."}}

- "result": The final analysis result object shown above.
- "status": "success" if the script was parsed and the taxonomy extracted; "failure" if the script could not be analyzed; "other" for partial or ambiguous results.
- "summary": a brief natural-language summary of what the script does and what was extracted.
- "command": the command/invocation used to run or analyze the script (or the script path), if applicable.
- "issues": any problems encountered during analysis (e.g., unreadable sections, missing context), or null.
- "reason": justification for the chosen status.

Output only the single JSON object. Do not include markdown formatting like ```json, greetings, or explanations.
"""


class ScriptAnalysisAgent(BaseSubAgent):
    """
    Agent optimized to analyze code / scripts associated with jobspecs.
    """

    name = "script-analysis"
    description = "An expert agent that takes an input script and is able to analyze it for imports, goals, software parameters, models, and other metadata, breaking into orchestration steps or logic."
    input_schema = {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "The script and details provided by the user.",
            },
            "max_turns": {
                "type": "integer",
                "default": 100,
                "description": "Max turns for the discovery and monitoring loop.",
            },
        },
        "required": ["script"],
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
            "result": {
                "type": "dict",
                "description": "The result object",
            },
        },
        "required": ["status", "summary", "result"],
    }

    async def __call__(
        self,
        script: str,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Executes the script analysis loop.
        """
        goal = f"Analyze this script."
        context = f"The following script is provided: '{script}'. "
        return await self.execute_loop(
            system_prompt=script_prompt,
            goal=goal,
            context=context,
            max_turns=max_turns,
            process_callback=process_callback,
        )


class JobAnalysisAgent(BaseSubAgent):
    """
    Agent optimized to analyze HPC batch scripts for intent, steps, and metadata."
    """

    name = "job-analysis"
    description = "An expert agent that is optimized to analyze HPC batch scripts for intent, steps, and metadata."
    input_schema = {
        "type": "object",
        "properties": {
            "requirement": {
                "type": "string",
                "description": "The user requirement.",
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
            "result": {
                "type": "dict",
                "description": "The result object",
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
        },
        "required": ["status", "summary", "result"],
    }

    async def __call__(
        self,
        requirement: str,
        max_turns: int = 100,
        process_callback: Optional[
            Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Executes the job analysis loop.
        """
        context = f"The following requirements are provided: '{requirement}'. "
        return await self.execute_loop(
            system_prompt=job_prompt,
            goal="Analyze this job specification",
            context=context,
            max_turns=max_turns,
            process_callback=process_callback,
        )
