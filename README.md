# fractale agents

> Agents to use with [fractale](https://github.com/converged-computing/fractale), agentic state-machine orchestrator for Science

[![PyPI version](https://badge.fury.io/py/fractale-agents.svg)](https://badge.fury.io/py/fractale-agents)

![https://github.com/converged-computing/fractale/raw/main/img/fractale-small.png](https://github.com/converged-computing/fractale/raw/main/img/fractale-small.png)

A sub-agent is a single step in a state machine that can call other sub-agents, tools, or prompts, and respond dynamically to work on a scoped task. Sub-agents are created in order to map our expertise and logic into a more controlled execution or interaction with an LLM.

## Agents

The following agents are availble.

| Name | Description | Path |
|----- | ----------- | ---- |
| flux-operator | expert to deploy Flux Operator MiniClusters to Kubernetes | fractale_agents.kubernetes.FluxOperatorAgent |
| flux-build | optimized to build containers for the Flux Operator | fractale_agents.kubernetes.FluxBuildAgent |
| result_parse | Parse specific metrics from output logs | fractale_agents.parsers.ResultParserAgent |
| optimize | General optimization agent | fractale_agents.optimize.OptimizeAgent |

The general prompt agent is provisioned by fractale directly, `fractale.agents.general.PromptAgent`.
Would you like to see an expert added? Please open an issue and let us know.

## Usage

A sub-agent is a specialized expert that can be imported and used in a fractale workflow. You can register agents on the fly, or via a configuration file.

```bash
# Register a sub-agent tool on the fly
fractale prompt -t fractale_agents.kubernetes.FluxOperatorAgent Deploy lammps to Kubernetes using the Flux Operator
```
Or write the tool into a registry file:

```yaml
# sub-agents.yaml
tools:
  - path: fractale_agents.kubernetes.FluxOperatorAgent
```

And then:

```bash
fractale prompt -r ./sub-agents.yaml Deploy lammps to Kubernetes using the Flux Operator
```

The agents here rely on the fractale backend.

## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614
