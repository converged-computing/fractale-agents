__version__ = "0.0.1"
AUTHOR = "Vanessa Sochat"
AUTHOR_EMAIL = "vsoch@users.noreply.github.com"
NAME = "fractale-agents"
PACKAGE_URL = "https://github.com/converged-computing/fractale-agents"
KEYWORDS = "cluster, mcp, agents, tools, orchestration, transformer, jobspec, flux"
DESCRIPTION = "Agents to support fractale agentic framework"
LICENSE = "LICENSE"


################################################################################
# TODO vsoch: refactor this to use newer pyproject stuff.

INSTALL_REQUIRES = (
    ("rich", {"min_version": None}),
    ("fractale", {"min_version": None}),
)
TESTS_REQUIRES = (("pytest", {"min_version": "4.6.2"}),)

INSTALL_REQUIRES_ALL = INSTALL_REQUIRES + TESTS_REQUIRES
