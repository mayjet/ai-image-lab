"""Hard limits for the agent graph, sourced from configs/agent.yaml."""

from image_agent.runtime.config import load_config

_CFG = load_config("agent.yaml").get("orchestration", {})

MAX_GRAPH_STEPS: int = int(_CFG.get("max_graph_steps", 20))
MAX_AUXILIARY_JOBS: int = int(_CFG.get("max_auxiliary_jobs", 3))
MAX_GENERATION_CANDIDATES: int = int(_CFG.get("max_generation_candidates", 2))
MAX_GENERATION_REVISIONS: int = int(_CFG.get("max_generation_revisions", 2))
MAX_PLAN_REPAIRS: int = int(_CFG.get("max_plan_repairs", 1))
MAX_UPSCALE_ATTEMPTS: int = int(_CFG.get("max_upscale_attempts", 2))
MAX_REFERENCES_PER_JOB: int = int(_CFG.get("max_references_per_job", 4))
