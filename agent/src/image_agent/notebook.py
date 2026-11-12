"""
Public API for Jupyter Notebook integration.

Models are NOT loaded on import. Generation only starts when explicitly called.
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Optional

from image_agent.orchestration.checkpoint import save_state
from image_agent.runtime.logging import get_logger
from image_agent.schemas.execution import AgentState, RunStatus, RunStatusEnum

logger = get_logger(__name__)


class ImageAgentSession:
    """
    High-level session object for Notebook usage.

    Usage::

        session = ImageAgentSession(
            prompt="./dataset/request.md",
            backend="flux2",
        )
        plan = session.plan()   # runs planner only
        session.show_plan()
        result = session.run()  # runs full pipeline
        session.show_result()
    """

    def __init__(
        self,
        prompt: str,
        dataset_root: str = "/workspace/dataset",
        runs_dir: str = "/workspace/agent/runs",
        backend: str = "flux2",
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        resume_run_id: Optional[str] = None,
    ) -> None:
        self.prompt = prompt
        self.dataset_root = dataset_root
        self.runs_dir = runs_dir
        self.backend = backend
        self.seed = seed
        self.width = width
        self.height = height
        self.resume_run_id = resume_run_id

        self._state: Optional[AgentState] = None
        self._run_id: Optional[str] = None

    # --- Planning ---

    def plan(self) -> Optional[object]:
        """
        Run the planner only (validate_input + analyze_and_plan + validate_plan).

        Returns the ScenePlan or None on failure.
        Does NOT start generation.
        """
        from image_agent.orchestration.nodes import (
            analyze_and_plan,
            validate_input,
            validate_plan,
        )
        self._init_state()
        assert self._state is not None

        self._state = validate_input(self._state)
        if self._state.status == RunStatusEnum.failed:
            self._print_error()
            return None

        self._state = analyze_and_plan(self._state)
        if self._state.status in (RunStatusEnum.failed, RunStatusEnum.user_required):
            self._print_error()
            return None

        self._state = validate_plan(self._state)
        return self._state.scene_plan

    def show_plan(self) -> None:
        """Display the current ScenePlan in the Notebook."""
        if self._state is None or self._state.scene_plan is None:
            print("No plan available. Run session.plan() first.")
            return
        try:
            from IPython.display import display, Markdown
            plan = self._state.scene_plan
            md = f"""## Scene Plan

**Subject:** {plan.subject}
**Style:** {plan.style}
**Scene:** {plan.scene}
**Composition:** {plan.composition}

### Constraints
- **Must preserve:** {', '.join(plan.constraints.must_preserve) or '—'}
- **May change:** {', '.join(plan.constraints.may_change) or '—'}
- **Must design:** {', '.join(plan.constraints.must_design) or '—'}

### Contradictions
{chr(10).join('- ' + c for c in plan.constraints.contradictions) if plan.constraints.contradictions else '—'}

### Generation Caption
> {plan.generation_caption}

### Jobs ({len(plan.jobs.jobs)})
{chr(10).join(f"- **{j.type.value}** — {j.purpose}" for j in plan.jobs.jobs)}

> ⚠️ AI-inferred items: {', '.join(plan.constraints.inferred) or '—'}
"""
            display(Markdown(md))
        except ImportError:
            print(json.dumps(self._state.scene_plan.model_dump(mode="json"), indent=2, ensure_ascii=False))

    # --- Full run ---

    def run(self) -> Optional[AgentState]:
        """
        Run the full generation pipeline.

        Call plan() first if you want to review the plan before generation.
        """
        from image_agent.orchestration.graph import run_agent

        self._state = run_agent(
            markdown_path=self.prompt,
            dataset_root=self.dataset_root,
            runs_dir=self.runs_dir,
            backend=self.backend,
            seed=self.seed,
            width=self.width,
            height=self.height,
            resume_run_id=self.resume_run_id,
            initial_state=self._state if self._state and self._state.scene_plan else None,
        )
        return self._state

    def show_result(self) -> None:
        """Display the final result in the Notebook."""
        if self._state is None:
            print("No result. Run session.run() first.")
            return

        status = self._state.status.value
        print(f"Status: {status}")

        if self._state.error:
            print(f"Error: {self._state.error}")

        if self._state.user_message:
            print(f"Message: {self._state.user_message}")

        if self._state.final_asset_id:
            manifest = self._state.asset_manifest
            if manifest:
                asset = manifest.by_id(self._state.final_asset_id)
                if asset and asset.container_path:
                    self._show_image(asset.container_path, "Final result")

        # Show upscale info
        if self._state.upscale_result:
            ur = self._state.upscale_result
            print(f"\nUpscaled: {ur.input_width}x{ur.input_height} → {ur.output_width}x{ur.output_height}")
            if ur.oom_fallback_used:
                print("⚠️ OOM fallback tile size was used during upscale")

    def show_constraints(self) -> None:
        """Highlight what is AI-inferred vs user-specified."""
        if self._state is None or self._state.scene_plan is None:
            return
        plan = self._state.scene_plan
        print("=== User-specified (must_preserve) ===")
        for item in plan.constraints.must_preserve:
            print(f"  ✓ {item}")
        print("\n=== AI-inferred (not from user Markdown) ===")
        for item in plan.constraints.inferred:
            print(f"  ~ {item}")
        if plan.constraints.uncertainty:
            print("\n=== Uncertainty ===")
            for item in plan.constraints.uncertainty:
                print(f"  ? {item}")

    def download_run(self) -> None:
        """Create a ZIP of the run directory for download."""
        if self._state is None:
            print("No run to download.")
            return
        run_dir = Path(self._state.run_dir)
        zip_path = run_dir.parent / f"{run_dir.name}.zip"
        import shutil
        shutil.make_archive(str(run_dir), "zip", str(run_dir.parent), run_dir.name)
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(zip_path)))
        except ImportError:
            print(f"Run saved to: {zip_path}")

    # --- Internal ---

    def _init_state(self) -> None:
        run_id = (
            self.resume_run_id
            or f"run_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        run_dir = str(Path(self.runs_dir) / run_id)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._state = AgentState(
            run_id=run_id,
            run_dir=run_dir,
            markdown_path=self.prompt,
            dataset_root=self.dataset_root,
            backend=self.backend,
            seed=self.seed,
            width=self.width,
            height=self.height,
            status=RunStatusEnum.pending,
        )

    def _print_error(self) -> None:
        if self._state:
            msg = self._state.error or self._state.user_message
            print(f"[image_agent] Error: {msg}")

    def _show_image(self, path: str, caption: str = "") -> None:
        try:
            from IPython.display import Image as IPImage, display
            print(caption)
            display(IPImage(path))
        except (ImportError, Exception):
            print(f"{caption}: {path}")
