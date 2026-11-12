"""
LangGraph StateGraph definition for the image generation agent.

The domain logic lives in nodes.py; this module only wires edges.
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Optional

from image_agent.errors import MaxStepsExceeded
from image_agent.orchestration.checkpoint import load_state, save_state
from image_agent.orchestration.nodes import (
    analyze_and_plan,
    critic_node,
    decide,
    finish,
    generate_auxiliary,
    generate_main,
    inspect_auxiliary,
    revise,
    review_plan,
    user_required,
    validate_input,
    validate_plan,
)
from image_agent.runtime.logging import get_logger
from image_agent.schemas.execution import AgentState, RunStatusEnum

logger = get_logger(__name__)


def build_graph():
    """
    Build and return a compiled LangGraph StateGraph.

    Uses low-level StateGraph API — no high-level agent helpers.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        raise ImportError("langgraph is required for orchestration") from exc

    def _wrap(fn):
        """Wrap a node function to handle AgentState <-> dict conversion."""
        def wrapped(state_dict: dict) -> dict:
            state = AgentState.model_validate(state_dict)
            new_state = fn(state)
            return new_state.model_dump(mode="json")
        wrapped.__name__ = fn.__name__
        return wrapped

    def _decide_wrapped(state_dict: dict) -> str:
        state = AgentState.model_validate(state_dict)
        return decide(state)

    def _continue_to(next_node: str):
        def route(state_dict: dict) -> str:
            state = AgentState.model_validate(state_dict)
            if state.status in (RunStatusEnum.failed, RunStatusEnum.user_required):
                return "stop"
            return next_node
        return route

    def _entry_route(state_dict: dict) -> str:
        """Continue a checkpoint from the node after its last completed node."""
        state = AgentState.model_validate(state_dict)
        if state.status == RunStatusEnum.completed:
            return "end"
        if state.status in (RunStatusEnum.failed, RunStatusEnum.user_required):
            return "stop"
        next_by_node = {
            None: "validate_input",
            "validate_input": "analyze_and_plan",
            "analyze_and_plan": "validate_plan",
            "validate_plan": "generate_auxiliary",
            "review_plan": "generate_auxiliary",
            "generate_auxiliary": "inspect_auxiliary",
            "inspect_auxiliary": "generate_main",
            "generate_main": "critic",
            "revise": "generate_main",
        }
        if state.current_node == "critic":
            return decide(state)
        return next_by_node.get(state.current_node, "validate_input")

    # Build the graph
    from langgraph.graph import StateGraph

    graph = StateGraph(dict)  # dict-based state (AgentState JSON)

    graph.add_node("validate_input", _wrap(validate_input))
    graph.add_node("analyze_and_plan", _wrap(analyze_and_plan))
    graph.add_node("validate_plan", _wrap(validate_plan))
    graph.add_node("review_plan", _wrap(review_plan))
    graph.add_node("generate_auxiliary", _wrap(generate_auxiliary))
    graph.add_node("inspect_auxiliary", _wrap(inspect_auxiliary))
    graph.add_node("generate_main", _wrap(generate_main))
    graph.add_node("critic", _wrap(critic_node))
    graph.add_node("finish", _wrap(finish))
    graph.add_node("revise", _wrap(revise))
    graph.add_node("user_required", _wrap(user_required))
    graph.add_node("stop", _wrap(user_required))

    # Fresh runs start at validate_input. Checkpoints continue after the last
    # successfully persisted node, including a Notebook-reviewed plan.
    graph.set_conditional_entry_point(
        _entry_route,
        {
            "validate_input": "validate_input",
            "analyze_and_plan": "analyze_and_plan",
            "validate_plan": "validate_plan",
            "generate_auxiliary": "generate_auxiliary",
            "inspect_auxiliary": "inspect_auxiliary",
            "generate_main": "generate_main",
            "critic": "critic",
            "revise": "revise",
            "finish": "finish",
            "user_required": "user_required",
            "stop": "stop",
            "end": END,
        },
    )

    # Every stage stops immediately on failed/user_required status.
    for node, target in (
        ("validate_input", "analyze_and_plan"),
        ("analyze_and_plan", "validate_plan"),
        ("validate_plan", "review_plan"),
        ("review_plan", "generate_auxiliary"),
        ("generate_auxiliary", "inspect_auxiliary"),
        ("inspect_auxiliary", "generate_main"),
        ("generate_main", "critic"),
        ("revise", "generate_main"),
    ):
        graph.add_conditional_edges(
            node,
            _continue_to(target),
            {target: target, "stop": "stop"},
        )

    # Conditional branching from critic -> decide
    graph.add_conditional_edges(
        "critic",
        _decide_wrapped,
        {
            "finish": "finish",
            "revise": "revise",
            "user_required": "user_required",
        },
    )

    # Terminal nodes
    graph.add_edge("finish", END)
    graph.add_edge("user_required", END)
    graph.add_edge("stop", END)

    return graph.compile()


def run_agent(
    markdown_path: str,
    dataset_root: str = "/workspace/dataset",
    runs_dir: str = "/workspace/agent/runs",
    backend: str = "flux2",
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    resume_run_id: Optional[str] = None,
    initial_state: Optional[AgentState] = None,
) -> AgentState:
    """
    Run the full agent pipeline and return the final AgentState.

    Pass resume_run_id to continue from a saved checkpoint.
    """
    if initial_state is not None:
        state = initial_state
        run_id = state.run_id
        run_dir = state.run_dir
    elif resume_run_id:
        run_id = resume_run_id
        run_dir = str(Path(runs_dir) / run_id)
        state = load_state(run_dir)
        if state is None:
            logger.warning("resume_failed", extra={"run_id": run_id})
            state = _new_state(run_id, run_dir, markdown_path, dataset_root, backend, seed, width, height)
        else:
            logger.info("resuming_run", extra={"run_id": run_id, "node": state.current_node})
    else:
        run_id = f"run_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = str(Path(runs_dir) / run_id)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        state = _new_state(run_id, run_dir, markdown_path, dataset_root, backend, seed, width, height)

    graph = build_graph()
    state_dict = state.model_dump(mode="json")

    try:
        final_dict = graph.invoke(state_dict)
    except MaxStepsExceeded as exc:
        logger.error("max_steps_exceeded", extra={"error": str(exc)})
        state = state.model_copy(update={"status": RunStatusEnum.failed, "error": str(exc)})
        save_state(state, run_dir)
        return state
    except Exception as exc:
        logger.exception("agent_failed", extra={"error": str(exc)})
        state = state.model_copy(update={"status": RunStatusEnum.failed, "error": str(exc)})
        save_state(state, run_dir)
        return state

    return AgentState.model_validate(final_dict)


def _new_state(
    run_id: str,
    run_dir: str,
    markdown_path: str,
    dataset_root: str,
    backend: str,
    seed: Optional[int],
    width: int,
    height: int,
) -> AgentState:
    return AgentState(
        run_id=run_id,
        run_dir=run_dir,
        markdown_path=markdown_path,
        dataset_root=dataset_root,
        backend=backend,
        seed=seed,
        width=width,
        height=height,
        status=RunStatusEnum.pending,
    )
