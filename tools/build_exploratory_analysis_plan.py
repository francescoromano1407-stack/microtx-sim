"""Build the non-empirical exploratory sidecar without running simulations."""

from __future__ import annotations

import json
from pathlib import Path

from microtx_sim.causal.analysis_plan import (
    build_exploratory_analysis_plan,
    load_exploratory_analysis_plan,
    load_prospective_analysis_plan,
    verify_exploratory_analysis_plan_parent,
    verify_loaded_exploratory_analysis_plan,
    verify_loaded_prospective_analysis_plan,
)
from microtx_sim.outputs.writers import write_text_atomic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARENT_PLAN_PATH = (
    REPOSITORY_ROOT / "inputs" / "prospective-analysis-plan-amendment-v3.json"
)
EXPLORATORY_PLAN_PATH = (
    REPOSITORY_ROOT / "inputs" / "exploratory-synthetic-analysis-plan-v1.json"
)
EXPLORATORY_PLAN_ID = (
    "illustrative.exploratory.synthetic.composite-harm.baseline-vs-safe.v1"
)


def build_plan():
    """Re-attest the exact v3 parent and derive its exploratory sidecar."""

    parent = verify_loaded_prospective_analysis_plan(
        load_prospective_analysis_plan(PARENT_PLAN_PATH)
    )
    return build_exploratory_analysis_plan(
        parent,
        plan_id=EXPLORATORY_PLAN_ID,
    )


def main() -> None:
    plan = build_plan()
    rendered = (
        json.dumps(
            plan.snapshot(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_text_atomic(EXPLORATORY_PLAN_PATH, rendered)
    loaded = verify_loaded_exploratory_analysis_plan(
        load_exploratory_analysis_plan(EXPLORATORY_PLAN_PATH)
    )
    parent = verify_loaded_prospective_analysis_plan(
        load_prospective_analysis_plan(PARENT_PLAN_PATH)
    )
    verify_exploratory_analysis_plan_parent(loaded.plan, parent)
    if loaded.plan != plan:
        raise RuntimeError("written exploratory plan differs from its builder")
    print(
        json.dumps(
            {
                "campaign_ready": False,
                "execution_status": "NOT_EXECUTED",
                "file_sha256": loaded.file_sha256,
                "plan_id": plan.plan_id,
                "plan_path": str(EXPLORATORY_PLAN_PATH),
                "plan_sha256": plan.plan_sha256,
                "preregistered": False,
                "simulation_execution_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
