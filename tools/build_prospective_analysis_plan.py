"""Build the checked-in illustrative prospective plan without simulation."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from microtx_sim.causal.analysis_plan import (
    AnalysisEstimandRole,
    CanonicalPopulationInclusionPredicate,
    FixedSeedStoppingRule,
    PlannedPopulationEstimand,
    PopulationMinorFilter,
    PopulationOutcomeMetric,
    PrimaryAggregateRule,
    analysis_plan_harm_weights_sha256,
    build_prospective_analysis_plan,
    load_prospective_analysis_plan,
    verify_loaded_prospective_analysis_plan,
)
from microtx_sim.causal.batch import resolve_policy_run_inputs
from microtx_sim.causal.design import assess_causal_design
from microtx_sim.causal.scenarios import ScenarioId
from microtx_sim.data.lineage import build_profile_input_lineage
from microtx_sim.data.population_evidence import PopulationEstimandRole
from microtx_sim.data.population_execution import (
    population_execution_input_sha256,
    resolve_population_projection_adapter,
)
from microtx_sim.data.profiles import load_profile_bundle
from microtx_sim.metrics.population_estimands import (
    PopulationInclusionField,
    PopulationInclusionRule,
    PopulationInclusionTiming,
    PopulationPeriodSemantics,
)
from microtx_sim.outputs.metric_contracts import metric_contract_registry_sha256
from microtx_sim.outputs.schema import PROSPECTIVE_ANALYSIS_SCHEMA_SHA256
from microtx_sim.outputs.writers import write_text_atomic
from microtx_sim.policy_config import load_policy_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "policy_prospective.toml"
PLAN_PATH = REPOSITORY_ROOT / "inputs" / "prospective-analysis-plan.json"


def build_plan() -> object:
    config = load_policy_config(CONFIG_PATH)
    if config.population is None:
        raise RuntimeError("prospective configuration must select projected population")
    if not config.output.include_player_rows:
        raise RuntimeError("prospective configuration must retain player outcomes")
    if (
        config.population.evidence_bundle_path is None
        or config.population.source_registry_path is None
    ):
        raise RuntimeError(
            "prospective configuration must bind population evidence and sources"
        )
    profiles = load_profile_bundle(
        sources_path=config.population.source_registry_path,
        source_bundle_path=None,
        population_bundle_path=config.population.evidence_bundle_path,
        campaign=False,
    )
    profile_lineage = build_profile_input_lineage(
        profiles.country_profiles,
        profile_bundle=profiles,
    )
    adapter = resolve_population_projection_adapter(
        config.population,
        profiles,
        player_count=config.batch.player_count,
    )
    run_inputs = resolve_policy_run_inputs(
        harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
    )
    inclusion = CanonicalPopulationInclusionPredicate(
        rule=PopulationInclusionRule(
            rule_id="all.illustrative.projected.players.v1",
            description=(
                "Include every player assigned to the attested illustrative "
                "projected population using pre-treatment fields only."
            ),
            source_fields=tuple(
                sorted(PopulationInclusionField, key=lambda item: item.value)
            ),
            timing=PopulationInclusionTiming.PRETREATMENT,
            evidence_role=PopulationEstimandRole.CALIBRATION,
        ),
        jurisdiction_codes=("BE", "JP", "KR", "UK"),
        age_min_inclusive=10,
        age_max_exclusive=70,
        minor_filter=PopulationMinorFilter.ANY,
        monthly_disposable_income_band_ids=(
            "runtime.personal.monthly.income.all",
        ),
        household_type_ids=("household.all",),
        gaming_states=(),
        payer_history_states=(),
    )
    primary = PlannedPopulationEstimand(
        estimand_id="primary.composite-harm.baseline-vs-safe.v1",
        role=AnalysisEstimandRole.PRIMARY,
        reference_scenario_id=ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
        comparison_scenario_id=ScenarioId.BASELINE_F2P,
        outcome_metric=PopulationOutcomeMetric.COMPOSITE_HARM,
        metric_contract_id="player_outcomes.csv:composite_harm",
        inclusion_predicate=inclusion,
        period=PopulationPeriodSemantics(
            period_start=date(2026, 1, 1),
            period_end=date(2026, 1, 14),
            description=(
                "Fourteen-day illustrative simulator horizon; calendar anchor "
                "is declarative and not empirically bound."
            ),
        ),
    )
    return build_prospective_analysis_plan(
        plan_id="illustrative.prospective.composite-harm.baseline-vs-safe.v2",
        expected_causal_design_sha256=assess_causal_design(
            config.batch.scenarios
        ).design_sha256(),
        expected_batch_spec_sha256=config.batch.snapshot_sha256(),
        expected_model_inputs_sha256=run_inputs.snapshot_sha256(),
        expected_population_input_sha256=population_execution_input_sha256(
            adapter
        ),
        expected_profile_input_sha256=profile_lineage.fingerprint_sha256,
        expected_metric_contract_sha256=metric_contract_registry_sha256(),
        expected_harm_weights_sha256=analysis_plan_harm_weights_sha256(
            run_inputs.harm_weights
        ),
        expected_output_profile_sha256=PROSPECTIVE_ANALYSIS_SCHEMA_SHA256,
        stopping_rule=FixedSeedStoppingRule(seeds=config.batch.seeds),
        estimands=(primary,),
        declared_harm_weights=run_inputs.harm_weights,
        primary_aggregate_rule=PrimaryAggregateRule(
            positive_result_interpretation=(
                "baseline_f2p has greater population-weighted composite "
                "simulated harm than safe_fixed_price_subscription"
            ),
            negative_result_interpretation=(
                "baseline_f2p has lower population-weighted composite "
                "simulated harm than safe_fixed_price_subscription"
            ),
        ),
    )


def main() -> None:
    plan = build_plan()
    text = json.dumps(
        plan.snapshot(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    write_text_atomic(PLAN_PATH, text)
    loaded = load_prospective_analysis_plan(PLAN_PATH)
    verify_loaded_prospective_analysis_plan(loaded)
    if loaded.plan != plan:
        raise RuntimeError("written plan differs from the programmatic builder result")
    print(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "plan_path": str(PLAN_PATH),
                "plan_sha256": plan.plan_sha256,
                "file_sha256": loaded.file_sha256,
                "campaign_ready": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
