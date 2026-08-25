"""Exhaustive semantic and provenance contracts for versioned output columns.

The registry describes every CSV column, not only numeric outcomes.  Identifier
and design columns make the table grain explicit; derived columns additionally
carry a versioned transformation recipe and upstream lineage.  All current
contracts are deliberately ``SYNTHETIC`` and therefore fail the campaign gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Final, Mapping

from ..metrics.reporting import REPEATED_SEED_METRIC_STEMS
from ..types import ProvenanceStatus
from .schema import OUTPUT_SCHEMA_VERSION, TABLE_COLUMNS


METRIC_CONTRACT_SCHEMA_VERSION: Final[str] = "1.0"
METRIC_RECIPE_SOURCE_VERSION: Final[str] = (
    "microtx-sim-output-metric-recipes/1.0"
)


class MetricContractValidationError(ValueError):
    """Raised when output contracts are incomplete or scientifically unsafe."""


class MetricRole(str, Enum):
    """The semantic role played by a column in its table."""

    IDENTIFIER = "identifier"
    DESIGN = "design"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class OutputUnit:
    """Structured unit metadata, including powers used by variances."""

    quantity: str
    symbol: str
    exponent: int = 1
    money_basis: str | None = None

    def __post_init__(self) -> None:
        if not self.quantity.strip() or not self.symbol.strip():
            raise MetricContractValidationError("output units must be named")
        if isinstance(self.exponent, bool) or not isinstance(self.exponent, int):
            raise MetricContractValidationError("unit exponent must be an integer")
        if self.exponent <= 0:
            raise MetricContractValidationError("unit exponent must be positive")
        if self.quantity == "money" and not self.money_basis:
            raise MetricContractValidationError("money units need an explicit basis")
        if self.quantity != "money" and self.money_basis is not None:
            raise MetricContractValidationError(
                "only money units may declare a money basis"
            )

    def squared(self) -> OutputUnit:
        """Return the algebraic square used by a variance contract."""

        return OutputUnit(
            quantity=self.quantity,
            symbol=self.symbol,
            exponent=self.exponent * 2,
            money_basis=self.money_basis,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "quantity": self.quantity,
            "symbol": self.symbol,
            "exponent": self.exponent,
            "money_basis": self.money_basis,
        }


@dataclass(frozen=True, slots=True)
class OutputMetricContract:
    """One immutable column-level transformation and provenance contract."""

    artifact: str
    column: str
    role: MetricRole
    storage_type: str
    nullable: bool
    description: str
    unit: OutputUnit
    period: str
    population_base: str
    condition: str
    recipe_id: str
    recipe_version: str
    source_version: str
    source_retrieved_on: date | None
    implementation: str
    formula: str
    inputs: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    status: ProvenanceStatus
    range_semantics: str = "unbounded unless constrained by the implementation"
    uncertainty_semantics: str | None = None
    missing_value_semantics: str = "not nullable"

    def __post_init__(self) -> None:
        for name in (
            "artifact",
            "column",
            "storage_type",
            "description",
            "period",
            "population_base",
            "condition",
            "recipe_id",
            "recipe_version",
            "source_version",
            "implementation",
            "formula",
            "range_semantics",
            "missing_value_semantics",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MetricContractValidationError(
                    f"metric contract {self.column!r} has empty {name}"
                )
        if not self.artifact.endswith(".csv"):
            raise MetricContractValidationError("metric artifact must be a CSV file")
        if not isinstance(self.role, MetricRole):
            raise MetricContractValidationError("metric role is invalid")
        if not isinstance(self.nullable, bool):
            raise MetricContractValidationError("nullable must be boolean")
        if not isinstance(self.unit, OutputUnit):
            raise MetricContractValidationError("metric unit is invalid")
        if not isinstance(self.status, ProvenanceStatus):
            raise MetricContractValidationError("metric status is invalid")
        if self.source_retrieved_on is not None and type(
            self.source_retrieved_on
        ) is not date:
            raise MetricContractValidationError(
                "source_retrieved_on must be an ISO calendar date"
            )
        for name in ("inputs", "lineage_ids"):
            values = getattr(self, name)
            if len(set(values)) != len(values) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise MetricContractValidationError(
                    f"metric contract {self.column!r} has invalid {name}"
                )
        if self.role is MetricRole.DERIVED and (not self.inputs or not self.lineage_ids):
            raise MetricContractValidationError(
                f"derived metric {self.column!r} needs inputs and lineage"
            )
        if (
            self.status is ProvenanceStatus.CALIBRATED
            and self.source_retrieved_on is None
        ):
            raise MetricContractValidationError(
                f"CALIBRATED metric {self.column!r} needs a retrieval date"
            )

    @property
    def contract_id(self) -> str:
        return f"{self.artifact}:{self.column}"

    def snapshot(self) -> dict[str, object]:
        """Return the canonical JSON-compatible contract payload."""

        return {
            "contract_id": self.contract_id,
            "artifact": self.artifact,
            "column": self.column,
            "role": self.role.value,
            "storage_type": self.storage_type,
            "nullable": self.nullable,
            "description": self.description,
            "unit": self.unit.snapshot(),
            "period": self.period,
            "population_base": self.population_base,
            "condition": self.condition,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "source_version": self.source_version,
            "source_retrieved_on": (
                self.source_retrieved_on.isoformat()
                if self.source_retrieved_on is not None
                else None
            ),
            "implementation": self.implementation,
            "formula": self.formula,
            "inputs": list(self.inputs),
            "lineage_ids": list(self.lineage_ids),
            "status": self.status.value,
            "range_semantics": self.range_semantics,
            "uncertainty_semantics": self.uncertainty_semantics,
            "missing_value_semantics": self.missing_value_semantics,
        }


_SIMULATION_MONEY_BASIS = (
    "internal illustrative simulation cents; not nominal FX/PPP-comparable "
    "currency across jurisdictions"
)
UNIT_IDENTIFIER = OutputUnit("identifier", "identifier")
UNIT_TEXT = OutputUnit("text", "text")
UNIT_SEED = OutputUnit("random_seed", "seed")
UNIT_SHA256 = OutputUnit("digest", "sha256_hex")
UNIT_COUNT = OutputUnit("count", "count")
UNIT_DAY = OutputUnit("time", "day")
UNIT_YEAR = OutputUnit("time", "year")
UNIT_MINUTE = OutputUnit("time", "minute")
UNIT_BOOLEAN = OutputUnit("boolean", "boolean")
UNIT_RATIO = OutputUnit("ratio", "ratio")
UNIT_SCORE = OutputUnit("model_score", "score")
UNIT_PARAMETER = OutputUnit("parameter_specific", "parameter_value")
UNIT_MONEY = OutputUnit(
    "money", "simulation_cent", money_basis=_SIMULATION_MONEY_BASIS
)

_RECIPE_VERSION = "1.0"
_BASE_LINEAGE = (
    "config_sha256",
    "profile_inputs.fingerprint_sha256",
    "repository.git_commit",
    "random_stream_contract",
)


def _contract(
    artifact: str,
    column: str,
    role: MetricRole,
    unit: OutputUnit,
    *,
    storage_type: str,
    description: str,
    period: str,
    population_base: str,
    condition: str,
    formula: str,
    inputs: tuple[str, ...],
    implementation: str,
    lineage_ids: tuple[str, ...] = _BASE_LINEAGE,
    nullable: bool = False,
    range_semantics: str = "unbounded unless constrained by the implementation",
    uncertainty_semantics: str | None = None,
    missing_value_semantics: str = "not nullable",
) -> OutputMetricContract:
    return OutputMetricContract(
        artifact=artifact,
        column=column,
        role=role,
        storage_type=storage_type,
        nullable=nullable,
        description=description,
        unit=unit,
        period=period,
        population_base=population_base,
        condition=condition,
        recipe_id=f"{artifact.removesuffix('.csv')}.{column}",
        recipe_version=_RECIPE_VERSION,
        source_version=METRIC_RECIPE_SOURCE_VERSION,
        source_retrieved_on=None,
        implementation=implementation,
        formula=formula,
        inputs=inputs,
        lineage_ids=lineage_ids,
        status=ProvenanceStatus.SYNTHETIC,
        range_semantics=range_semantics,
        uncertainty_semantics=uncertainty_semantics,
        missing_value_semantics=missing_value_semantics,
    )


def _identifier_or_design_contracts(
    artifact: str,
    specs: Mapping[
        str,
        tuple[MetricRole, OutputUnit, str, str, str, str, str],
    ],
    *,
    implementation: str,
) -> dict[str, OutputMetricContract]:
    contracts: dict[str, OutputMetricContract] = {}
    for column, (
        role,
        unit,
        storage_type,
        description,
        period,
        population_base,
        formula,
    ) in specs.items():
        contracts[column] = _contract(
            artifact,
            column,
            role,
            unit,
            storage_type=storage_type,
            description=description,
            period=period,
            population_base=population_base,
            condition="all emitted rows",
            formula=formula,
            inputs=(formula,),
            implementation=implementation,
            lineage_ids=("output_schema_version", "repository.git_commit"),
            range_semantics=(
                "strict Python integer in the inclusive range [0, 2**64 - 1]"
                if unit is UNIT_SEED
                else "value belongs to the declared run design domain"
            ),
            missing_value_semantics=(
                "Not nullable; booleans, foreign integer scalars, floating-point "
                "values, and modulo-wrapped aliases are rejected."
                if unit is UNIT_SEED
                else "not nullable"
            ),
        )
    return contracts


_SEED_MONEY_COLUMNS = frozenset(
    {
        "total_revenue_cents",
        "producer_cost_cents",
        "producer_profit_cents",
        "total_spending_cents",
        "harmful_spending_cents",
        "unplanned_spending_cents",
        "total_opportunity_cost_proxy_cents",
        "adult_opportunity_cost_proxy_cents",
        "youth_opportunity_cost_proxy_cents",
        "high_risk_mean_budget_cents",
        "spend_p10_cents",
        "spend_p50_cents",
        "spend_p90_cents",
        "harmful_spending_effect_vs_safe_cents",
        "total_revenue_effect_vs_safe_cents",
        "total_spending_effect_vs_safe_cents",
        "epgc_minimum_public_contribution_cents",
        "epgc_profit_safe_cents",
    }
)
_SEED_MONEY_COLUMNS = _SEED_MONEY_COLUMNS.union(
    column
    for column in TABLE_COLUMNS["seed_results.csv"]
    if column.startswith("revenue_") and column.endswith("_cents")
)
_SEED_FLOAT_MONEY_COLUMNS = frozenset(
    {
        "high_risk_mean_budget_cents",
        "spend_p10_cents",
        "spend_p50_cents",
        "spend_p90_cents",
    }
)


def _seed_unit(column: str) -> OutputUnit:
    if column == "harm_variance_players":
        return UNIT_SCORE.squared()
    if column in _SEED_MONEY_COLUMNS:
        return UNIT_MONEY
    if column == "high_risk_count":
        return UNIT_COUNT
    if column == "high_risk_mean_age":
        return UNIT_YEAR
    if column in {"high_risk_share", "high_risk_minor_share"}:
        return UNIT_RATIO
    return UNIT_SCORE


def _seed_recipe(column: str) -> tuple[str, tuple[str, ...]]:
    if column == "total_revenue_cents":
        return "sum(revenue_composition_cents[source])", (
            "PolicyScenarioResult.revenue_composition_cents",
        )
    if column == "producer_cost_cents":
        return "PolicyScenarioResult.producer_cost_cents", (
            "PolicyScenarioResult.producer_cost_cents",
        )
    if column == "producer_profit_cents":
        return "total_revenue_cents - producer_cost_cents", (
            "seed_results.csv:total_revenue_cents",
            "seed_results.csv:producer_cost_cents",
        )
    if column.startswith("revenue_") and column.endswith("_cents"):
        source = column.removeprefix("revenue_").removesuffix("_cents")
        return f"revenue_composition_cents[{source!r}]", (
            f"PolicyScenarioResult.revenue_composition_cents[{source!r}]",
        )
    if column == "total_spending_cents":
        return "sum(player spending_cents)", ("PolicyScenarioResult.spending_cents",)
    if column == "harmful_spending_cents":
        return "sum(player harmful_spending_cents)", (
            "WelfareHarmResult.harmful_spending_cents",
        )
    if column == "unplanned_spending_cents":
        return "sum(player unplanned_spending_cents)", (
            "WelfareHarmResult.unplanned_spending_cents",
        )
    if column == "harm_variance_players":
        return "population_variance(composite_harm, ddof=0)", (
            "PolicyScenarioResult.composite_harm",
        )
    if column.startswith("harm_p"):
        probability = int(column.removeprefix("harm_p")) / 100
        return f"quantile(composite_harm, {probability:.2f})", (
            "PolicyScenarioResult.composite_harm",
        )
    if column.startswith("spend_p"):
        probability = int(
            column.removeprefix("spend_p").removesuffix("_cents")
        ) / 100
        return f"quantile(spending_cents, {probability:.2f})", (
            "PolicyScenarioResult.spending_cents",
        )
    if column == "mean_harm":
        return "mean(composite_harm)", ("PolicyScenarioResult.composite_harm",)
    component_columns = {
        "mean_monetary_harm": "M",
        "mean_opportunity_cost_score": "OC",
        "mean_sleep_burden": "S",
        "mean_education_work_burden": "E",
        "mean_social_burden": "F",
        "mean_wellbeing_burden": "W",
    }
    if column in component_columns:
        component = component_columns[column]
        return f"mean(harm_component[{component}])", (
            f"WelfareHarmResult.component_scores[:, {component}]",
        )
    opportunity_proxy_fields = {
        "total_opportunity_cost_proxy_cents": "opportunity_cost_proxy_cents",
        "adult_opportunity_cost_proxy_cents": (
            "adult_opportunity_cost_proxy_cents"
        ),
        "youth_opportunity_cost_proxy_cents": (
            "youth_opportunity_cost_proxy_cents"
        ),
    }
    if column in opportunity_proxy_fields:
        field = opportunity_proxy_fields[column]
        return f"sum(WelfareHarmResult.{field})", (
            f"WelfareHarmResult.{field}",
        )
    if column == "mean_enjoyment":
        return "mean(player enjoyment)", ("PolicyScenarioResult.enjoyment",)
    if column == "high_risk_count":
        return "count_nonzero(high_risk)", ("PolicyScenarioResult.high_risk",)
    if column == "high_risk_share":
        return "mean(high_risk)", ("PolicyScenarioResult.high_risk",)
    high_risk_means = {
        "high_risk_mean_age": "age_years",
        "high_risk_minor_share": "is_minor",
        "high_risk_mean_budget_cents": "disposable_budget_cents",
        "high_risk_mean_baseline_vulnerability": "baseline_vulnerability",
    }
    if column in high_risk_means:
        field = high_risk_means[column]
        return f"mean({field}[high_risk])", (
            f"PolicyScenarioResult.{field}",
            "PolicyScenarioResult.high_risk",
        )
    effect_inputs = {
        "mean_harm_effect_vs_safe": (
            "mean(scenario composite_harm - safe-reference composite_harm)",
            (
                "PolicyScenarioResult.composite_harm[scenario]",
                "PolicyScenarioResult.composite_harm[safe-reference]",
            ),
        ),
        "total_spending_effect_vs_safe_cents": (
            "scenario total_spending_cents - safe-reference total_spending_cents",
            (
                "PolicyScenarioResult.spending_cents[scenario]",
                "PolicyScenarioResult.spending_cents[safe-reference]",
            ),
        ),
        "harmful_spending_effect_vs_safe_cents": (
            "scenario harmful_spending_cents - safe-reference harmful_spending_cents",
            (
                "WelfareHarmResult.harmful_spending_cents[scenario]",
                "WelfareHarmResult.harmful_spending_cents[safe-reference]",
            ),
        ),
        "total_revenue_effect_vs_safe_cents": (
            "scenario total_revenue_cents - safe-reference total_revenue_cents",
            (
                "PolicyScenarioResult.total_revenue_cents[scenario]",
                "PolicyScenarioResult.total_revenue_cents[safe-reference]",
            ),
        ),
    }
    if column in effect_inputs:
        formula, sources = effect_inputs[column]
        return formula, (*sources, "PolicyBatchSpec.reference_scenario")
    if column == "epgc_minimum_public_contribution_cents":
        return "EPGC minimum contribution, else 0", (
            "PolicyScenarioResult.epgc.minimum_public_contribution_cents",
        )
    if column == "epgc_profit_safe_cents":
        return "EPGC policy-adjusted safe profit, else 0", (
            "PolicyScenarioResult.epgc.profit_safe_cents",
        )
    raise MetricContractValidationError(f"missing seed recipe for {column}")


def _build_seed_contracts() -> dict[str, OutputMetricContract]:
    artifact = "seed_results.csv"
    contracts = _identifier_or_design_contracts(
        artifact,
        {
            "scenario_id": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "string",
                "Stable scenario identifier.",
                "one configured scenario",
                "scenario-seed result",
                "ScenarioSpec.scenario_id.value",
            ),
            "scenario_label": (
                MetricRole.IDENTIFIER,
                UNIT_TEXT,
                "string",
                "Human-readable scenario label.",
                "one configured scenario",
                "scenario-seed result",
                "ScenarioSpec.label",
            ),
            "seed": (
                MetricRole.IDENTIFIER,
                UNIT_SEED,
                "integer",
                "Configured counter-based random seed.",
                "one replication",
                "scenario-seed result",
                "PolicyScenarioResult.seed",
            ),
            "cohort_digest": (
                MetricRole.IDENTIFIER,
                UNIT_SHA256,
                "string",
                "Content digest of the shared initial cohort and life state.",
                "one replication",
                "scenario-seed result",
                "sha256(initial PlayerTable and PlayerLife arrays)",
            ),
            "days": (
                MetricRole.DESIGN,
                UNIT_DAY,
                "integer",
                "Configured simulation horizon.",
                "one replication",
                "scenario-seed result",
                "PolicyScenarioResult.days",
            ),
            "player_count": (
                MetricRole.DESIGN,
                UNIT_COUNT,
                "integer",
                "Number of synthetic players in the cohort.",
                "one replication",
                "scenario-seed result",
                "len(PolicyScenarioResult.player_ids)",
            ),
        },
        implementation="microtx_sim.causal.batch._seed_row",
    )
    contracts["cohort_digest"] = replace(
        contracts["cohort_digest"],
        implementation="microtx_sim.causal.batch._cohort_digest",
    )
    high_risk_subset_columns = {
        "high_risk_mean_age",
        "high_risk_minor_share",
        "high_risk_mean_budget_cents",
        "high_risk_mean_baseline_vulnerability",
    }
    player_reduction_columns = {
        "total_spending_cents",
        "harmful_spending_cents",
        "unplanned_spending_cents",
        "mean_harm",
        "harm_variance_players",
        "harm_p10",
        "harm_p50",
        "harm_p90",
        "spend_p10_cents",
        "spend_p50_cents",
        "spend_p90_cents",
        "mean_monetary_harm",
        "mean_opportunity_cost_score",
        "mean_sleep_burden",
        "mean_education_work_burden",
        "mean_social_burden",
        "mean_wellbeing_burden",
        "total_opportunity_cost_proxy_cents",
        "adult_opportunity_cost_proxy_cents",
        "youth_opportunity_cost_proxy_cents",
        "mean_enjoyment",
        "high_risk_count",
        "high_risk_share",
        "mean_harm_effect_vs_safe",
        "total_spending_effect_vs_safe_cents",
        "harmful_spending_effect_vs_safe_cents",
        "revenue_direct_purchase_cents",
        "revenue_opaque_virtual_currency_cents",
        "revenue_paid_random_rewards_cents",
        "revenue_fixed_price_cents",
        "revenue_subscription_cents",
    }
    for column in TABLE_COLUMNS[artifact]:
        if column in contracts:
            continue
        formula, inputs = _seed_recipe(column)
        high_risk_subset = column in high_risk_subset_columns
        effect = "effect_vs_safe" in column
        contracts[column] = _contract(
            artifact,
            column,
            MetricRole.DERIVED,
            _seed_unit(column),
            storage_type=(
                "float"
                if column in _SEED_FLOAT_MONEY_COLUMNS
                or _seed_unit(column).quantity
                in {"model_score", "ratio", "time"}
                else "integer"
            ),
            description=(
                column.replace("_", " ").capitalize()
                + " computed inside the synthetic structural model."
            ),
            period="configured simulation horizon",
            population_base=(
                "synthetic players classified high-risk in one scenario-seed"
                if high_risk_subset
                else "one synthetic scenario-seed cohort"
            ),
            condition=(
                "players satisfying the model high-risk threshold"
                if high_risk_subset
                else (
                    "all players classified by the model high-risk threshold"
                    if column in {"high_risk_count", "high_risk_share"}
                else (
                    "paired against the configured safe-reference scenario"
                    if effect
                    else "all synthetic players or transactions in the row"
                )
                )
            ),
            formula=formula,
            inputs=inputs,
            implementation=(
                "microtx_sim.causal.batch.run_policy_batch"
                if effect
                else (
                    "microtx_sim.simulation.policy_orchestrator.run_policy_scenario"
                    if column in {"total_revenue_cents", "producer_profit_cents"}
                    else "microtx_sim.causal.batch._seed_row"
                )
            ),
            range_semantics=(
                "[0, 1]"
                if column in {"high_risk_share", "high_risk_minor_share"}
                else "model-defined; paired effects may be negative"
            ),
            uncertainty_semantics=(
                "Population variance across players with ddof=0; distinct from "
                "between-seed sample variance."
                if column == "harm_variance_players"
                else None
            ),
            missing_value_semantics=(
                "Encoded as 0 when the high-risk subset is empty."
                if high_risk_subset
                else (
                    "Encoded as 0 outside EPGC rows."
                    if column.startswith("epgc_")
                    else (
                        "Empty player arrays reduce to 0."
                        if column in player_reduction_columns
                        else (
                            "Not nullable; producer, policy, and revenue values "
                            "remain defined for empty player cohorts."
                        )
                    )
                )
            ),
        )
    return contracts


def _build_summary_contracts(
    seed_contracts: Mapping[str, OutputMetricContract],
) -> dict[str, OutputMetricContract]:
    artifact = "scenario_summary.csv"
    contracts = _identifier_or_design_contracts(
        artifact,
        {
            "scenario_id": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "string",
                "Stable scenario identifier.",
                "one configured scenario",
                "scenario summary",
                "ScenarioSpec.scenario_id.value",
            ),
            "scenario_label": (
                MetricRole.IDENTIFIER,
                UNIT_TEXT,
                "string",
                "Human-readable scenario label.",
                "one configured scenario",
                "scenario summary",
                "ScenarioSpec.label",
            ),
            "seed_count": (
                MetricRole.DESIGN,
                UNIT_COUNT,
                "integer",
                "Number of unique configured replications.",
                "repeated-seed design",
                "scenario summary",
                "len(seed rows for scenario)",
            ),
            "player_count": (
                MetricRole.DESIGN,
                UNIT_COUNT,
                "integer",
                "Synthetic cohort size per replication.",
                "one replication",
                "scenario summary",
                "PolicyBatchSpec.player_count",
            ),
            "days": (
                MetricRole.DESIGN,
                UNIT_DAY,
                "integer",
                "Configured simulation horizon per replication.",
                "one replication",
                "scenario summary",
                "PolicyBatchSpec.days",
            ),
        },
        implementation="microtx_sim.causal.batch.PolicyBatchResult.scenario_rows",
    )
    suffixes = ("ci95_low", "ci95_high", "variance", "mean", "sd")
    uncertainty = (
        "Monte Carlo dispersion across configured seeds. Variance is the sample "
        "variance with ddof=1; CI bounds are mean ± 1.96*sd/sqrt(seed_count). "
        "A one-seed design has zero variance and a zero-width interval. These are "
        "simulation-mean intervals, not empirical outcome uncertainty."
    )
    for column in TABLE_COLUMNS[artifact]:
        if column in contracts:
            continue
        suffix = next(
            (candidate for candidate in suffixes if column.endswith(f"_{candidate}")),
            None,
        )
        if suffix is None:
            raise MetricContractValidationError(
                f"summary column {column!r} lacks an uncertainty suffix"
            )
        stem = column.removesuffix(f"_{suffix}")
        if stem not in REPEATED_SEED_METRIC_STEMS:
            raise MetricContractValidationError(
                f"summary metric {stem!r} is not a declared repeated-seed stem"
            )
        base = seed_contracts[stem]
        if suffix == "mean":
            formula = "mean(x_s)"
            unit = base.unit
        elif suffix == "variance":
            formula = "sum((x_s - mean(x_s))^2) / (seed_count - 1); 0 if n=1"
            unit = base.unit.squared()
        elif suffix == "sd":
            formula = "sqrt(sample_variance(x_s, ddof=1)); 0 if n=1"
            unit = base.unit
        elif suffix == "ci95_low":
            formula = "mean(x_s) - 1.96*sd(x_s)/sqrt(seed_count)"
            unit = base.unit
        else:
            formula = "mean(x_s) + 1.96*sd(x_s)/sqrt(seed_count)"
            unit = base.unit
        contracts[column] = _contract(
            artifact,
            column,
            MetricRole.DERIVED,
            unit,
            storage_type="float",
            description=(
                f"Repeated-seed {suffix.replace('_', ' ')} for {stem.replace('_', ' ')}."
            ),
            period="across replications of the configured simulation horizon",
            population_base="configured unique seeds within one scenario",
            condition="all seed rows for the scenario",
            formula=formula,
            inputs=(f"seed_results.csv:{stem}", "seed_results.csv:scenario_id"),
            implementation="microtx_sim.causal.batch._uncertainty",
            lineage_ids=_BASE_LINEAGE + (f"seed_results.csv:{stem}",),
            range_semantics=(
                "non-negative"
                if suffix in {"variance", "sd"}
                else "inherits the source metric; normal CI bounds may cross limits"
            ),
            uncertainty_semantics=uncertainty,
            missing_value_semantics="A one-seed design is encoded with zero dispersion.",
        )
    return contracts


def _build_epgc_contracts() -> dict[str, OutputMetricContract]:
    artifact = "epgc_financing.csv"
    contracts = _identifier_or_design_contracts(
        artifact,
        {
            "scenario_id": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "string",
                "EPGC scenario identifier.",
                "one configured scenario",
                "scenario-seed financing row",
                "PolicyScenarioResult.scenario.scenario_id.value",
            ),
            "seed": (
                MetricRole.IDENTIFIER,
                UNIT_SEED,
                "integer",
                "Configured counter-based random seed.",
                "one replication",
                "scenario-seed financing row",
                "PolicyScenarioResult.seed",
            ),
        },
        implementation="microtx_sim.causal.batch.PolicyBatchResult.epgc_rows",
    )
    formulas = {
        "public_contract_revenue_cents": (
            UNIT_MONEY,
            "integer",
            "Public-contract revenue paid in the simulated financing calculation.",
            "budget_limited_public_contract_cents - clawback_cents - penalty_cents",
            (
                "EPGCResult.budget_limited_public_contract_cents",
                "EPGCResult.clawback_cents",
                "EPGCResult.penalty_cents",
            ),
        ),
        "minimum_public_contribution_cents": (
            UNIT_MONEY,
            "integer",
            "Minimum contribution needed to reach the safe-profit target.",
            "max(0, development_cost_cents + maintenance_cost_cents - fixed_price_revenue_cents - institutional_licensing_revenue_cents - non_targeted_sponsorship_revenue_cents)",
            (
                "EPGCResult.development_cost_cents",
                "EPGCResult.maintenance_cost_cents",
                "EPGCResult.fixed_price_revenue_cents",
                "EPGCResult.institutional_licensing_revenue_cents",
                "EPGCResult.non_targeted_sponsorship_revenue_cents",
            ),
        ),
        "maximum_budget_cents": (
            UNIT_MONEY,
            "integer",
            "Configured policy budget cap available to the EPGC calculation.",
            "EPGCPolicy.maximum_budget_cents",
            ("EPGCPolicy.maximum_budget_cents",),
        ),
        "profit_safe_cents": (
            UNIT_MONEY,
            "integer",
            "Policy-adjusted producer profit after EPGC transfers and sanctions.",
            "public_contract_revenue_cents + fixed_price_revenue_cents + institutional_licensing_revenue_cents + non_targeted_sponsorship_revenue_cents - development_cost_cents - maintenance_cost_cents",
            (
                "EPGCResult.public_contract_revenue_cents",
                "EPGCResult.fixed_price_revenue_cents",
                "EPGCResult.institutional_licensing_revenue_cents",
                "EPGCResult.non_targeted_sponsorship_revenue_cents",
                "EPGCResult.development_cost_cents",
                "EPGCResult.maintenance_cost_cents",
            ),
        ),
        "feasible_under_budget_cap": (
            UNIT_BOOLEAN,
            "boolean",
            "Whether the minimum contribution fits within the budget cap.",
            "minimum_public_contribution_cents <= maximum_budget_cents",
            (
                "EPGCResult.minimum_public_contribution_cents",
                "EPGCResult.maximum_budget_cents",
            ),
        ),
        "sustainable_under_policy": (
            UNIT_BOOLEAN,
            "boolean",
            "Whether policy-adjusted profit reaches the safe-profit target.",
            "profit_safe_cents >= 0",
            ("EPGCResult.profit_safe_cents",),
        ),
        "clawback_cents": (
            UNIT_MONEY,
            "integer",
            "Configured EPGC clawback applied to the simulated producer.",
            "budget_limited_public_contract_cents * clawback_basis_points // 10000 when prohibited mechanics are enabled; otherwise 0",
            (
                "EPGCFirmInputs.prohibited_mechanics_enabled",
                "EPGCResult.budget_limited_public_contract_cents",
                "EPGCPolicy.prohibited_mechanics_clawback_basis_points",
            ),
        ),
        "penalty_cents": (
            UNIT_MONEY,
            "integer",
            "Configured EPGC penalty applied to the simulated producer.",
            "min(prohibited_mechanics_penalty_cents, budget_limited_public_contract_cents - clawback_cents) when prohibited mechanics are enabled; otherwise 0",
            (
                "EPGCFirmInputs.prohibited_mechanics_enabled",
                "EPGCPolicy.prohibited_mechanics_penalty_cents",
                "EPGCResult.budget_limited_public_contract_cents",
                "EPGCResult.clawback_cents",
            ),
        ),
    }
    for column in TABLE_COLUMNS[artifact]:
        if column in contracts:
            continue
        unit, storage_type, description, formula, inputs = formulas[column]
        contracts[column] = _contract(
            artifact,
            column,
            MetricRole.DERIVED,
            unit,
            storage_type=storage_type,
            description=description,
            period="configured financing round for one simulated horizon",
            population_base="one EPGC scenario-seed producer calculation",
            condition="rows with an EPGCResult; non-EPGC scenarios are omitted",
            formula=formula,
            inputs=inputs,
            implementation="microtx_sim.funding.epgc.evaluate_epgc",
            range_semantics=(
                "boolean"
                if unit is UNIT_BOOLEAN
                else (
                    "signed internal simulation cents"
                    if column == "profit_safe_cents"
                    else "non-negative internal simulation cents"
                )
            ),
        )
    return contracts


def _build_sensitivity_contracts() -> dict[str, OutputMetricContract]:
    artifact = "sensitivity.csv"
    contracts = _identifier_or_design_contracts(
        artifact,
        {
            "parameter": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "string",
                "Sensitivity parameter name.",
                "one OAT grid",
                "parameter-level scenario row",
                "SensitivityCase.parameter",
            ),
            "scenario_id": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "string",
                "Scenario evaluated by the OAT case.",
                "one configured scenario",
                "parameter-level scenario row",
                "SensitivityCase.scenario_id.value",
            ),
            "parameter_value": (
                MetricRole.DESIGN,
                UNIT_PARAMETER,
                "float",
                "Configured OAT level; unit depends on the named parameter.",
                "one OAT level",
                "parameter-level scenario row",
                "SensitivityCase.values[level]",
            ),
            "seed_count": (
                MetricRole.DESIGN,
                UNIT_COUNT,
                "integer",
                "Number of common-cohort seed replications at this level.",
                "one OAT level",
                "parameter-level scenario row",
                "len(PolicyBatchSpec.seeds)",
            ),
            "expected_direction": (
                MetricRole.DESIGN,
                UNIT_TEXT,
                "string",
                "Prospectively configured monotonic direction.",
                "one OAT grid",
                "parameter-level scenario row",
                "SensitivityCase.expected_direction",
            ),
            "monotonic_expected": (
                MetricRole.DESIGN,
                UNIT_BOOLEAN,
                "boolean",
                "Whether the case declares a non-neutral expected direction.",
                "one OAT grid",
                "parameter-level scenario row",
                "SensitivityCase.expected_direction != 'none'",
            ),
        },
        implementation="microtx_sim.analysis.sensitivity.run_sensitivity_analysis",
    )
    uncertainty = (
        "Between-seed Monte Carlo uncertainty at one OAT level: sample variance "
        "uses ddof=1 and CI bounds use mean ± 1.96*sd/sqrt(seed_count). A one-seed "
        "level has zero dispersion. This is not empirical parameter uncertainty."
    )
    specs = {
        "mean_harm": (UNIT_SCORE, "mean(seed mean composite_harm)"),
        "harm_variance": (
            UNIT_SCORE.squared(),
            "sample_variance(seed mean composite_harm, ddof=1)",
        ),
        "harm_sd": (UNIT_SCORE, "sqrt(harm_variance)"),
        "harm_ci95_low": (
            UNIT_SCORE,
            "mean_harm - 1.96*harm_sd/sqrt(seed_count)",
        ),
        "harm_ci95_high": (
            UNIT_SCORE,
            "mean_harm + 1.96*harm_sd/sqrt(seed_count)",
        ),
        "harm_coefficient_of_variation": (
            UNIT_RATIO,
            "harm_sd / abs(mean_harm) when abs(mean_harm) > 1e-12; otherwise 0 when harm_sd == 0, else infinity",
        ),
        "total_revenue_cents": (
            UNIT_MONEY,
            "mean(seed total_revenue_cents)",
        ),
        "opportunity_cost_burden": (
            UNIT_SCORE,
            "mean(seed mean opportunity-cost component score)",
        ),
        "minimum_public_contribution_cents": (
            UNIT_MONEY,
            "mean(seed EPGC minimum contribution, using 0 without EPGC)",
        ),
        "monotonic_observed": (
            UNIT_BOOLEAN,
            "none: true; increasing: every adjacent right + 1e-12 >= left; decreasing: every adjacent right <= left + 1e-12",
        ),
        "unstable": (
            UNIT_BOOLEAN,
            "(non-neutral expected direction is violated across the full grid) OR any level harm_coefficient_of_variation exceeds instability_cv_threshold",
        ),
    }
    for column in TABLE_COLUMNS[artifact]:
        if column in contracts:
            continue
        unit, formula = specs[column]
        case_wide = column in {"monotonic_observed", "unstable"}
        inputs = (
            (
                "all level mean_harm values sorted by parameter_value",
                "SensitivityCase.expected_direction",
                "instability_cv_threshold",
                "all level harm_coefficient_of_variation values",
                "microtx_sim.analysis.sensitivity._MONOTONIC_TOLERANCE=1e-12",
            )
            if column == "unstable"
            else (
                "all level mean_harm values sorted by parameter_value",
                "SensitivityCase.expected_direction",
                "microtx_sim.analysis.sensitivity._MONOTONIC_TOLERANCE=1e-12",
            )
            if column == "monotonic_observed"
            else (
                "harm_stats.mean",
                "harm_stats.sd",
                "microtx_sim.analysis.sensitivity._CV_ZERO_MEAN_TOLERANCE=1e-12",
            )
            if column == "harm_coefficient_of_variation"
            else (
                "PolicyScenarioResult outputs by seed",
                "SensitivityCase.values",
                "SensitivityCase.expected_direction",
            )
        )
        contracts[column] = _contract(
            artifact,
            column,
            MetricRole.DERIVED,
            unit,
            storage_type="boolean" if unit is UNIT_BOOLEAN else "float",
            description=(
                column.replace("_", " ").capitalize()
                + " from the synthetic OAT diagnostic."
            ),
            period=(
                "one complete OAT case across all configured levels and seeds"
                if case_wide
                else "across common-cohort replications at one OAT level"
            ),
            population_base=(
                "all parameter levels and configured seeds for one scenario"
                if case_wide
                else "configured seeds for one parameter-level scenario"
            ),
            condition=(
                "all configured levels and seeds in the OAT case"
                if case_wide
                else "all configured seeds at the OAT level"
            ),
            formula=formula,
            inputs=inputs,
            implementation="microtx_sim.analysis.sensitivity.run_sensitivity_analysis",
            range_semantics=(
                "boolean"
                if unit is UNIT_BOOLEAN
                else (
                    "non-negative; may be infinite before CSV validation"
                    if column == "harm_coefficient_of_variation"
                    else "model-defined"
                )
            ),
            uncertainty_semantics=(
                uncertainty
                if column.startswith("harm_") or column == "mean_harm"
                else None
            ),
            missing_value_semantics=(
                "EPGC absence contributes 0."
                if column == "minimum_public_contribution_cents"
                else (
                    "Not nullable; the case-wide value is repeated on every level row."
                    if case_wide
                    else "A one-seed level has zero dispersion."
                )
            ),
        )
    return contracts


def _build_player_contracts() -> dict[str, OutputMetricContract]:
    artifact = "player_outcomes.csv"
    contracts = _identifier_or_design_contracts(
        artifact,
        {
            "scenario_id": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "string",
                "Stable scenario identifier.",
                "one configured scenario",
                "synthetic player-scenario-seed row",
                "PolicyScenarioResult.scenario.scenario_id.value",
            ),
            "seed": (
                MetricRole.IDENTIFIER,
                UNIT_SEED,
                "integer",
                "Configured counter-based random seed.",
                "one replication",
                "synthetic player-scenario-seed row",
                "PolicyScenarioResult.seed",
            ),
            "player_id": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "integer",
                "Synthetic player identifier within the cohort.",
                "one replication",
                "synthetic player-scenario-seed row",
                "PolicyScenarioResult.player_ids[index]",
            ),
        },
        implementation="microtx_sim.causal.batch.PolicyBatchResult.player_rows",
    )
    money = {
        "spending_cents",
        "harmful_spending_cents",
        "opportunity_cost_proxy_cents",
    }
    booleans = {"is_minor", "high_risk"}
    scores = {
        "baseline_vulnerability",
        "composite_harm",
        "monetary_harm",
        "opportunity_cost",
        "sleep_burden",
        "education_work_burden",
        "social_burden",
        "wellbeing_burden",
        "enjoyment",
    }
    player_inputs = {
        "age_years": ("PolicyScenarioResult.age_years",),
        "is_minor": ("PolicyScenarioResult.is_minor",),
        "baseline_vulnerability": (
            "PolicyScenarioResult.baseline_vulnerability",
        ),
        "spending_cents": ("PolicyScenarioResult.spending_cents",),
        "harmful_spending_cents": (
            "WelfareHarmResult.harmful_spending_cents",
        ),
        "composite_harm": ("PolicyScenarioResult.composite_harm",),
        "monetary_harm": ("WelfareHarmResult.component_scores[:, M]",),
        "opportunity_cost": ("WelfareHarmResult.component_scores[:, OC]",),
        "sleep_burden": ("WelfareHarmResult.component_scores[:, S]",),
        "education_work_burden": (
            "WelfareHarmResult.component_scores[:, E]",
        ),
        "social_burden": ("WelfareHarmResult.component_scores[:, F]",),
        "wellbeing_burden": ("WelfareHarmResult.component_scores[:, W]",),
        "opportunity_cost_proxy_cents": (
            "WelfareHarmResult.opportunity_cost_proxy_cents",
        ),
        "enjoyment": ("PolicyScenarioResult.enjoyment",),
        "high_risk": ("PolicyScenarioResult.high_risk",),
    }
    for column in TABLE_COLUMNS[artifact]:
        if column in contracts:
            continue
        if column in money:
            unit, storage = UNIT_MONEY, "integer"
        elif column in booleans:
            unit, storage = UNIT_BOOLEAN, "boolean"
        elif column == "age_years":
            unit, storage = UNIT_YEAR, "integer"
        elif column in scores:
            unit, storage = UNIT_SCORE, "float"
        else:
            raise MetricContractValidationError(f"missing player unit for {column}")
        contracts[column] = _contract(
            artifact,
            column,
            MetricRole.DERIVED,
            unit,
            storage_type=storage,
            description=(
                column.replace("_", " ").capitalize()
                + " for one synthetic player outcome."
            ),
            period=(
                "simulation initialisation"
                if column in {"age_years", "is_minor", "baseline_vulnerability"}
                else "configured simulation horizon"
            ),
            population_base="one synthetic player in one scenario-seed cohort",
            condition="player row retained when player-level export is enabled",
            formula=f"PolicyScenarioResult player-index projection for {column}",
            inputs=player_inputs[column] + ("PolicyScenarioResult.player_ids",),
            implementation="microtx_sim.causal.batch.PolicyBatchResult.player_rows",
            range_semantics=(
                "boolean"
                if unit is UNIT_BOOLEAN
                else (
                    "non-negative internal simulation cents"
                    if unit is UNIT_MONEY
                    else "model-defined score or age domain"
                )
            ),
        )
    return contracts


def _build_opportunity_contracts() -> dict[str, OutputMetricContract]:
    artifact = "opportunity_cost_decomposition.csv"
    contracts = _identifier_or_design_contracts(
        artifact,
        {
            "scenario_id": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "string",
                "Stable scenario identifier.",
                "one configured scenario",
                "scenario-component row",
                "ScenarioSpec.scenario_id.value",
            ),
            "component": (
                MetricRole.IDENTIFIER,
                UNIT_IDENTIFIER,
                "string",
                "Displaced-activity or aggregate component identifier.",
                "configured simulation horizon",
                "scenario-component row",
                "declared opportunity decomposition component",
            ),
        },
        implementation="microtx_sim.causal.batch.PolicyBatchResult.opportunity_rows",
    )
    specs = {
        "mean_minutes": (
            UNIT_MINUTE,
            "float",
            False,
            (
                "component row: mean(concatenate(displaced-minute arrays over "
                "scenario seeds)); aggregate row: sum(the four emitted component "
                "mean_minutes values)"
            ),
            (
                "WelfareHarmResult.displaced_sleep_minutes",
                "WelfareHarmResult.displaced_work_study_minutes",
                "WelfareHarmResult.displaced_social_minutes",
                "WelfareHarmResult.displaced_physical_activity_minutes",
                "opportunity_cost_decomposition.csv:component mean_minutes rows",
            ),
            "synthetic player-seed outcomes within one scenario",
            "declared activity component or sum of the four component means",
        ),
        "mean_burden": (
            UNIT_SCORE,
            "float",
            False,
            (
                "component row: mean(concatenate(component_scores[:, S/E/F] over "
                "scenario seeds)), with physical_activity fixed to 0; aggregate "
                "row: mean(per-seed mean(component_scores[:, OC]))"
            ),
            ("WelfareHarmResult.component_scores[:, S/E/F/OC]",),
            "synthetic player-seed outcomes within one scenario",
            "declared burden component or all_displaced_activities aggregate",
        ),
        "monetary_proxy_cents": (
            UNIT_MONEY,
            "float_or_empty",
            True,
            (
                "component row: blank; aggregate row: mean over scenario seeds of "
                "sum(WelfareHarmResult.opportunity_cost_proxy_cents within seed)"
            ),
            ("WelfareHarmResult.opportunity_cost_proxy_cents",),
            "configured scenario-seed cohort totals",
            "numeric only for all_displaced_activities; activity components are blank",
        ),
    }
    for column in TABLE_COLUMNS[artifact]:
        if column in contracts:
            continue
        (
            unit,
            storage,
            nullable,
            formula,
            inputs,
            population_base,
            condition,
        ) = specs[column]
        contracts[column] = _contract(
            artifact,
            column,
            MetricRole.DERIVED,
            unit,
            storage_type=storage,
            nullable=nullable,
            description=(
                column.replace("_", " ").capitalize()
                + " in the synthetic displaced-activity decomposition."
            ),
            period="across configured seeds for the simulation horizon",
            population_base=population_base,
            condition=condition,
            formula=formula,
            inputs=inputs,
            implementation="microtx_sim.causal.batch.PolicyBatchResult.opportunity_rows",
            range_semantics="non-negative in the current structural model",
            missing_value_semantics=(
                "Blank for activity-component monetary proxies because no "
                "component allocation is implemented; aggregate row is numeric."
                if nullable
                else "Empty arrays and undefined physical-activity burden are encoded as 0."
            ),
        )
    return contracts


def _build_registry() -> Mapping[
    tuple[str, str], OutputMetricContract
]:
    seed = _build_seed_contracts()
    by_artifact = {
        "seed_results.csv": seed,
        "scenario_summary.csv": _build_summary_contracts(seed),
        "epgc_financing.csv": _build_epgc_contracts(),
        "sensitivity.csv": _build_sensitivity_contracts(),
        "player_outcomes.csv": _build_player_contracts(),
        "opportunity_cost_decomposition.csv": _build_opportunity_contracts(),
    }
    ordered: dict[tuple[str, str], OutputMetricContract] = {}
    for artifact, columns in TABLE_COLUMNS.items():
        contracts = by_artifact[artifact]
        for column in columns:
            ordered[(artifact, column)] = contracts[column]
    return MappingProxyType(ordered)


def validate_metric_contract_registry(
    registry: Mapping[tuple[str, str], OutputMetricContract],
) -> None:
    """Require exact table coverage, stable order, and valid unit algebra."""

    expected = tuple(
        (artifact, column)
        for artifact, columns in TABLE_COLUMNS.items()
        for column in columns
    )
    if tuple(registry) != expected:
        missing = sorted(set(expected).difference(registry))
        extra = sorted(set(registry).difference(expected))
        raise MetricContractValidationError(
            f"metric registry differs from output schema: missing={missing}, extra={extra}"
        )
    for key, contract in registry.items():
        if not isinstance(contract, OutputMetricContract):
            raise MetricContractValidationError("registry values must be contracts")
        if key != (contract.artifact, contract.column):
            raise MetricContractValidationError(
                f"registry key does not match contract {contract.contract_id}"
            )
    summary = "scenario_summary.csv"
    seed = "seed_results.csv"
    suffixes = ("ci95_low", "ci95_high", "variance", "mean", "sd")
    summary_stems: set[str] = set()
    for column in TABLE_COLUMNS[summary][5:]:
        suffix = next(item for item in suffixes if column.endswith(f"_{item}"))
        stem = column.removesuffix(f"_{suffix}")
        summary_stems.add(stem)
        base_unit = registry[(seed, stem)].unit
        expected_unit = base_unit.squared() if suffix == "variance" else base_unit
        if registry[(summary, column)].unit != expected_unit:
            raise MetricContractValidationError(
                f"summary unit algebra is wrong for {column}"
            )
    if summary_stems != set(REPEATED_SEED_METRIC_STEMS):
        raise MetricContractValidationError(
            "summary contracts differ from repeated-seed metric stems"
        )


OUTPUT_METRIC_CONTRACTS: Final[
    Mapping[tuple[str, str], OutputMetricContract]
] = _build_registry()
validate_metric_contract_registry(OUTPUT_METRIC_CONTRACTS)


def metric_contract_registry_snapshot() -> list[dict[str, object]]:
    """Return an ordered, detached JSON-compatible registry snapshot."""

    return [contract.snapshot() for contract in OUTPUT_METRIC_CONTRACTS.values()]


def metric_contract_registry_sha256() -> str:
    """Hash the canonical registry snapshot for manifest verification."""

    encoded = json.dumps(
        {
            "schema_version": METRIC_CONTRACT_SCHEMA_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "contracts": metric_contract_registry_snapshot(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def metric_contract_campaign_blockers(
    *,
    configuration_status: str,
    profile_lineage_status: str,
    profile_dependencies_calibrated: bool,
    run_source_retrieved_on: date | None,
    monetary_outputs_cross_country_comparable: bool,
) -> tuple[str, ...]:
    """Return explicit blockers without promoting software lineage to evidence."""

    blockers: list[str] = []
    if configuration_status.upper() != ProvenanceStatus.CALIBRATED.value:
        blockers.append("policy configuration is not CALIBRATED")
    if profile_lineage_status != "registered_profile_bundle":
        blockers.append("profile inputs are not a registered source bundle")
    if not profile_dependencies_calibrated:
        blockers.append("profile input dependencies are not all CALIBRATED")
    derived = tuple(
        contract
        for contract in OUTPUT_METRIC_CONTRACTS.values()
        if contract.role is MetricRole.DERIVED
    )
    non_calibrated = sum(
        contract.status is not ProvenanceStatus.CALIBRATED for contract in derived
    )
    if non_calibrated:
        blockers.append(
            f"{non_calibrated} derived output contracts are not CALIBRATED"
        )
    missing_retrieval = sum(
        contract.source_retrieved_on is None for contract in derived
    )
    if missing_retrieval:
        blockers.append(
            f"{missing_retrieval} derived output contracts lack source retrieval dates"
        )
    if run_source_retrieved_on is None:
        blockers.append("run profile sources lack a retrieval date")
    if not monetary_outputs_cross_country_comparable:
        blockers.append(
            "simulation-cent outputs lack a dated cross-country FX/PPP contract"
        )
    return tuple(blockers)


def validate_metric_contracts_for_campaign(
    *,
    configuration_status: str,
    profile_lineage_status: str,
    profile_dependencies_calibrated: bool,
    run_source_retrieved_on: date | None,
    monetary_outputs_cross_country_comparable: bool,
) -> None:
    """Fail closed unless every output and upstream campaign contract is ready."""

    blockers = metric_contract_campaign_blockers(
        configuration_status=configuration_status,
        profile_lineage_status=profile_lineage_status,
        profile_dependencies_calibrated=profile_dependencies_calibrated,
        run_source_retrieved_on=run_source_retrieved_on,
        monetary_outputs_cross_country_comparable=(
            monetary_outputs_cross_country_comparable
        ),
    )
    if blockers:
        raise MetricContractValidationError(
            "output metric campaign validation failed: " + "; ".join(blockers)
        )


def build_metric_contract_manifest_payload(
    *,
    configuration_status: str,
    profile_lineage_status: str,
    profile_dependencies_calibrated: bool,
    profile_input_fingerprint_sha256: str | None,
    run_source_retrieved_on: date | None,
    monetary_outputs_cross_country_comparable: bool,
) -> dict[str, object]:
    """Build the exact registry snapshot and run-specific lineage pointers."""

    blockers = metric_contract_campaign_blockers(
        configuration_status=configuration_status,
        profile_lineage_status=profile_lineage_status,
        profile_dependencies_calibrated=profile_dependencies_calibrated,
        run_source_retrieved_on=run_source_retrieved_on,
        monetary_outputs_cross_country_comparable=(
            monetary_outputs_cross_country_comparable
        ),
    )
    role_counts = Counter(
        contract.role.value for contract in OUTPUT_METRIC_CONTRACTS.values()
    )
    status_counts = Counter(
        contract.status.value for contract in OUTPUT_METRIC_CONTRACTS.values()
    )
    return {
        "schema_version": METRIC_CONTRACT_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "recipe_source_version": METRIC_RECIPE_SOURCE_VERSION,
        "registry_sha256": metric_contract_registry_sha256(),
        "contract_count": len(OUTPUT_METRIC_CONTRACTS),
        "role_counts": dict(sorted(role_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "run_input_lineage": {
            "profile_lineage_status": profile_lineage_status,
            "profile_dependencies_calibrated": profile_dependencies_calibrated,
            "profile_input_fingerprint_sha256": profile_input_fingerprint_sha256,
            "profile_source_retrieved_on": (
                run_source_retrieved_on.isoformat()
                if run_source_retrieved_on is not None
                else None
            ),
            "monetary_outputs_cross_country_comparable": (
                monetary_outputs_cross_country_comparable
            ),
        },
        "campaign_ready": not blockers,
        "campaign_blockers": list(blockers),
        "contracts": metric_contract_registry_snapshot(),
    }


__all__ = [
    "METRIC_CONTRACT_SCHEMA_VERSION",
    "METRIC_RECIPE_SOURCE_VERSION",
    "MetricContractValidationError",
    "MetricRole",
    "OUTPUT_METRIC_CONTRACTS",
    "OutputMetricContract",
    "OutputUnit",
    "build_metric_contract_manifest_payload",
    "metric_contract_campaign_blockers",
    "metric_contract_registry_sha256",
    "metric_contract_registry_snapshot",
    "validate_metric_contract_registry",
    "validate_metric_contracts_for_campaign",
]
