from __future__ import annotations

from collections.abc import (
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
)
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..metrics.outcomes import (
    HarmWeights,
    OutcomeSnapshot,
    _immutable_array_copy,
)
from ..config import SimulationConfig
from ..core.ledger import Ledger
from ..core.world import World
from ..simulation import RunResult, SimulationOrchestrator
from ..data.profiles import ProfileBundle
from .interventions import Intervention, NullIntervention


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


class BalanceMismatchKind(str, Enum):
    """Machine-readable reason that two pre-treatment values differ."""

    TYPE = "type"
    DTYPE = "dtype"
    SHAPE = "shape"
    LENGTH = "length"
    KEYS = "keys"
    ALIAS = "alias"
    SHARED_MUTABLE = "shared_mutable"
    VALUE = "value"


@dataclass(frozen=True, slots=True)
class PreTreatmentMismatch:
    """One exact mismatch in the treated/control initial-state graph."""

    path: str
    kind: BalanceMismatchKind

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise TypeError("pre-treatment mismatch path must be a non-empty string")
        if type(self.kind) is not BalanceMismatchKind:
            raise TypeError("pre-treatment mismatch kind must be BalanceMismatchKind")


@dataclass(frozen=True, slots=True)
class PreTreatmentBalanceReport:
    """Exhaustive exact-balance assessment retained with a paired run."""

    checked_paths: tuple[str, ...]
    mismatches: tuple[PreTreatmentMismatch, ...]

    def __post_init__(self) -> None:
        if type(self.checked_paths) is not tuple or any(
            type(path) is not str or not path for path in self.checked_paths
        ):
            raise TypeError("checked_paths must be a tuple of non-empty strings")
        if len(set(self.checked_paths)) != len(self.checked_paths):
            raise ValueError("checked_paths must be unique")
        if type(self.mismatches) is not tuple or any(
            type(item) is not PreTreatmentMismatch for item in self.mismatches
        ):
            raise TypeError(
                "mismatches must be a tuple of PreTreatmentMismatch instances"
            )
        mismatch_paths = tuple(item.path for item in self.mismatches)
        if len(set(mismatch_paths)) != len(mismatch_paths):
            raise ValueError("pre-treatment mismatch paths must be unique")
        if any(path not in self.checked_paths for path in mismatch_paths):
            raise ValueError("every mismatch path must also be a checked path")

    @property
    def balanced(self) -> bool:
        """Whether every declared initial-state value is exactly equal."""

        return not self.mismatches

    def require_balanced(self) -> None:
        """Fail closed before either branch receives an intervention."""

        if self.mismatches:
            raise PreTreatmentBalanceError(self)


class PreTreatmentBalanceError(ValueError):
    """Raised before treatment when paired initial states are not identical."""

    def __init__(self, report: PreTreatmentBalanceReport) -> None:
        self.report = report
        summary = ", ".join(
            f"{item.path} ({item.kind.value})" for item in report.mismatches[:5]
        )
        if len(report.mismatches) > 5:
            summary += f", ... +{len(report.mismatches) - 5} more"
        super().__init__("paired worlds differ before treatment: " + summary)


class NegativeControlValidationError(ValueError):
    """Raised when a paired run changes a declared negative-control outcome."""

    def __init__(self, *, field: str, nonzero_count: int) -> None:
        self.field = field
        self.nonzero_count = nonzero_count
        super().__init__(
            "paired run failed pre-treatment/exogenous negative control "
            f"{field}: {nonzero_count} nonzero differences"
        )


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    player_harm_difference: FloatArray
    player_spend_difference_cents: IntArray
    player_income_negative_control_difference_cents: IntArray
    player_debt_difference_cents: IntArray
    firm_margin_difference_cents: IntArray
    firm_cash_difference_cents: IntArray
    state_subsidy_difference_cents: IntArray

    def __post_init__(self) -> None:
        contracts = (
            ("player_harm_difference", np.dtype(np.float64), 2),
            ("player_spend_difference_cents", np.dtype(np.int64), 1),
            (
                "player_income_negative_control_difference_cents",
                np.dtype(np.int64),
                1,
            ),
            ("player_debt_difference_cents", np.dtype(np.int64), 1),
            ("firm_margin_difference_cents", np.dtype(np.int64), 1),
            ("firm_cash_difference_cents", np.dtype(np.int64), 1),
            ("state_subsidy_difference_cents", np.dtype(np.int64), 1),
        )
        for name, expected_dtype, expected_rank in contracts:
            values = getattr(self, name)
            if type(values) is not np.ndarray or values.dtype != expected_dtype:
                raise TypeError(
                    f"{name} must be a {expected_dtype.name} numpy array"
                )
            if values.ndim != expected_rank:
                raise ValueError(f"{name} must have rank {expected_rank}")
        if self.player_harm_difference.shape[1] != 7:
            raise ValueError(
                "player_harm_difference must retain seven harm dimensions"
            )
        if not np.all(np.isfinite(self.player_harm_difference)):
            raise ValueError("player_harm_difference must be finite")
        players = self.player_harm_difference.shape[0]
        for name in (
            "player_spend_difference_cents",
            "player_income_negative_control_difference_cents",
            "player_debt_difference_cents",
        ):
            if getattr(self, name).shape != (players,):
                raise ValueError(f"{name} is not aligned with paired players")
        firms = self.firm_cash_difference_cents.shape
        if self.firm_margin_difference_cents.shape != firms:
            raise ValueError("paired firm difference columns are not aligned")
        if self.state_subsidy_difference_cents.ndim != 1:
            raise ValueError(
                "state_subsidy_difference_cents must be one-dimensional"
            )
        for name, _, _ in contracts:
            object.__setattr__(
                self,
                name,
                _immutable_array_copy(getattr(self, name)),
            )

    @property
    def player_income_negative_control_passed(self) -> bool:
        """Whether pre-treatment/exogenous player income stayed unchanged."""

        return bool(
            np.all(self.player_income_negative_control_difference_cents == 0)
        )

    def require_negative_controls(self) -> None:
        """Fail a paired run when an exogenous diagnostic changes."""

        if not self.player_income_negative_control_passed:
            raise NegativeControlValidationError(
                field="player_income_cents",
                nonzero_count=int(
                    np.count_nonzero(
                        self.player_income_negative_control_difference_cents
                    )
                ),
            )


@dataclass(frozen=True, slots=True)
class RegimeEffect:
    estimand: str
    mean_composite_harm_effect: float
    total_spend_effect_cents: int
    total_debt_effect_cents: int
    total_operating_margin_effect_cents: int
    total_subsidy_effect_cents: int
    affected_player_share: float


@dataclass(frozen=True, slots=True)
class PairedWorldRun:
    """Outputs from two structurally identical counterfactual markets."""

    treated_run: RunResult
    control_run: RunResult
    pre_treatment_balance: PreTreatmentBalanceReport
    paired_outcome: PairedOutcome
    effect: RegimeEffect

    def __post_init__(self) -> None:
        if type(self.pre_treatment_balance) is not PreTreatmentBalanceReport:
            raise TypeError(
                "pre_treatment_balance must be PreTreatmentBalanceReport"
            )
        self.pre_treatment_balance.require_balanced()
        self.paired_outcome.require_negative_controls()


def compare_outcomes(
    treated: OutcomeSnapshot,
    control: OutcomeSnapshot,
    *,
    estimand: str = "market_regime_total_effect",
    weights: HarmWeights | None = None,
) -> tuple[PairedOutcome, RegimeEffect]:
    """Compare structurally paired worlds without regressing away vulnerability."""

    _validate_outcome_pair(treated, control, estimand=estimand, weights=weights)

    paired = PairedOutcome(
        player_harm_difference=_checked_float64_difference(
            treated.player_harm,
            control.player_harm,
            name="player_harm",
        ),
        player_spend_difference_cents=_checked_int64_difference(
            treated.player_spend_cents,
            control.player_spend_cents,
            name="player_spend_cents",
        ),
        player_income_negative_control_difference_cents=_checked_int64_difference(
            treated.player_income_cents,
            control.player_income_cents,
            name="player_income_cents",
        ),
        player_debt_difference_cents=_checked_int64_difference(
            treated.player_debt_cents,
            control.player_debt_cents,
            name="player_debt_cents",
        ),
        firm_margin_difference_cents=_checked_int64_difference(
            treated.firm_operating_margin_cents,
            control.firm_operating_margin_cents,
            name="firm_operating_margin_cents",
        ),
        firm_cash_difference_cents=_checked_int64_difference(
            treated.firm_cash_cents,
            control.firm_cash_cents,
            name="firm_cash_cents",
        ),
        state_subsidy_difference_cents=_checked_int64_difference(
            treated.state_subsidy_outlay_cents,
            control.state_subsidy_outlay_cents,
            name="state_subsidy_outlay_cents",
        ),
    )
    weight_array = (weights or HarmWeights()).as_array()
    with np.errstate(over="ignore", invalid="ignore"):
        individual_composite = paired.player_harm_difference @ (
            weight_array / weight_array.sum()
        )
    if not np.all(np.isfinite(individual_composite)):
        raise OverflowError("weighted paired composite harm is not finite")
    with np.errstate(over="ignore", invalid="ignore"):
        mean_composite = (
            float(individual_composite.mean())
            if len(individual_composite)
            else 0.0
        )
    if not np.isfinite(mean_composite):
        raise OverflowError("mean paired composite harm effect is not finite")
    affected_share = (
        float(np.count_nonzero(np.abs(individual_composite) > 1e-12))
        / len(individual_composite)
        if len(individual_composite)
        else 0.0
    )
    effect = RegimeEffect(
        estimand=estimand,
        mean_composite_harm_effect=mean_composite,
        total_spend_effect_cents=int(
            sum(int(value) for value in paired.player_spend_difference_cents)
        ),
        total_debt_effect_cents=int(
            sum(int(value) for value in paired.player_debt_difference_cents)
        ),
        total_operating_margin_effect_cents=int(
            sum(int(value) for value in paired.firm_margin_difference_cents)
        ),
        total_subsidy_effect_cents=int(
            sum(int(value) for value in paired.state_subsidy_difference_cents)
        ),
        affected_player_share=affected_share,
    )
    return paired, effect


def run_paired_worlds(
    config: SimulationConfig,
    *,
    treated: Intervention,
    control: Intervention | None = None,
    cycles: int | None = None,
    campaign: bool = False,
    profiles: ProfileBundle | None = None,
    treated_ledger: Ledger | None = None,
    control_ledger: Ledger | None = None,
) -> PairedWorldRun:
    """Run an explicit treated/control pair with common random numbers.

    Each branch owns separate mutable state. Counter-based random streams share
    coordinates, so an action occurring only in one branch cannot shift later
    exogenous draws in the other branch.
    """

    if not config.causal.common_random_numbers:
        raise ValueError("paired worlds require common_random_numbers=true")
    if (treated_ledger is None) != (control_ledger is None):
        raise ValueError("paired worlds require both branch ledgers or neither")
    if (
        treated_ledger is not None
        and control_ledger is not None
        and treated_ledger.shares_storage_with(control_ledger)
    ):
        raise ValueError("paired worlds require physically distinct ledger storage")
    control_intervention = control or NullIntervention()
    treated_profiles = _detached_profile_bundle(profiles)
    control_profiles = _detached_profile_bundle(profiles)
    treated_world: World | None = None
    control_world: World | None = None
    try:
        treated_world = World.create(
            config,
            profiles=treated_profiles,
            campaign=campaign,
            ledger=treated_ledger,
        )
        control_world = World.create(
            config,
            profiles=control_profiles,
            campaign=campaign,
            ledger=control_ledger,
        )
        pre_treatment_balance = assess_pre_treatment_balance(
            treated_world,
            control_world,
        )
        pre_treatment_balance.require_balanced()

        treated.apply(treated_world)
        control_intervention.apply(control_world)
        treated_run = SimulationOrchestrator.run(
            treated_world, cycles=cycles, campaign=campaign
        )
        control_run = SimulationOrchestrator.run(
            control_world, cycles=cycles, campaign=campaign
        )
        paired, effect = compare_outcomes(
            treated_run.final_outcome,
            control_run.final_outcome,
            estimand=config.causal.estimand,
        )
        paired.require_negative_controls()
        return PairedWorldRun(
            treated_run=treated_run,
            control_run=control_run,
            pre_treatment_balance=pre_treatment_balance,
            paired_outcome=paired,
            effect=effect,
        )
    finally:
        if control_world is not None:
            control_world.close()
        if treated_world is not None:
            treated_world.close()


def assess_pre_treatment_balance(
    treated: World,
    control: World,
) -> PreTreatmentBalanceReport:
    """Compare the complete initial world state without mutating either branch.

    Exact equality is the appropriate balance contract here because the paired
    worlds are intended to be independently allocated copies of one synthetic
    pre-treatment market, not two samples requiring a statistical balance test.
    The recursive walk covers every current dataclass field, array, mapping,
    sequence, object attribute, and slot reachable from the two ``World`` roots.
    """

    if type(treated) is not World or type(control) is not World:
        raise TypeError("pre-treatment balance requires two World instances")
    checked_paths: list[str] = []
    mismatches: list[PreTreatmentMismatch] = []
    _compare_balance_values(
        treated,
        control,
        path="world",
        checked_paths=checked_paths,
        mismatches=mismatches,
        traversal=_BalanceTraversal(),
    )
    return PreTreatmentBalanceReport(
        checked_paths=tuple(checked_paths),
        mismatches=tuple(mismatches),
    )


def _validate_outcome_pair(
    treated: OutcomeSnapshot,
    control: OutcomeSnapshot,
    *,
    estimand: str,
    weights: HarmWeights | None,
) -> None:
    if type(treated) is not OutcomeSnapshot or type(control) is not OutcomeSnapshot:
        raise TypeError("treated and control must be OutcomeSnapshot instances")
    if type(treated.tick) is not int or type(control.tick) is not int:
        raise TypeError("paired outcome ticks must be Python integers")
    if treated.tick < 0 or control.tick < 0:
        raise ValueError("paired outcome ticks cannot be negative")
    if treated.tick != control.tick:
        raise ValueError("paired outcomes must have the same tick")
    if type(estimand) is not str or not estimand.strip():
        raise TypeError("estimand must be a non-empty string")
    if weights is not None and type(weights) is not HarmWeights:
        raise TypeError("weights must be HarmWeights or None")

    domains = {
        "player_ids": (
            treated.player_ids,
            control.player_ids,
            np.dtype(np.int64),
            1,
        ),
        "player_harm": (
            treated.player_harm,
            control.player_harm,
            np.dtype(np.float64),
            2,
        ),
        "player_spend_cents": (
            treated.player_spend_cents,
            control.player_spend_cents,
            np.dtype(np.int64),
            1,
        ),
        "player_income_cents": (
            treated.player_income_cents,
            control.player_income_cents,
            np.dtype(np.int64),
            1,
        ),
        "player_debt_cents": (
            treated.player_debt_cents,
            control.player_debt_cents,
            np.dtype(np.int64),
            1,
        ),
        "firm_ids": (
            treated.firm_ids,
            control.firm_ids,
            np.dtype(np.int64),
            1,
        ),
        "firm_cash_cents": (
            treated.firm_cash_cents,
            control.firm_cash_cents,
            np.dtype(np.int64),
            1,
        ),
        "firm_operating_margin_cents": (
            treated.firm_operating_margin_cents,
            control.firm_operating_margin_cents,
            np.dtype(np.int64),
            1,
        ),
        "firm_safe_revenue_share": (
            treated.firm_safe_revenue_share,
            control.firm_safe_revenue_share,
            np.dtype(np.float64),
            1,
        ),
        "jurisdiction_ids": (
            treated.jurisdiction_ids,
            control.jurisdiction_ids,
            np.dtype(np.int64),
            1,
        ),
        "state_subsidy_outlay_cents": (
            treated.state_subsidy_outlay_cents,
            control.state_subsidy_outlay_cents,
            np.dtype(np.int64),
            1,
        ),
    }
    for name, (
        treated_values,
        control_values,
        expected_dtype,
        expected_rank,
    ) in domains.items():
        if (
            type(treated_values) is not np.ndarray
            or type(control_values) is not np.ndarray
        ):
            raise TypeError(f"paired {name} values must be numpy arrays")
        if (
            treated_values.dtype != expected_dtype
            or control_values.dtype != expected_dtype
        ):
            raise ValueError(
                f"paired {name} values must have dtype {expected_dtype.name}"
            )
        if (
            treated_values.ndim != expected_rank
            or control_values.ndim != expected_rank
        ):
            raise ValueError(
                f"paired {name} values must have rank {expected_rank}"
            )
        if treated_values.shape != control_values.shape:
            raise ValueError(f"paired {name} values must have the same shape")
        if np.issubdtype(expected_dtype, np.floating) and (
            not np.all(np.isfinite(treated_values))
            or not np.all(np.isfinite(control_values))
        ):
            raise ValueError(f"paired {name} values must be finite")

    for name in ("player_ids", "firm_ids", "jurisdiction_ids"):
        for snapshot in (treated, control):
            values = getattr(snapshot, name)
            if np.any(values < 0):
                raise ValueError(f"paired {name} values must be non-negative")
            if not _has_unique_integer_ids(values):
                raise ValueError(f"paired {name} values must be unique")
        if not np.array_equal(getattr(treated, name), getattr(control, name)):
            raise ValueError(f"paired outcomes must have the same ordered {name}")
    if treated.player_harm.shape[1] != 7:
        raise ValueError("paired player_harm values must retain seven dimensions")
    player_count = treated.player_ids.size
    if treated.player_harm.shape[0] != player_count:
        raise ValueError("paired player_harm values are not aligned with player_ids")
    for name in ("player_spend_cents", "player_income_cents", "player_debt_cents"):
        if getattr(treated, name).shape != (player_count,):
            raise ValueError(f"paired {name} values are not aligned with players")
    firm_count = treated.firm_ids.size
    for name in (
        "firm_cash_cents",
        "firm_operating_margin_cents",
        "firm_safe_revenue_share",
    ):
        if getattr(treated, name).shape != (firm_count,):
            raise ValueError(f"paired {name} values are not aligned with firms")
    if treated.state_subsidy_outlay_cents.shape != (
        treated.jurisdiction_ids.size,
    ):
        raise ValueError(
            "paired state_subsidy_outlay_cents values are not aligned with "
            "jurisdictions"
        )
    for snapshot in (treated, control):
        harm = snapshot.player_harm
        if np.any((harm < 0.0) | (harm > 1.0)):
            raise ValueError("paired player_harm values must be in [0, 1]")
        safe_share = snapshot.firm_safe_revenue_share
        if np.any((safe_share < 0.0) | (safe_share > 1.0)):
            raise ValueError(
                "paired firm_safe_revenue_share values must be in [0, 1]"
            )


def _checked_float64_difference(
    treated: FloatArray,
    control: FloatArray,
    *,
    name: str,
) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = treated - control
    if not np.all(np.isfinite(difference)):
        raise OverflowError(f"paired {name} difference is not finite")
    return difference


def _has_unique_integer_ids(values: IntArray) -> bool:
    if values.size < 2 or bool(np.all(values[1:] > values[:-1])):
        return True
    return np.unique(values).size == values.size


def _checked_int64_difference(
    treated: IntArray,
    control: IntArray,
    *,
    name: str,
) -> IntArray:
    differences: list[int] = []
    for treated_value, control_value in zip(
        treated.ravel(),
        control.ravel(),
        strict=True,
    ):
        difference = int(treated_value) - int(control_value)
        if difference < _INT64_MIN or difference > _INT64_MAX:
            raise OverflowError(f"paired {name} difference exceeds int64")
        differences.append(difference)
    return np.asarray(differences, dtype=np.int64).reshape(treated.shape)


@dataclass(slots=True)
class _BalanceTraversal:
    """Identity and buffer topology observed while comparing two object graphs."""

    visited: set[tuple[int, int]] = field(default_factory=set)
    treated_aliases: dict[int, int] = field(default_factory=dict)
    control_aliases: dict[int, int] = field(default_factory=dict)
    treated_mutable_ids: set[int] = field(default_factory=set)
    control_mutable_ids: set[int] = field(default_factory=set)
    treated_arrays: list[np.ndarray] = field(default_factory=list)
    control_arrays: list[np.ndarray] = field(default_factory=list)
    array_pairs: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def _compare_balance_values(
    treated: object,
    control: object,
    *,
    path: str,
    checked_paths: list[str],
    mismatches: list[PreTreatmentMismatch],
    traversal: _BalanceTraversal,
) -> None:
    if type(treated) is not type(control):
        _record_balance_check(
            path,
            BalanceMismatchKind.TYPE,
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        return

    if isinstance(
        treated,
        (str, bytes, int, float, bool, type(None), Enum, Path, np.generic),
    ):
        _record_balance_check(
            path,
            None if treated == control else BalanceMismatchKind.VALUE,
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        return

    topology_mismatch = _register_mutable_topology(
        treated,
        control,
        traversal=traversal,
    )
    if topology_mismatch is not None:
        _record_balance_check(
            path,
            topology_mismatch,
            checked_paths=checked_paths,
            mismatches=mismatches,
        )

    treated_id = id(treated)
    control_id = id(control)

    pair = (treated_id, control_id)
    if pair in traversal.visited:
        _record_balance_check(
            path,
            None,
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        return
    traversal.visited.add(pair)

    if isinstance(treated, Ledger):
        _record_balance_check(
            f"{path}.storage",
            (
                BalanceMismatchKind.SHARED_MUTABLE
                if treated.shares_storage_with(control)
                else None
            ),
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        _compare_balance_values(
            treated.balance_snapshot(),
            control.balance_snapshot(),
            path=f"{path}.logical_state",
            checked_paths=checked_paths,
            mismatches=mismatches,
            traversal=traversal,
        )
        return

    if isinstance(treated, np.ndarray):
        kind: BalanceMismatchKind | None = None
        if treated.dtype != control.dtype:  # type: ignore[union-attr]
            kind = BalanceMismatchKind.DTYPE
        elif treated.shape != control.shape:  # type: ignore[union-attr]
            kind = BalanceMismatchKind.SHAPE
        elif not np.array_equal(treated, control):
            kind = BalanceMismatchKind.VALUE
        _record_balance_check(
            path,
            kind,
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        return

    if is_dataclass(treated) and not isinstance(treated, type):
        for descriptor in fields(treated):
            _compare_balance_values(
                getattr(treated, descriptor.name),
                getattr(control, descriptor.name),
                path=f"{path}.{descriptor.name}",
                checked_paths=checked_paths,
                mismatches=mismatches,
                traversal=traversal,
            )
        return

    if isinstance(treated, Mapping):
        treated_keys = set(treated)
        control_keys = set(control)  # type: ignore[arg-type]
        _record_balance_check(
            f"{path}.keys",
            (
                None
                if treated_keys == control_keys
                else BalanceMismatchKind.KEYS
            ),
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        for key in sorted(treated_keys & control_keys, key=repr):
            _compare_balance_values(
                treated[key],
                control[key],  # type: ignore[index]
                path=f"{path}[{key!r}]",
                checked_paths=checked_paths,
                mismatches=mismatches,
                traversal=traversal,
            )
        return

    if isinstance(treated, Sequence) and not isinstance(treated, (str, bytes)):
        _record_balance_check(
            f"{path}.length",
            (
                None
                if len(treated) == len(control)  # type: ignore[arg-type]
                else BalanceMismatchKind.LENGTH
            ),
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        for index, (treated_item, control_item) in enumerate(
            zip(treated, control, strict=False)  # type: ignore[arg-type]
        ):
            _compare_balance_values(
                treated_item,
                control_item,
                path=f"{path}[{index}]",
                checked_paths=checked_paths,
                mismatches=mismatches,
                traversal=traversal,
            )
        return

    if isinstance(treated, (set, frozenset)):
        _record_balance_check(
            path,
            None if treated == control else BalanceMismatchKind.VALUE,
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        return

    attributes = _balance_attributes(treated)
    control_attributes = _balance_attributes(control)
    if attributes is not None or control_attributes is not None:
        treated_attributes = attributes or {}
        paired_attributes = control_attributes or {}
        treated_names = set(treated_attributes)
        control_names = set(paired_attributes)
        _record_balance_check(
            f"{path}.__attributes__",
            (
                None
                if treated_names == control_names
                else BalanceMismatchKind.KEYS
            ),
            checked_paths=checked_paths,
            mismatches=mismatches,
        )
        for name in sorted(treated_names & control_names):
            _compare_balance_values(
                treated_attributes[name],
                paired_attributes[name],
                path=f"{path}.{name}",
                checked_paths=checked_paths,
                mismatches=mismatches,
                traversal=traversal,
            )
        return

    try:
        equal = bool(treated == control)
    except (TypeError, ValueError):
        equal = False
    _record_balance_check(
        path,
        None if equal else BalanceMismatchKind.VALUE,
        checked_paths=checked_paths,
        mismatches=mismatches,
    )


def _balance_attributes(value: object) -> dict[str, object] | None:
    attributes: dict[str, object] = {}
    has_structural_attributes = hasattr(value, "__dict__")
    if hasattr(value, "__dict__"):
        attributes.update(vars(value))
    for cls in type(value).__mro__:
        if "__slots__" in cls.__dict__:
            has_structural_attributes = True
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__") or name in attributes:
                continue
            if hasattr(value, name):
                attributes[name] = getattr(value, name)
    return attributes if has_structural_attributes else None


def _is_mutable_balance_value(value: object) -> bool:
    if isinstance(
        value,
        (np.ndarray, MutableMapping, MutableSequence, MutableSet),
    ):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        return parameters is None or not bool(parameters.frozen)
    attributes = _balance_attributes(value)
    return bool(attributes)


def _register_mutable_topology(
    treated: object,
    control: object,
    *,
    traversal: _BalanceTraversal,
) -> BalanceMismatchKind | None:
    """Retain mutable alias relations and reject any cross-branch ownership."""

    treated_mutable = _is_mutable_balance_value(treated)
    control_mutable = _is_mutable_balance_value(control)
    if not treated_mutable and not control_mutable:
        return None

    treated_id = id(treated)
    control_id = id(control)
    shared_mutable = (
        (treated_mutable and treated_id in traversal.control_mutable_ids)
        or (control_mutable and control_id in traversal.treated_mutable_ids)
        or (treated_mutable and control_mutable and treated is control)
    )
    alias_mismatch = False
    if treated_mutable and control_mutable:
        alias_mismatch = (
            treated_id in traversal.treated_aliases
            and traversal.treated_aliases[treated_id] != control_id
        ) or (
            control_id in traversal.control_aliases
            and traversal.control_aliases[control_id] != treated_id
        )

    if isinstance(treated, np.ndarray) and isinstance(control, np.ndarray):
        shared_mutable = shared_mutable or bool(np.shares_memory(treated, control))
        shared_mutable = shared_mutable or any(
            np.shares_memory(treated, prior_control)
            for prior_control in traversal.control_arrays
        )
        shared_mutable = shared_mutable or any(
            np.shares_memory(control, prior_treated)
            for prior_treated in traversal.treated_arrays
        )
        alias_mismatch = alias_mismatch or any(
            bool(np.shares_memory(treated, prior_treated))
            != bool(np.shares_memory(control, prior_control))
            for prior_treated, prior_control in traversal.array_pairs
        )
        traversal.treated_arrays.append(treated)
        traversal.control_arrays.append(control)
        traversal.array_pairs.append((treated, control))

    if treated_mutable:
        traversal.treated_mutable_ids.add(treated_id)
    if control_mutable:
        traversal.control_mutable_ids.add(control_id)
    if treated_mutable and control_mutable:
        traversal.treated_aliases.setdefault(treated_id, control_id)
        traversal.control_aliases.setdefault(control_id, treated_id)

    if shared_mutable:
        return BalanceMismatchKind.SHARED_MUTABLE
    if alias_mismatch:
        return BalanceMismatchKind.ALIAS
    return None


def _detached_profile_bundle(profiles: ProfileBundle | None) -> ProfileBundle | None:
    """Copy mutable jurisdiction templates while sharing immutable metadata."""

    if profiles is None:
        return None
    return replace(
        profiles,
        state_agents=tuple(deepcopy(state) for state in profiles.state_agents),
    )


def _record_balance_check(
    path: str,
    mismatch_kind: BalanceMismatchKind | None,
    *,
    checked_paths: list[str],
    mismatches: list[PreTreatmentMismatch],
) -> None:
    if path not in checked_paths:
        checked_paths.append(path)
    if mismatch_kind is not None and not any(
        item.path == path for item in mismatches
    ):
        mismatches.append(PreTreatmentMismatch(path, mismatch_kind))
