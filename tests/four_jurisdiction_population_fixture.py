"""Reusable exact four-jurisdiction projected-population test fixture.

The files written here are algebraic test inputs.  Their missing signature and
schema-v1 population contracts remain explicitly campaign-ineligible.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from microtx_sim.data.population_design import (
    CANONICAL_SOURCE_CLUSTER_ID_V1,
    CANONICAL_SOURCE_RECORD_ID_V1,
    EXACT_RATIONAL_HAMILTON_V1,
    SHA256_CLUSTER_THRESHOLD_V1,
    apportion_population_hamilton,
    assigned_population_partition_role,
    build_population_calibration_target,
    load_and_verify_population_design_bundle,
)
from microtx_sim.data.population_evidence import (
    POPULATION_CELL_CSV_COLUMNS,
    PopulationEstimandRole,
    exact_csv_joint_population_recipe_json,
    load_and_verify_population_evidence_bundle,
)
from microtx_sim.data.population_projection import (
    RUNTIME_INCOME_CONCEPT,
    SOURCE_INCOME_CONCEPT,
    PopulationProjectionAdapter,
    build_population_projection_adapter,
    load_population_runtime_mapping_bundle,
    verify_population_projection_adapter,
)
from microtx_sim.data.profiles import DEFAULT_SOURCES_PATH


DEFAULT_PLAYER_COUNT = 16
REGISTERED_PROFILE_CODES = ("UK", "KR", "JP", "BE")
CANONICAL_DESIGN_CODES = tuple(sorted(REGISTERED_PROFILE_CODES))

_JURISDICTIONS = {
    "BE": ("Belgium", "EUR"),
    "JP": ("Japan", "JPY"),
    "KR": ("South Korea", "KRW"),
    "UK": ("United Kingdom", "GBP"),
}
_PARTITION_SEED = "4" * 64
_IDENTITY_NAMESPACE = "test.four-jurisdiction.population.units"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _binding_id(code: str, role: PopulationEstimandRole) -> str:
    role_offset = 0 if role is PopulationEstimandRole.CALIBRATION else 1
    ordinal = CANONICAL_DESIGN_CODES.index(code) * 2 + role_offset
    return f"{ordinal:02d}.{code.lower()}.{role.value.lower()}"


def _target_id(code: str, role: PopulationEstimandRole) -> str:
    return f"{code.lower()}.fixture.{role.value.lower()}"


def _source_id(code: str, role: PopulationEstimandRole) -> str:
    return f"POP_FIXTURE_{code}_{role.value}"


def _write_population_evidence(
    root: Path,
) -> tuple[object, tuple[object, ...]]:
    source_blocks: list[str] = []
    for code in CANONICAL_DESIGN_CODES:
        geography, _currency = _JURISDICTIONS[code]
        for role in PopulationEstimandRole:
            source_id = _source_id(code, role)
            source_blocks.append(
                f'''
[[source]]
id = "{source_id}"
publisher = "Test population fixture"
title = "Test-only {role.value.lower()} cells for {code}"
url = "https://example.invalid/{source_id.lower()}"
period = "2025"
geography = "{geography}"
supports = [
    "age_structure",
    "income_distribution",
    "household_composition",
    "gaming_reach",
    "conditional_payer_rate",
]
calibration_status = "CALIBRATED"
'''
            )
    sources_path = root / "population_sources.toml"
    sources_path.write_text(
        DEFAULT_SOURCES_PATH.read_text(encoding="utf-8")
        + "".join(source_blocks),
        encoding="utf-8",
        newline="",
    )
    source_registry_sha256 = sha256(sources_path.read_bytes()).hexdigest()

    rows = [list(POPULATION_CELL_CSV_COLUMNS)]
    for code in CANONICAL_DESIGN_CODES:
        for role in PopulationEstimandRole:
            cell_ordinal = 0
            for gaming_state in ("GAMER", "NON_GAMER"):
                for payer_state in ("EVER_PAYER", "NEVER_PAYER"):
                    rows.append(
                        [
                            _target_id(code, role),
                            code,
                            role.value,
                            f"cell.{cell_ordinal:02d}",
                            "10",
                            "70",
                            "income.all",
                            "household.all",
                            gaming_state,
                            payer_state,
                            "1",
                            "4",
                        ]
                    )
                    cell_ordinal += 1
    content = ("\n".join(",".join(row) for row in rows) + "\n").encode("utf-8")
    artifact_root = root / "population_artifacts"
    artifact_root.mkdir()
    (artifact_root / "joint.csv").write_bytes(content)

    bindings: list[str] = []
    for code in CANONICAL_DESIGN_CODES:
        geography, currency = _JURISDICTIONS[code]
        for role in PopulationEstimandRole:
            recipe = exact_csv_joint_population_recipe_json(
                target_population_id=_target_id(code, role),
                jurisdiction_code=code,
                estimand_role=role,
            )
            bindings.append(
                f'''[[bindings]]
binding_id = "{_binding_id(code, role)}"
artifact_id = "joint.population"
target_population_id = "{_target_id(code, role)}"
jurisdiction_code = "{code}"
geography = "{geography}"
reference_period_start = 2025-01-01
reference_period_end = 2025-12-31
population_base = "resident population"
universe = "mobile-game players and non-players"
unit_of_analysis = "person"
eligibility = "usual residents aged 10 to 69"
exclusion = "institutional residents"
age_min_inclusive = 10
age_max_exclusive = 70
household_income_definition = "gross household income"
household_income_currency = "{currency}"
household_income_period = "annual"
household_income_equivalisation = "none"
household_definition = "shared dwelling and budget"
gaming_definition = "mobile-game play in reference period"
payer_definition = "ever paid before the reference-period end"
zero_spender_treatment = "retained as never-payer cells"
estimand_role = "{role.value}"
status = "CALIBRATED"
source_ids = ["{_source_id(code, role)}"]
retrieved_on = 2026-08-30
recipe_json = {json.dumps(recipe)}

'''
            )

    bundle_path = root / "population_bundle.toml"
    bundle_path.write_text(
        f'''schema_version = 1
bundle_id = "test-four-jurisdiction-population-evidence"
provenance_status = "CALIBRATED"
source_registry_sha256 = "{source_registry_sha256}"
artifact_root = "population_artifacts"
notes = "Test-only exact population fixture; not substantive evidence."

[[artifacts]]
artifact_id = "joint.population"
relative_path = "joint.csv"
media_type = "text/csv"
sha256 = "{sha256(content).hexdigest()}"
byte_length = {len(content)}

{"".join(bindings)}[signature]
status = "MISSING"
algorithm = "NONE"
key_id = ""
value = ""
''',
        encoding="utf-8",
        newline="",
    )
    return load_and_verify_population_evidence_bundle(
        bundle_path,
        expected_source_registry_sha256=source_registry_sha256,
    )


def _cluster_for_role(
    role: PopulationEstimandRole,
    semantic_key: str,
) -> str:
    candidate = 0
    while True:
        digest = sha256(
            f"cluster:{semantic_key}:{candidate}".encode("utf-8")
        ).hexdigest()
        assigned = assigned_population_partition_role(
            identity_namespace=_IDENTITY_NAMESPACE,
            assignment_seed_sha256=_PARTITION_SEED,
            cluster_identity_sha256=digest,
            calibration_threshold_numerator=1,
            calibration_threshold_denominator=2,
        )
        if assigned is role:
            return digest
        candidate += 1


def _write_population_design(
    root: Path,
    evidence_bundle: object,
    evidence_results: tuple[object, ...],
) -> object:
    results = {
        (result.jurisdiction_code, result.estimand_role): result
        for result in evidence_results
    }
    income_tables: list[str] = []
    jurisdiction_tables: list[str] = []
    for code in CANONICAL_DESIGN_CODES:
        _geography, currency = _JURISDICTIONS[code]
        income_tables.append(
            f'''[[domains.income_bands]]
ordinal = 0
jurisdiction_code = "{code}"
income_band_id = "income.all"
definition = "all harmonized household-income categories"
currency = "{currency}"
period = "annual"
lower_unbounded = true
lower_bound_numerator = 0
lower_bound_denominator = 1
upper_unbounded = true
upper_bound_numerator = 0
upper_bound_denominator = 1

'''
        )
        calibration = results[(code, PopulationEstimandRole.CALIBRATION)]
        validation = results[(code, PopulationEstimandRole.VALIDATION)]
        jurisdiction_tables.append(
            f'''[[jurisdictions]]
jurisdiction_code = "{code}"
target_population_count = 400
calibration_binding_id = "{calibration.binding_id}"
calibration_target_population_id = "{calibration.target_population_id}"
calibration_evidence_sha256 = "{calibration.evidence_sha256}"
validation_binding_id = "{validation.binding_id}"
validation_target_population_id = "{validation.target_population_id}"
validation_evidence_sha256 = "{validation.evidence_sha256}"

'''
        )

    records: list[tuple[str, str]] = []
    for result in evidence_results:
        for cell in result.cells:
            semantic_key = f"{result.binding_id}:{cell.cell_id}"
            record_id = sha256(f"record:{semantic_key}".encode("utf-8")).hexdigest()
            cluster_id = _cluster_for_role(result.estimand_role, semantic_key)
            records.append(
                (
                    record_id,
                    f'''[[partition.records]]
record_identity_sha256 = "{record_id}"
cluster_identity_sha256 = "{cluster_id}"
estimand_role = "{result.estimand_role.value}"
binding_id = "{result.binding_id}"
cell_id = "{cell.cell_id}"
record_weight_numerator = {cell.target_mass_numerator}
record_weight_denominator = {cell.target_mass_denominator}

''',
                )
            )
    record_tables = "".join(text for _record_id, text in sorted(records))
    result_digests = ", ".join(
        f'"{result.evidence_sha256}"' for result in evidence_results
    )
    design_path = root / "population_design.toml"
    design_path.write_text(
        f'''schema_version = 1
design_id = "test-four-jurisdiction-population-design"
provenance_status = "CALIBRATED"
notes = "Test-only exact static design; authenticity remains unverified."
population_evidence_bundle_sha256 = "{evidence_bundle.bundle_sha256}"
population_evidence_result_sha256s = [{result_digests}]
hamilton_recipe = "{EXACT_RATIONAL_HAMILTON_V1}"

[domains]
income_missing_policy = "REJECT"
household_missing_policy = "REJECT"
gaming_states = ["GAMER", "NON_GAMER"]
payer_history_states = ["EVER_PAYER", "NEVER_PAYER"]

[[domains.age_bands]]
ordinal = 0
age_band_id = "age.10-69"
age_min_inclusive = 10
age_max_exclusive = 70

{"".join(income_tables)}[[domains.household_types]]
ordinal = 0
household_type_id = "household.all"
definition = "all declared source household types"

{"".join(jurisdiction_tables)}[partition]
identity_namespace = "{_IDENTITY_NAMESPACE}"
record_id_recipe = "{CANONICAL_SOURCE_RECORD_ID_V1}"
cluster_id_recipe = "{CANONICAL_SOURCE_CLUSTER_ID_V1}"
role_assignment_recipe = "{SHA256_CLUSTER_THRESHOLD_V1}"
assignment_seed_sha256 = "{_PARTITION_SEED}"
calibration_threshold_numerator = 1
calibration_threshold_denominator = 2

{record_tables}''',
        encoding="utf-8",
        newline="",
    )
    return load_and_verify_population_design_bundle(
        design_path,
        population_evidence_bundle=evidence_bundle,
        population_evidence_results=evidence_results,
    )


def _write_runtime_mapping(root: Path, verification: object) -> object:
    entries: list[dict[str, object]] = []
    for income_band in verification.bundle.income_bands:
        _geography, currency = _JURISDICTIONS[income_band.jurisdiction_code]
        for household in verification.bundle.household_types:
            recipe_id = (
                f"fixture.{income_band.jurisdiction_code.lower()}.income.all"
            )
            entries.append(
                {
                    "jurisdiction_code": income_band.jurisdiction_code,
                    "source_household_income_band_id": income_band.income_band_id,
                    "source_household_income_definition": income_band.definition,
                    "source_household_income_currency": income_band.currency,
                    "source_household_income_period": income_band.period,
                    "source_household_income_lower_unbounded": (
                        income_band.lower_unbounded
                    ),
                    "source_household_income_lower_bound": [
                        income_band.lower_bound_numerator,
                        income_band.lower_bound_denominator,
                    ],
                    "source_household_income_upper_unbounded": (
                        income_band.upper_unbounded
                    ),
                    "source_household_income_upper_bound": [
                        income_band.upper_bound_numerator,
                        income_band.upper_bound_denominator,
                    ],
                    "source_household_type_id": household.household_type_id,
                    "source_household_type_definition": household.definition,
                    "runtime_personal_monthly_disposable_income_band_id": (
                        "runtime.personal.monthly.income.all"
                    ),
                    "runtime_personal_monthly_disposable_income_currency": currency,
                    "runtime_personal_monthly_disposable_income_min_cents": 10_000,
                    "runtime_personal_monthly_disposable_income_max_cents_exclusive": (
                        20_000
                    ),
                    "modeled_players_per_household": 1,
                    "conversion_recipe_id": recipe_id,
                    "conversion_recipe_sha256": sha256(
                        recipe_id.encode("utf-8")
                    ).hexdigest(),
                }
            )
    entries.sort(
        key=lambda entry: (
            entry["jurisdiction_code"],
            entry["source_household_income_band_id"],
            entry["source_household_type_id"],
        )
    )
    payload = {
        "schema_version": 1,
        "mapping_id": "test.four-jurisdiction.runtime-mapping",
        "design_id": verification.bundle.design_id,
        "design_bundle_sha256": verification.bundle.bundle_sha256,
        "domain_sha256": verification.bundle.domain_sha256,
        "source_income_concept": SOURCE_INCOME_CONCEPT,
        "runtime_income_concept": RUNTIME_INCOME_CONCEPT,
        "entries": entries,
    }
    mapping_path = root / "runtime_mapping.json"
    mapping_path.write_text(
        _canonical_json(payload) + "\n",
        encoding="utf-8",
        newline="",
    )
    return load_population_runtime_mapping_bundle(mapping_path)


def write_four_jurisdiction_population_fixture(
    root: Path,
    *,
    player_count: int = DEFAULT_PLAYER_COUNT,
    first_player_id: int = 0,
) -> PopulationProjectionAdapter:
    """Write exact file-backed inputs and return their verified adapter.

    Static design jurisdictions use the schema-required canonical order.  Runtime
    initialization may supply the same unique code set in profile-bundle order.
    """

    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    root.mkdir(parents=True, exist_ok=True)
    evidence_bundle, evidence_results = _write_population_evidence(root)
    verification = _write_population_design(
        root,
        evidence_bundle,
        evidence_results,
    )
    target = build_population_calibration_target(verification)
    plan = apportion_population_hamilton(
        target,
        player_count,
        first_player_id=first_player_id,
    )
    mapping = _write_runtime_mapping(root, verification)
    adapter = build_population_projection_adapter(
        verification,
        plan,
        mapping,
        adapter_id="test.four-jurisdiction.population-projection",
    )
    verified = verify_population_projection_adapter(adapter)
    if verified.campaign_ready:
        raise AssertionError("test-only population fixture cannot be campaign-ready")
    return verified


__all__ = [
    "CANONICAL_DESIGN_CODES",
    "DEFAULT_PLAYER_COUNT",
    "REGISTERED_PROFILE_CODES",
    "write_four_jurisdiction_population_fixture",
]
