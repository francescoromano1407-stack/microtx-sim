"""Build the checked-in illustrative joint-population declarations.

This is a deterministic artifact builder, not a campaign command.  It records
one complete modeled joint distribution for development and validation while
leaving every empirical/campaign readiness flag closed.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re

from microtx_sim.data.population_design import (
    CANONICAL_SOURCE_CLUSTER_ID_V1,
    CANONICAL_SOURCE_RECORD_ID_V1,
    EXACT_RATIONAL_HAMILTON_V1,
    SHA256_CLUSTER_THRESHOLD_V1,
    assigned_population_partition_role,
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
    load_population_runtime_mapping_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_ROOT = PROJECT_ROOT / "data" / "provenance"
SOURCES_PATH = PROVENANCE_ROOT / "sources.toml"
SOURCE_BUNDLE_PATH = PROVENANCE_ROOT / "source_bundle.toml"
POPULATION_BUNDLE_PATH = PROVENANCE_ROOT / "population_bundle.toml"
POPULATION_DESIGN_PATH = PROVENANCE_ROOT / "population_design.toml"
POPULATION_RUNTIME_MAPPING_PATH = (
    PROVENANCE_ROOT / "population_runtime_mapping.json"
)
POPULATION_ARTIFACT_ROOT = PROVENANCE_ROOT / "population_artifacts"
POPULATION_CSV_PATH = POPULATION_ARTIFACT_ROOT / "joint_population.csv"

DESIGN_ID = "standardized-four-country-person-population-2024-v1"
EVIDENCE_BUNDLE_ID = "illustrative-four-country-joint-population-2024-v1"
ARTIFACT_ID = "joint.population.2024"
IDENTITY_NAMESPACE = "microtx-sim.population.standardized-person-2024"
PARTITION_SEED_SHA256 = sha256(
    b"microtx-sim illustrative population partition 2024 v1"
).hexdigest()
TARGET_COUNT_PER_JURISDICTION = 10_000

JURISDICTIONS = {
    "BE": {
        "name": "Belgium",
        "currency": "EUR",
        "source_id": "MICROTX_POP_MODEL_BE_2024",
        "age": (11, 12, 17, 18, 18, 24),
        "gaming": (84, 72, 59, 48, 38, 27),
        "income_shift": 0,
    },
    "JP": {
        "name": "Japan",
        "currency": "JPY",
        "source_id": "MICROTX_POP_MODEL_JP_2024",
        "age": (8, 10, 16, 19, 22, 25),
        "gaming": (85, 73, 63, 52, 41, 29),
        "income_shift": 2,
    },
    "KR": {
        "name": "South Korea",
        "currency": "KRW",
        "source_id": "MICROTX_POP_MODEL_KR_2024",
        "age": (9, 12, 18, 20, 22, 19),
        "gaming": (82, 78, 71, 62, 50, 35),
        "income_shift": 4,
    },
    "UK": {
        "name": "United Kingdom",
        "currency": "GBP",
        "source_id": "MICROTX_POP_MODEL_UK_2024",
        "age": (12, 12, 17, 18, 18, 23),
        "gaming": (90, 79, 67, 54, 43, 31),
        "income_shift": 0,
    },
}

AGE_BANDS = (
    ("age.10-17", 10, 18),
    ("age.18-24", 18, 25),
    ("age.25-34", 25, 35),
    ("age.35-44", 35, 45),
    ("age.45-54", 45, 55),
    ("age.55-69", 55, 70),
)
INCOME_BANDS = ("income.low", "income.middle", "income.high")
INCOME_DEFINITIONS = (
    "lower harmonized modeled annual equivalised-household-income band",
    "middle harmonized modeled annual equivalised-household-income band",
    "upper harmonized modeled annual equivalised-household-income band",
)
INCOME_BOUNDS = {
    "BE": (24_000, 48_000),
    "JP": (3_000_000, 6_000_000),
    "KR": (30_000_000, 60_000_000),
    "UK": (24_000, 48_000),
}
HOUSEHOLD_TYPES = (
    (
        "household.one-person",
        "one-person modeled household with no co-resident minor",
    ),
    (
        "household.multi-no-minor",
        "multi-person modeled household with no co-resident minor",
    ),
    (
        "household.with-minor",
        "modeled household containing at least one person aged under 18",
    ),
)
# Exercise source-to-canonical mapping explicitly: evidence rows use this
# source order while the design domain retains canonical ordinals 0, 1, 2.
SOURCE_HOUSEHOLD_ORDINALS = (2, 0, 1)
HOUSEHOLD_SHARES = (
    (0, 0, 100),
    (45, 45, 10),
    (35, 50, 15),
    (25, 55, 20),
    (27, 65, 8),
    (38, 60, 2),
)
RUNTIME_INCOME_MODELS = {
    "BE": (
        (50_000, 120_000, 199_999, (11, 20)),
        (200_000, 285_000, 399_999, (9, 20)),
        (400_000, 550_000, 999_999, (11, 20)),
    ),
    "JP": (
        (50_000, 160_000, 249_999, (29, 50)),
        (250_000, 340_000, 499_999, (12, 25)),
        (500_000, 650_000, 1_499_999, (29, 50)),
    ),
    "KR": (
        (500_000, 1_500_000, 2_499_999, (9, 20)),
        (2_500_000, 3_500_000, 4_999_999, (7, 20)),
        (5_000_000, 7_000_000, 14_999_999, (9, 20)),
    ),
    "UK": (
        (50_000, 130_000, 199_999, (11, 20)),
        (200_000, 280_000, 399_999, (9, 20)),
        (400_000, 550_000, 1_199_999, (11, 20)),
    ),
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _binding_id(code: str, role: PopulationEstimandRole) -> str:
    role_offset = 0 if role is PopulationEstimandRole.CALIBRATION else 1
    ordinal = tuple(JURISDICTIONS).index(code) * 2 + role_offset
    return f"{ordinal:02d}.{code.lower()}.{role.value.lower()}"


def _target_id(code: str, role: PopulationEstimandRole) -> str:
    return f"standardized.person.2024.{code.lower()}.{role.value.lower()}"


def _income_shares(
    *, age_ordinal: int, household_ordinal: int, jurisdiction_shift: int
) -> tuple[int, int, int]:
    low = 42 - 4 * age_ordinal - 5 * household_ordinal - jurisdiction_shift
    high = 18 + 3 * age_ordinal + 5 * household_ordinal + jurisdiction_shift
    middle = 100 - low - high
    if min(low, middle, high) <= 0:
        raise AssertionError("income assumption produced a non-positive share")
    return low, middle, high


def _payer_share(
    *,
    age_ordinal: int,
    income_ordinal: int,
    gamer: bool,
    validation: bool,
) -> int:
    if gamer:
        share = 24 + 7 * income_ordinal + max(0, 3 - age_ordinal) * 2
        if validation:
            share = max(1, share - 3)
    else:
        share = 3 + 2 * income_ordinal + max(0, 2 - age_ordinal)
        if validation:
            share = min(99, share + 1)
    return share


def _joint_probabilities(
    code: str,
    role: PopulationEstimandRole,
) -> tuple[Fraction, ...]:
    assumptions = JURISDICTIONS[code]
    validation = role is PopulationEstimandRole.VALIDATION
    probabilities: list[Fraction] = []
    for age_ordinal, age_share in enumerate(assumptions["age"]):
        for income_ordinal in range(len(INCOME_BANDS)):
            for household_ordinal in SOURCE_HOUSEHOLD_ORDINALS:
                household_share = HOUSEHOLD_SHARES[age_ordinal][
                    household_ordinal
                ]
                income_share = _income_shares(
                    age_ordinal=age_ordinal,
                    household_ordinal=household_ordinal,
                    jurisdiction_shift=int(assumptions["income_shift"]),
                )[income_ordinal]
                gamer_share = int(assumptions["gaming"][age_ordinal])
                if validation:
                    gamer_share = max(1, gamer_share - 4)
                for gamer in (True, False):
                    selected_gaming_share = gamer_share if gamer else 100 - gamer_share
                    payer_share = _payer_share(
                        age_ordinal=age_ordinal,
                        income_ordinal=income_ordinal,
                        gamer=gamer,
                        validation=validation,
                    )
                    for ever_payer in (True, False):
                        selected_payer_share = (
                            payer_share if ever_payer else 100 - payer_share
                        )
                        probability = (
                            Fraction(age_share, 100)
                            * Fraction(household_share, 100)
                            * Fraction(income_share, 100)
                            * Fraction(selected_gaming_share, 100)
                            * Fraction(selected_payer_share, 100)
                        )
                        probabilities.append(probability)
    if sum(probabilities, Fraction()) != 1:
        raise AssertionError("joint conditional construction must sum exactly to one")
    return tuple(probabilities)


def _hamilton_integer_masses(
    probabilities: tuple[Fraction, ...],
) -> tuple[int, ...]:
    quotas = tuple(value * TARGET_COUNT_PER_JURISDICTION for value in probabilities)
    counts = [quota.numerator // quota.denominator for quota in quotas]
    remaining = TARGET_COUNT_PER_JURISDICTION - sum(counts)
    order = sorted(
        range(len(quotas)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    if sum(counts) != TARGET_COUNT_PER_JURISDICTION:
        raise AssertionError("integerized joint masses do not preserve the target")
    return tuple(counts)


def _write_rate_source_bundle_hash(source_registry_sha256: str) -> None:
    text = SOURCE_BUNDLE_PATH.read_text(encoding="utf-8")
    updated, substitutions = re.subn(
        r'(?m)^source_registry_sha256 = "[0-9a-f]{64}"$',
        f'source_registry_sha256 = "{source_registry_sha256}"',
        text,
    )
    if substitutions != 1:
        raise RuntimeError("source_bundle.toml has an unexpected digest declaration")
    SOURCE_BUNDLE_PATH.write_text(updated, encoding="utf-8", newline="\n")


def _write_population_evidence(
    source_registry_sha256: str,
) -> tuple[object, tuple[object, ...]]:
    rows = [list(POPULATION_CELL_CSV_COLUMNS)]
    for code in JURISDICTIONS:
        for role in PopulationEstimandRole:
            counts = _hamilton_integer_masses(_joint_probabilities(code, role))
            cell_ordinal = 0
            for age_band_id, age_min, age_max in AGE_BANDS:
                del age_band_id
                for income_band in INCOME_BANDS:
                    for household_ordinal in SOURCE_HOUSEHOLD_ORDINALS:
                        household_type = HOUSEHOLD_TYPES[household_ordinal][0]
                        for gaming_state in ("GAMER", "NON_GAMER"):
                            for payer_state in ("EVER_PAYER", "NEVER_PAYER"):
                                mass = Fraction(
                                    counts[cell_ordinal],
                                    TARGET_COUNT_PER_JURISDICTION,
                                )
                                rows.append(
                                    [
                                        _target_id(code, role),
                                        code,
                                        role.value,
                                        f"cell.{cell_ordinal:03d}",
                                        str(age_min),
                                        str(age_max),
                                        income_band,
                                        household_type,
                                        gaming_state,
                                        payer_state,
                                        str(mass.numerator),
                                        str(mass.denominator),
                                    ]
                                )
                                cell_ordinal += 1
            if cell_ordinal != len(counts):
                raise AssertionError("joint cell enumeration changed")
    content = ("\n".join(",".join(row) for row in rows) + "\n").encode("utf-8")
    POPULATION_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    POPULATION_CSV_PATH.write_bytes(content)

    bindings: list[str] = []
    for code, jurisdiction in JURISDICTIONS.items():
        for role in PopulationEstimandRole:
            age_shares = ";".join(
                country_code
                + "="
                + "/".join(str(value) for value in values["age"])
                for country_code, values in JURISDICTIONS.items()
            )
            gaming_shares = ";".join(
                country_code
                + "="
                + "/".join(str(value) for value in values["gaming"])
                for country_code, values in JURISDICTIONS.items()
            )
            income_shifts = ";".join(
                f"{country_code}={values['income_shift']}"
                for country_code, values in JURISDICTIONS.items()
            )
            household_matrix = ";".join(
                "/".join(str(value) for value in row)
                for row in HOUSEHOLD_SHARES
            )
            recipe = exact_csv_joint_population_recipe_json(
                target_population_id=_target_id(code, role),
                jurisdiction_code=code,
                estimand_role=role,
            )
            bindings.append(
                f'''[[bindings]]
binding_id = "{_binding_id(code, role)}"
artifact_id = "{ARTIFACT_ID}"
target_population_id = "{_target_id(code, role)}"
jurisdiction_code = "{code}"
geography = "{jurisdiction['name']}"
reference_period_start = 2024-01-01
reference_period_end = 2024-12-31
population_base = "10,000 standardized design-person units per jurisdiction; not national headcounts; canonical age-band percentage shares 10-17/18-24/25-34/35-44/45-54/55-69={age_shares}; exact cell counts use floor quotas then largest rational remainders with source ordinal as final tie-break"
universe = "usual-resident gamers and non-gamers aged 10 to 69"
unit_of_analysis = "one modeled usual-resident person/player"
eligibility = "usual resident of UK, KR, JP, or BE and aged 10 to 69 before treatment"
exclusion = "outside the four jurisdictions, outside the age frame, or missing any declared pre-treatment joint field"
age_min_inclusive = 10
age_max_exclusive = 70
household_income_definition = "modeled annual equivalised household disposable-income band; conditional percentages by canonical age ordinal a, canonical household ordinal h, and jurisdiction shift s with {income_shifts}: low=42-4a-5h-s; high=18+3a+5h+s; middle=100-low-high"
household_income_currency = "{jurisdiction['currency']}"
household_income_period = "annual 2024 nominal local currency"
household_income_equivalisation = "conceptual modified-OECD basis; no source microdata transformation has been validated"
household_definition = "canonical type order one-person/multi-no-minor/with-minor; conditional percentage matrix by canonical age ordinal={household_matrix}; with-minor means every realized modeled household contains at least one pre-treatment minor"
gaming_definition = "pre-treatment current-gaming percentage by canonical age ordinal={gaming_shares}; calibration uses the base percentages and validation subtracts 4 percentage points with floor 1; depends on jurisdiction, age, and estimand role only and is not caused by income"
payer_definition = "pre-treatment EVER_PAYER percentage: if gamer, 24+7i+2*max(0,3-a), else 3+2i+max(0,2-a), where a and i are canonical age/income ordinals; validation subtracts 3 with floor 1 for gamers and adds 1 with cap 99 for non-gamers"
zero_spender_treatment = "never-payers and non-gamers are retained explicitly; no zero-spender deletion"
estimand_role = "{role.value}"
status = "ILLUSTRATIVE"
source_ids = ["{jurisdiction['source_id']}"]
retrieved_on = 2026-08-30
recipe_json = {json.dumps(recipe)}

'''
            )

    POPULATION_BUNDLE_PATH.write_text(
        f'''schema_version = 1
bundle_id = "{EVIDENCE_BUNDLE_ID}"
provenance_status = "ILLUSTRATIVE"
source_registry_sha256 = "{source_registry_sha256}"
artifact_root = "population_artifacts"
notes = "Complete illustrative joint recipe: P(age)*P(household|age)*P(income|age,household,jurisdiction)*P(gaming|age,jurisdiction,role)*P(payer|age,income,gaming,role). Every factor and role adjustment is stated in each binding; no additional independence is implied. Exact 10,000-unit jurisdiction counts use Hamilton floor-plus-largest-remainder integerization. Evidence household source order is canonical ordinals 2/0/1 and design construction maps semantic keys to canonical 0/1/2. Official sources are contextual anchors only; assumptions are unsigned and campaign readiness is false."

[[artifacts]]
artifact_id = "{ARTIFACT_ID}"
relative_path = "{POPULATION_CSV_PATH.name}"
media_type = "text/csv"
sha256 = "{sha256(content).hexdigest()}"
byte_length = {len(content)}

{''.join(bindings)}[signature]
status = "MISSING"
algorithm = "NONE"
key_id = ""
value = ""
''',
        encoding="utf-8",
        newline="\n",
    )
    return load_and_verify_population_evidence_bundle(
        POPULATION_BUNDLE_PATH,
        expected_source_registry_sha256=source_registry_sha256,
    )


def _cluster_for_role(role: PopulationEstimandRole, semantic_key: str) -> str:
    candidate = 0
    while True:
        digest = sha256(
            f"cluster:{semantic_key}:{candidate}".encode("utf-8")
        ).hexdigest()
        if assigned_population_partition_role(
            identity_namespace=IDENTITY_NAMESPACE,
            assignment_seed_sha256=PARTITION_SEED_SHA256,
            cluster_identity_sha256=digest,
            calibration_threshold_numerator=1,
            calibration_threshold_denominator=2,
        ) is role:
            return digest
        candidate += 1


def _write_population_design(
    evidence_bundle: object,
    evidence_results: tuple[object, ...],
) -> object:
    results = {
        (result.jurisdiction_code, result.estimand_role): result
        for result in evidence_results
    }
    age_tables = "".join(
        f'''[[domains.age_bands]]
ordinal = {ordinal}
age_band_id = "{age_band_id}"
age_min_inclusive = {age_min}
age_max_exclusive = {age_max}

'''
        for ordinal, (age_band_id, age_min, age_max) in enumerate(AGE_BANDS)
    )
    income_tables: list[str] = []
    jurisdiction_tables: list[str] = []
    for code, jurisdiction in JURISDICTIONS.items():
        low_upper, middle_upper = INCOME_BOUNDS[code]
        bounds = (
            (True, 0, False, low_upper),
            (False, low_upper, False, middle_upper),
            (False, middle_upper, True, 0),
        )
        for ordinal, (income_id, definition, bound) in enumerate(
            zip(INCOME_BANDS, INCOME_DEFINITIONS, bounds, strict=True)
        ):
            lower_unbounded, lower, upper_unbounded, upper = bound
            income_tables.append(
                f'''[[domains.income_bands]]
ordinal = {ordinal}
jurisdiction_code = "{code}"
income_band_id = "{income_id}"
definition = "{definition}"
currency = "{jurisdiction['currency']}"
period = "annual 2024 nominal local currency"
lower_unbounded = {str(lower_unbounded).lower()}
lower_bound_numerator = {lower}
lower_bound_denominator = 1
upper_unbounded = {str(upper_unbounded).lower()}
upper_bound_numerator = {upper}
upper_bound_denominator = 1

'''
            )
        calibration = results[(code, PopulationEstimandRole.CALIBRATION)]
        validation = results[(code, PopulationEstimandRole.VALIDATION)]
        jurisdiction_tables.append(
            f'''[[jurisdictions]]
jurisdiction_code = "{code}"
target_population_count = {TARGET_COUNT_PER_JURISDICTION}
calibration_binding_id = "{calibration.binding_id}"
calibration_target_population_id = "{calibration.target_population_id}"
calibration_evidence_sha256 = "{calibration.evidence_sha256}"
validation_binding_id = "{validation.binding_id}"
validation_target_population_id = "{validation.target_population_id}"
validation_evidence_sha256 = "{validation.evidence_sha256}"

'''
        )

    household_tables = "".join(
        f'''[[domains.household_types]]
ordinal = {ordinal}
household_type_id = "{household_id}"
definition = "{definition}"

'''
        for ordinal, (household_id, definition) in enumerate(HOUSEHOLD_TYPES)
    )
    records: list[tuple[str, str]] = []
    for result in evidence_results:
        for cell in result.cells:
            if cell.target_mass == 0:
                continue
            semantic_key = f"{result.binding_id}:{cell.cell_id}"
            record_id = sha256(
                f"{IDENTITY_NAMESPACE}:record:{semantic_key}".encode("utf-8")
            ).hexdigest()
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
    result_digests = ", ".join(
        f'"{result.evidence_sha256}"' for result in evidence_results
    )
    design_text = f'''schema_version = 1
design_id = "{DESIGN_ID}"
provenance_status = "ILLUSTRATIVE"
notes = "One person/player aged 10-69 in an equal-country standardized UK/KR/JP/BE design. Joint masses follow declared conditional assumptions and exact Hamilton integerization, not independent marginal sampling. Analysis weights target a mean/proportion over 40,000 design-person equivalents; expansion weights target additive totals in those design units only."
population_evidence_bundle_sha256 = "{evidence_bundle.bundle_sha256}"
population_evidence_result_sha256s = [{result_digests}]
hamilton_recipe = "{EXACT_RATIONAL_HAMILTON_V1}"

[domains]
income_missing_policy = "REJECT"
household_missing_policy = "REJECT"
gaming_states = ["GAMER", "NON_GAMER"]
payer_history_states = ["EVER_PAYER", "NEVER_PAYER"]

{age_tables}{''.join(income_tables)}{household_tables}{''.join(jurisdiction_tables)}[partition]
identity_namespace = "{IDENTITY_NAMESPACE}"
record_id_recipe = "{CANONICAL_SOURCE_RECORD_ID_V1}"
cluster_id_recipe = "{CANONICAL_SOURCE_CLUSTER_ID_V1}"
role_assignment_recipe = "{SHA256_CLUSTER_THRESHOLD_V1}"
assignment_seed_sha256 = "{PARTITION_SEED_SHA256}"
calibration_threshold_numerator = 1
calibration_threshold_denominator = 2

{''.join(text for _record_id, text in sorted(records))}'''.rstrip() + "\n"
    POPULATION_DESIGN_PATH.write_text(
        design_text,
        encoding="utf-8",
        newline="\n",
    )
    return load_and_verify_population_design_bundle(
        POPULATION_DESIGN_PATH,
        population_evidence_bundle=evidence_bundle,
        population_evidence_results=evidence_results,
    )


def _write_runtime_mapping(verification: object) -> object:
    household_sizes = {
        "household.one-person": 1,
        "household.multi-no-minor": 2,
        "household.with-minor": 3,
    }
    entries: list[dict[str, object]] = []
    for income_band in verification.bundle.income_bands:
        code = income_band.jurisdiction_code
        currency = JURISDICTIONS[code]["currency"]
        model_values = RUNTIME_INCOME_MODELS[code][income_band.ordinal]
        minimum, median, maximum_inclusive, log_sigma = model_values
        for household in verification.bundle.household_types:
            transformation = (
                "NO POINTWISE SOURCE-INCOME TRANSFORM: the runtime within-band "
                "log-normal is a standalone ILLUSTRATIVE assumption specified "
                "directly in monthly personal minor units; the source annual "
                "equivalised-household band is only a joint-cell stratum label, "
                "and no division-by-12, equivalisation, or household allocation "
                "identity is asserted; household category affects joint mass and "
                "modeled capacity only; censor, round half-to-even, and apply no "
                "gaming/minor income adjustment"
            )
            income_model: dict[str, object] = {
                "target_quantity": RUNTIME_INCOME_CONCEPT,
                "model_family": "LOG_NORMAL",
                "median_cents": median,
                "log_sigma": list(log_sigma),
                "lower_bound_cents": minimum,
                "upper_bound_cents_inclusive": maximum_inclusive,
                "currency": currency,
                "time_period": "monthly 2024 nominal local-currency minor units",
                "source_id": JURISDICTIONS[code]["source_id"],
                "calibration_target": (
                    "illustrative band-specific median and log-sigma; no verified "
                    "microdata calibration or dispersion estimate"
                ),
                "transformation": transformation,
                "boundary_rule": "CENSOR_TO_INCLUSIVE_BOUNDS",
                "rounding_rule": "ROUND_HALF_TO_EVEN_CENTS",
                "minor_gaming_adjustment": "NONE",
                "minor_gaming_adjustment_reason": (
                    "INSUFFICIENT_VERIFIED_EVIDENCE"
                ),
            }
            recipe_payload = {
                "schema_version": 2,
                "source_income_band": income_band.snapshot(),
                "source_household_type": household.snapshot(),
                "modeled_players_per_household": household_sizes[
                    household.household_type_id
                ],
                "runtime_income_model": income_model,
            }
            recipe_id = (
                f"population.income.{code.lower()}.{income_band.income_band_id}."
                f"{household.household_type_id}.v2"
            )
            entries.append(
                {
                    "jurisdiction_code": code,
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
                        f"runtime.{code.lower()}.{income_band.income_band_id}"
                    ),
                    "runtime_personal_monthly_disposable_income_currency": currency,
                    "runtime_personal_monthly_disposable_income_min_cents": minimum,
                    "runtime_personal_monthly_disposable_income_max_cents_exclusive": (
                        maximum_inclusive + 1
                    ),
                    "modeled_players_per_household": household_sizes[
                        household.household_type_id
                    ],
                    "conversion_recipe_id": recipe_id,
                    "conversion_recipe_sha256": sha256(
                        _canonical_json(recipe_payload).encode("utf-8")
                    ).hexdigest(),
                    "runtime_personal_monthly_disposable_income_model": income_model,
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
        "schema_version": 2,
        "mapping_id": "standardized-four-country-runtime-income-v2",
        "design_id": verification.bundle.design_id,
        "design_bundle_sha256": verification.bundle.bundle_sha256,
        "domain_sha256": verification.bundle.domain_sha256,
        "source_income_concept": SOURCE_INCOME_CONCEPT,
        "runtime_income_concept": RUNTIME_INCOME_CONCEPT,
        "entries": entries,
    }
    POPULATION_RUNTIME_MAPPING_PATH.write_text(
        _canonical_json(payload) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return load_population_runtime_mapping_bundle(
        POPULATION_RUNTIME_MAPPING_PATH
    )


def main() -> None:
    source_registry_sha256 = sha256(SOURCES_PATH.read_bytes()).hexdigest()
    _write_rate_source_bundle_hash(source_registry_sha256)
    evidence_bundle, evidence_results = _write_population_evidence(
        source_registry_sha256
    )
    verification = _write_population_design(evidence_bundle, evidence_results)
    mapping = _write_runtime_mapping(verification)
    print(
        _canonical_json(
            {
                "source_registry_sha256": source_registry_sha256,
                "population_evidence_bundle_sha256": evidence_bundle.bundle_sha256,
                "population_evidence_result_count": len(evidence_results),
                "population_design_sha256": verification.bundle.bundle_sha256,
                "population_domain_sha256": verification.bundle.domain_sha256,
                "runtime_mapping_sha256": mapping.mapping_sha256,
                "runtime_mapping_schema_version": mapping.schema_version,
                "joint_cell_rows": sum(len(item.cells) for item in evidence_results),
                "campaign_ready": False,
            }
        )
    )


if __name__ == "__main__":
    main()
