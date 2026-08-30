from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from microtx_sim.data.population_design import (
    apportion_population_hamilton,
    build_population_calibration_target,
    load_and_verify_population_design_bundle,
)
from microtx_sim.data.population_evidence import verify_population_evidence_bundle
from microtx_sim.data.population_execution import (
    CAMPAIGN_POPULATION_ADAPTER_ID,
    PopulationExecutionValidationError,
    build_population_seed_execution_record,
    validate_population_campaign_preflight,
)
from microtx_sim.data.population_projection import (
    build_population_projection_adapter,
    initialize_population_projection,
    load_population_runtime_mapping_bundle,
    verify_population_projection_adapter,
)
from microtx_sim.data.profiles import load_profile_bundle
from microtx_sim.metrics.population_balance import (
    PopulationBalanceValidationError,
    build_population_balance_artifact,
)
from microtx_sim.rng import CounterRNG
from microtx_sim.types import ProvenanceStatus


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "data" / "provenance" / "population_design.toml"
MAPPING = ROOT / "data" / "provenance" / "population_runtime_mapping.json"


class RegisteredPopulationDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_profile_bundle(campaign=False)
        evidence = cls.profiles.population_evidence_bundle
        assert evidence is not None
        results = verify_population_evidence_bundle(
            evidence,
            expected_source_registry_sha256=cls.profiles.source_registry_sha256,
        )
        cls.verification = load_and_verify_population_design_bundle(
            DESIGN,
            population_evidence_bundle=evidence,
            population_evidence_results=results,
        )
        cls.target = build_population_calibration_target(cls.verification)
        cls.plan = apportion_population_hamilton(cls.target, 50_000)
        cls.mapping = load_population_runtime_mapping_bundle(MAPPING)
        cls.adapter = verify_population_projection_adapter(
            build_population_projection_adapter(
                cls.verification,
                cls.plan,
                cls.mapping,
                adapter_id=CAMPAIGN_POPULATION_ADAPTER_ID,
            )
        )
        cls.execution = initialize_population_projection(
            cls.adapter,
            cls.profiles.country_profiles,
            CounterRNG(41),
        )

    def test_registered_joint_design_is_complete_exact_and_honestly_illustrative(
        self,
    ) -> None:
        evidence = self.verification.evidence_bundle
        design = self.verification.bundle
        self.assertIs(evidence.provenance_status, ProvenanceStatus.ILLUSTRATIVE)
        self.assertIs(design.provenance_status, ProvenanceStatus.ILLUSTRATIVE)
        self.assertTrue(design.declaration_complete)
        self.assertFalse(evidence.campaign_ready)
        self.assertFalse(design.campaign_ready)
        self.assertEqual(
            tuple(item.jurisdiction_code for item in design.jurisdictions),
            ("BE", "JP", "KR", "UK"),
        )
        self.assertEqual(len(self.target.cells), 864)
        self.assertEqual(self.target.total_population_count, 40_000)
        self.assertEqual(
            sum((cell.target_mass for cell in self.target.cells), Fraction()),
            Fraction(1, 1),
        )
        self.assertEqual(
            sum((cell.target_population for cell in self.target.cells), Fraction()),
            Fraction(40_000, 1),
        )

    def test_mapping_v2_declares_bounded_lognormal_without_minor_adjustment(
        self,
    ) -> None:
        self.assertEqual(self.mapping.schema_version, 2)
        self.assertEqual(self.adapter.schema_version, 2)
        self.assertEqual(len(self.mapping.entries), 36)
        for entry in self.mapping.entries:
            model = entry.income_model
            assert model is not None
            self.assertEqual(model.model_family, "LOG_NORMAL")
            self.assertEqual(model.minor_gaming_adjustment, "NONE")
            self.assertEqual(
                model.minor_gaming_adjustment_reason,
                "INSUFFICIENT_VERIFIED_EVIDENCE",
            )
            self.assertIn(model.source_id, self.profiles.sources)
            self.assertIs(
                self.profiles.sources[model.source_id].status,
                ProvenanceStatus.ILLUSTRATIVE,
            )

    def test_apportionment_retains_exact_analysis_and_expansion_weights(self) -> None:
        analysis_total = sum(
            (
                item.analysis_weight * item.sample_count
                for item in self.plan.cells
            ),
            Fraction(),
        )
        expansion_total = sum(
            (
                item.expansion_weight * item.sample_count
                for item in self.plan.cells
            ),
            Fraction(),
        )
        self.assertEqual(analysis_total, 1)
        self.assertEqual(expansion_total, 40_000)
        positive_analysis_weights = {
            item.analysis_weight for item in self.plan.cells if item.sample_count
        }
        self.assertGreater(len(positive_analysis_weights), 1)

    def test_projected_execution_preserves_source_and_canonical_order_identities(
        self,
    ) -> None:
        execution = self.execution
        self.assertEqual(
            execution.players.jurisdiction_codes,
            tuple(profile.code for profile in self.profiles.country_profiles),
        )
        self.assertEqual(
            tuple(
                item.jurisdiction_code
                for item in self.adapter.verification.bundle.jurisdictions
            ),
            ("BE", "JP", "KR", "UK"),
        )
        record = build_population_seed_execution_record(
            execution,
            seed=41,
            cohort_digest="a" * 64,
            policy_days=14,
        )
        self.assertTrue(record.balance.exact_balance_passed)
        self.assertEqual(record.exact_weights.weight_sum, 1)
        self.assertEqual(len(record.exact_weights.player_ids), 50_000)
        self.assertEqual(record.assignment_sha256, execution.assignment_sha256)

        results_by_binding = {
            result.binding_id: result
            for result in self.verification.evidence_results
        }
        age_by_id = {
            item.age_band_id: item
            for item in self.verification.bundle.age_bands
        }
        source_order: list[tuple[str, str]] = []
        evidence_by_identity = {}
        for jurisdiction in self.verification.bundle.jurisdictions:
            result = results_by_binding[jurisdiction.calibration_binding_id]
            for cell in result.cells:
                identity = (jurisdiction.jurisdiction_code, cell.cell_id)
                source_order.append(identity)
                evidence_by_identity[identity] = cell
        canonical_order = [
            (cell.jurisdiction_code, cell.evidence_cell_id)
            for cell in self.target.cells
        ]
        self.assertNotEqual(source_order, canonical_order)
        self.assertEqual(
            tuple(item.evidence_cell_id for item in self.adapter.cells),
            tuple(cell.evidence_cell_id for cell in self.target.cells),
        )
        for cell in self.target.cells:
            evidence_cell = evidence_by_identity[
                (cell.jurisdiction_code, cell.evidence_cell_id)
            ]
            age = age_by_id[cell.age_band_id]
            self.assertEqual(
                evidence_cell.semantic_key,
                (
                    age.age_min_inclusive,
                    age.age_max_exclusive,
                    cell.income_band_id,
                    cell.household_type_id,
                    cell.gaming_state.value,
                    cell.payer_history_state.value,
                ),
            )

    def test_projected_households_realize_and_balance_declared_minor_semantics(
        self,
    ) -> None:
        execution = self.execution
        players = execution.players
        assignment = players.projected_population
        assert assignment is not None
        household_type_by_cell = tuple(
            cell.household_type for cell in assignment.metadata.cells
        )
        player_household_type = np.asarray(
            [household_type_by_cell[int(index)] for index in assignment.cell_index],
            dtype=object,
        )
        with_minor_households = 0
        for household_id in np.unique(players.household_id):
            positions = np.flatnonzero(players.household_id == household_id)
            household_types = set(player_household_type[positions])
            self.assertEqual(len(household_types), 1)
            household_type = household_types.pop()
            if household_type == "household.with-minor":
                with_minor_households += 1
                self.assertTrue(np.any(players.is_minor[positions]))
            elif household_type in {
                "household.one-person",
                "household.multi-no-minor",
            }:
                self.assertFalse(np.any(players.is_minor[positions]))
        self.assertGreater(with_minor_households, 0)

        group_by_cell = tuple(
            (
                cell.jurisdiction_code,
                cell.monthly_disposable_income_band_id,
                cell.household_type,
            )
            for cell in assignment.metadata.cells
        )
        households_by_group: dict[
            tuple[str, str, str], list[np.ndarray]
        ] = {}
        for household_id in np.unique(players.household_id):
            positions = np.flatnonzero(players.household_id == household_id)
            group = group_by_cell[int(assignment.cell_index[positions[0]])]
            households_by_group.setdefault(group, []).append(positions)
        swap: tuple[int, int] | None = None
        for group, households in households_by_group.items():
            if group[2] != "household.with-minor":
                continue
            single_minor = next(
                (
                    positions
                    for positions in households
                    if np.count_nonzero(players.is_minor[positions]) == 1
                ),
                None,
            )
            adult_household = next(
                (
                    positions
                    for positions in households
                    if single_minor is not None
                    and not np.array_equal(positions, single_minor)
                    and np.any(~players.is_minor[positions])
                ),
                None,
            )
            if single_minor is not None and adult_household is not None:
                minor_position = int(
                    single_minor[np.flatnonzero(players.is_minor[single_minor])[0]]
                )
                adult_position = int(
                    adult_household[
                        np.flatnonzero(~players.is_minor[adult_household])[0]
                    ]
                )
                swap = (minor_position, adult_position)
                break
        self.assertIsNotNone(swap)
        assert swap is not None
        changed_households = players.household_id.copy()
        changed_households[swap[0]], changed_households[swap[1]] = (
            changed_households[swap[1]],
            changed_households[swap[0]],
        )
        tampered_players = replace(
            players,
            household_id=changed_households,
        )
        tampered_execution = replace(execution, players=tampered_players)
        with self.assertRaisesRegex(
            PopulationBalanceValidationError,
            "household.with-minor.*without a minor",
        ):
            build_population_balance_artifact(tampered_execution)

    def test_campaign_gate_rejects_model_based_unsigned_evidence(self) -> None:
        with self.assertRaisesRegex(
            PopulationExecutionValidationError,
            "population_adapter_authenticity=not_verified",
        ):
            validate_population_campaign_preflight(self.adapter)

    def test_campaign_gate_binds_runtime_income_sources_to_population_evidence(
        self,
    ) -> None:
        payload = json.loads(MAPPING.read_text("utf-8"))
        payload["entries"][0][
            "runtime_personal_monthly_disposable_income_model"
        ]["source_id"] = "PLAYTIKA_10K_2024"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unbound-source-mapping.json"
            path.write_text(
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
                newline="",
            )
            mapping = load_population_runtime_mapping_bundle(path)
            adapter = build_population_projection_adapter(
                self.verification,
                self.plan,
                mapping,
                adapter_id=CAMPAIGN_POPULATION_ADAPTER_ID,
            )
            with self.assertRaisesRegex(
                PopulationExecutionValidationError,
                "population_runtime_mapping_sources_unbound=BE:PLAYTIKA_10K_2024",
            ):
                validate_population_campaign_preflight(adapter)

    def test_campaign_gate_rejects_an_unregistered_adapter_identity(self) -> None:
        unregistered = build_population_projection_adapter(
            self.verification,
            self.plan,
            self.mapping,
            adapter_id="campaign.standardized.population.unregistered",
        )
        with self.assertRaisesRegex(
            PopulationExecutionValidationError,
            "population_adapter_id=campaign.standardized.population.unregistered",
        ):
            validate_population_campaign_preflight(unregistered)


if __name__ == "__main__":
    unittest.main()
