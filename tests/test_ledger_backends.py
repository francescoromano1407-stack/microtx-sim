from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from microtx_sim.core.ledger import (
    INT64_MAX,
    Ledger,
    LedgerEntry,
    LedgerVerificationError,
)
from microtx_sim.types import LedgerBackend


GOLDEN_LOGICAL_SHA256 = (
    "0e136bab284fe2958741e4684f3953c04faacc0c97d027457a72a21d5d37e609"
)


@contextmanager
def _temporary_directory(test_case: unittest.TestCase) -> Iterator[str]:
    """Keep the root alive until ledger cleanups have closed Windows handles."""

    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    yield directory.name


def _entry(
    number: int,
    *,
    tick: int | None = None,
    amount_cents: int | None = None,
    kind: str = "sale",
    reference: str | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        tick=number if tick is None else tick,
        debit_account=f"payer:{number}",
        credit_account=f"firm:{number % 3}",
        amount_cents=number + 1 if amount_cents is None else amount_cents,
        kind=kind,
        reference=f"entry:{number}" if reference is None else reference,
    )


def _canonical_entries() -> tuple[LedgerEntry, ...]:
    return (
        LedgerEntry(
            tick=7,
            debit_account="acct:\x00α\n",
            credit_account="firm|one",
            amount_cents=13,
            kind="sale\nkind",
            reference="Ref",
        ),
        LedgerEntry(
            tick=0,
            debit_account="firm|one",
            credit_account="acct:\x00α\n",
            amount_cents=5,
            kind="refund",
            reference="ref",
        ),
        LedgerEntry(
            tick=INT64_MAX,
            debit_account="external:income",
            credit_account="acct:β",
            amount_cents=INT64_MAX,
            kind="grant",
            reference="x:y\nz",
        ),
    )


class LedgerBackendContractTests(unittest.TestCase):
    def _backend_ledgers(self, root: Path) -> list[tuple[str, Ledger]]:
        ledgers = [
            ("memory", Ledger()),
            ("persistent", Ledger.create(root / "persistent.sqlite3")),
            ("temporary", Ledger.temporary(directory=root)),
        ]
        for _, ledger in ledgers:
            self.addCleanup(ledger.close)
        return ledgers

    @staticmethod
    def _state(ledger: Ledger) -> tuple[object, ...]:
        return (
            ledger.entries,
            ledger.entry_count(),
            ledger.account_net_cents(),
            ledger.total_flow_cents(),
            ledger.logical_sha256(),
            ledger.balance_snapshot(),
        )

    def test_all_backends_are_logically_equivalent_and_explicitly_ordered(
        self,
    ) -> None:
        entries = _canonical_entries()
        with _temporary_directory(self) as directory:
            ledgers = self._backend_ledgers(Path(directory))
            snapshots = []
            for name, ledger in ledgers:
                with self.subTest(backend=name):
                    self.assertEqual(ledger.append_many(entries[:2]), entries[:2])
                    self.assertIsNone(ledger.extend(iter(entries[2:])))
                    self.assertEqual(ledger.entries, entries)
                    self.assertEqual(tuple(ledger.iter_entries()), entries)
                    self.assertEqual(
                        tuple(ledger.iter_entries("sale\nkind")),
                        entries[:1],
                    )
                    self.assertEqual(ledger.entry_count(), 3)
                    self.assertEqual(ledger.entry_count("refund"), 1)
                    self.assertEqual(ledger.entry_count("missing"), 0)
                    self.assertEqual(
                        ledger.conflicting_references(("Ref", "ref", "new")),
                        frozenset(("Ref", "ref")),
                    )
                    ledger.validate_references(("NEW", "new"))
                    with self.assertRaisesRegex(ValueError, "Ref"):
                        ledger.validate_references(("unused", "Ref"))
                    ledger.assert_balanced()
                    ledger.assert_balanced(full=True)
                    self.assertEqual(ledger.logical_sha256(), GOLDEN_LOGICAL_SHA256)
                    snapshots.append(ledger.balance_snapshot())

            self.assertIs(ledgers[0][1].backend, LedgerBackend.MEMORY)
            self.assertIsNone(ledgers[0][1].path)
            self.assertFalse(ledgers[0][1].temporary_store)
            for _, ledger in ledgers[1:]:
                self.assertIs(ledger.backend, LedgerBackend.SQLITE)
                self.assertIsNotNone(ledger.path)
            self.assertFalse(ledgers[1][1].temporary_store)
            self.assertTrue(ledgers[2][1].temporary_store)
            self.assertEqual(snapshots[0], snapshots[1])
            self.assertEqual(snapshots[1], snapshots[2])

    def test_sqlite_schema_and_connection_settings_are_explicit(self) -> None:
        with _temporary_directory(self) as directory:
            ledger = Ledger.create(Path(directory) / "settings.sqlite3")
            self.addCleanup(ledger.close)
            connection = ledger._connection
            self.assertIsNone(connection.isolation_level)
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "delete")
            self.assertEqual(connection.execute("PRAGMA synchronous").fetchone()[0], 3)
            self.assertEqual(connection.execute("PRAGMA cache_size").fetchone()[0], -2048)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(
                connection.execute("PRAGMA application_id").fetchone()[0],
                0x4D54584C,
            )
            table_sql = connection.execute(
                "SELECT sql FROM sqlite_schema WHERE name = 'ledger_entry'"
            ).fetchone()[0]
            self.assertIn("STRICT", table_sql)
            self.assertIn("sequence INTEGER PRIMARY KEY", table_sql)
            self.assertIn("reference TEXT COLLATE BINARY", table_sql)

    def test_create_is_fresh_and_never_overwrites_existing_artifacts(self) -> None:
        with _temporary_directory(self) as directory:
            root = Path(directory)
            occupied = root / "occupied.sqlite3"
            occupied.write_bytes(b"keep me")
            with self.assertRaises(FileExistsError):
                Ledger.create(occupied)
            self.assertEqual(occupied.read_bytes(), b"keep me")

            reserved = root / "reserved.sqlite3"
            seal_path = reserved.with_name(reserved.name + ".seal.json")
            seal_path.write_text("reserved", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                Ledger.create(reserved)
            self.assertFalse(reserved.exists())
            self.assertEqual(seal_path.read_text("utf-8"), "reserved")

            sidecar_reserved = root / "sidecar-reserved.sqlite3"
            wal_path = Path(str(sidecar_reserved) + "-wal")
            wal_path.write_bytes(b"reserved sidecar")
            with self.assertRaises(FileExistsError):
                Ledger.create(sidecar_reserved)
            self.assertFalse(sidecar_reserved.exists())
            self.assertEqual(wal_path.read_bytes(), b"reserved sidecar")

    def test_strict_builtin_int64_and_text_domains_apply_at_every_boundary(
        self,
    ) -> None:
        class IntSubclass(int):
            pass

        class StrSubclass(str):
            pass

        valid = {
            "tick": 0,
            "debit_account": "source",
            "credit_account": "destination",
            "amount_cents": 1,
            "kind": "kind",
            "reference": "reference",
        }
        for field, value in (
            ("tick", True),
            ("tick", 1.0),
            ("tick", np.int64(1)),
            ("tick", IntSubclass(1)),
            ("amount_cents", False),
            ("amount_cents", 1.0),
            ("amount_cents", np.int64(1)),
            ("amount_cents", IntSubclass(1)),
        ):
            with self.subTest(field=field, value_type=type(value).__name__):
                candidate = dict(valid)
                candidate[field] = value
                with self.assertRaises(TypeError):
                    LedgerEntry(**candidate)

        for field, value, exception in (
            ("tick", -1, ValueError),
            ("tick", INT64_MAX + 1, OverflowError),
            ("amount_cents", 0, ValueError),
            ("amount_cents", -1, ValueError),
            ("amount_cents", INT64_MAX + 1, OverflowError),
        ):
            with self.subTest(field=field, value=value):
                candidate = dict(valid)
                candidate[field] = value
                with self.assertRaises(exception):
                    LedgerEntry(**candidate)

        for field in (
            "debit_account",
            "credit_account",
            "kind",
            "reference",
        ):
            for value, exception in (
                (b"bytes", TypeError),
                (StrSubclass("subclass"), TypeError),
                ("", ValueError),
                ("\ud800", ValueError),
            ):
                with self.subTest(field=field, value=repr(value)):
                    candidate = dict(valid)
                    candidate[field] = value
                    with self.assertRaises(exception):
                        LedgerEntry(**candidate)

        with self.assertRaisesRegex(ValueError, "must differ"):
            LedgerEntry(**(valid | {"credit_account": "source"}))

        boundary = LedgerEntry(
            **(
                valid
                | {
                    "tick": INT64_MAX,
                    "amount_cents": INT64_MAX,
                    "reference": "boundary",
                }
            )
        )
        ledger = Ledger()
        self.addCleanup(ledger.close)
        ledger.append_many((boundary,))
        self.assertEqual(ledger.entries, (boundary,))

        forged = _entry(10, reference="forged")
        object.__setattr__(forged, "amount_cents", True)
        before = self._state(ledger)
        with self.assertRaises(TypeError):
            ledger.append_many((forged,))
        self.assertEqual(self._state(ledger), before)

        with self.assertRaises(TypeError):
            ledger.transfer(
                tick=1,
                debit_account="a",
                credit_account="b",
                amount_cents=True,
                kind="kind",
                reference="bad-transfer",
            )
        self.assertEqual(self._state(ledger), before)

    def test_duplicate_and_generator_failures_are_whole_batch_atomic(self) -> None:
        class GeneratorFailure(Exception):
            pass

        with _temporary_directory(self) as directory:
            for name, ledger in self._backend_ledgers(Path(directory))[:2]:
                with self.subTest(backend=name):
                    seed = _entry(1, reference=f"{name}:seed")
                    ledger.append_many((seed,))
                    before = self._state(ledger)

                    duplicate_a = _entry(2, reference=f"{name}:inside")
                    duplicate_b = _entry(3, reference=f"{name}:inside")
                    with self.assertRaisesRegex(ValueError, "duplicate"):
                        ledger.append_many((duplicate_a, duplicate_b))
                    self.assertEqual(self._state(ledger), before)

                    new_entry = _entry(4, reference=f"{name}:new")
                    existing = _entry(5, reference=seed.reference)
                    with self.assertRaisesRegex(ValueError, seed.reference):
                        ledger.extend((new_entry, existing))
                    self.assertEqual(self._state(ledger), before)

                    def broken_entries():
                        yield _entry(6, reference=f"{name}:yielded")
                        raise GeneratorFailure

                    with self.assertRaises(GeneratorFailure):
                        ledger.append_many(broken_entries())
                    self.assertEqual(self._state(ledger), before)
                    with self.assertRaises(GeneratorFailure):
                        ledger.extend(broken_entries())
                    self.assertEqual(self._state(ledger), before)

                    ledger.append_many((new_entry,))
                    self.assertEqual(ledger.entries, (seed, new_entry))

    def test_sqlite_native_failure_rolls_back_every_prior_row_in_the_batch(
        self,
    ) -> None:
        with _temporary_directory(self) as directory:
            ledger = Ledger.create(Path(directory) / "failure.sqlite3")
            self.addCleanup(ledger.close)
            ledger.append_many((_entry(1),))
            before = self._state(ledger)
            ledger._connection.execute(
                """
                CREATE TRIGGER ledger_test_failure
                BEFORE INSERT ON ledger_entry
                WHEN NEW.reference = 'forced'
                BEGIN
                    SELECT RAISE(ABORT, 'forced insertion failure');
                END
                """
            )
            batch = (
                _entry(2, reference="before-forced"),
                _entry(3, reference="forced"),
                _entry(4, reference="after-forced"),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                ledger.append_many(batch)
            self.assertEqual(self._state(ledger), before)
            ledger._connection.execute("DROP TRIGGER ledger_test_failure")
            ledger.append_many(batch)
            self.assertEqual(ledger.entries, (_entry(1),) + batch)

    def test_public_and_caller_owned_outer_transactions_use_savepoints(self) -> None:
        ledger = Ledger()
        self.addCleanup(ledger.close)

        first = _entry(1)
        second = _entry(2)
        with ledger.transaction():
            with self.assertRaisesRegex(RuntimeError, "active transaction"):
                ledger.close()
            self.assertFalse(ledger.closed)
            ledger.append_many((first,))
            with self.assertRaisesRegex(ValueError, first.reference):
                ledger.append_many((_entry(3, reference=first.reference),))
            ledger.append_many((second,))
        self.assertEqual(ledger.entries, (first, second))

        before = self._state(ledger)
        with self.assertRaisesRegex(RuntimeError, "outer failure"):
            with ledger.transaction():
                ledger.append_many((_entry(4),))
                raise RuntimeError("outer failure")
        self.assertEqual(self._state(ledger), before)

        raw_entry = _entry(5)
        ledger._connection.execute("BEGIN IMMEDIATE")
        try:
            ledger.append_many((raw_entry,))
            self.assertTrue(ledger._connection.in_transaction)
        finally:
            ledger._connection.execute("ROLLBACK")
        self.assertEqual(self._state(ledger), before)

    def test_python_integer_aggregates_exceed_sqlite_int64_exactly(self) -> None:
        with _temporary_directory(self) as directory:
            for name, ledger in self._backend_ledgers(Path(directory))[:2]:
                with self.subTest(backend=name):
                    ledger.transfer(
                        tick=0,
                        debit_account="source",
                        credit_account="destination",
                        amount_cents=INT64_MAX,
                        kind="huge",
                        reference=f"{name}:huge:1",
                    )
                    ledger.transfer(
                        tick=1,
                        debit_account="source",
                        credit_account="destination",
                        amount_cents=INT64_MAX,
                        kind="huge",
                        reference=f"{name}:huge:2",
                    )
                    expected = 2 * INT64_MAX
                    self.assertEqual(ledger.total_flow_cents(), expected)
                    self.assertEqual(ledger.total_flow_cents(kind="huge"), expected)
                    self.assertEqual(
                        ledger.account_net_cents(),
                        {"source": -expected, "destination": expected},
                    )
                    ledger.assert_balanced(full=True)

    def test_full_validation_detects_a_noncontiguous_stored_sequence(self) -> None:
        ledger = Ledger()
        self.addCleanup(ledger.close)
        ledger.append_many((_entry(1), _entry(2), _entry(3)))
        ledger._connection.execute("DELETE FROM ledger_entry WHERE sequence = 2")
        ledger.assert_balanced()
        with self.assertRaisesRegex(LedgerVerificationError, "not contiguous"):
            ledger.assert_balanced(full=True)
        with self.assertRaisesRegex(LedgerVerificationError, "not contiguous"):
            ledger.logical_sha256()

    def test_close_lifecycle_temporary_cleanup_and_storage_identity(self) -> None:
        with _temporary_directory(self) as directory:
            root = Path(directory)
            temporary = Ledger.temporary(directory=root)
            temporary_path = temporary.path
            self.assertIsNotNone(temporary_path)
            assert temporary_path is not None
            temporary_parent = temporary_path.parent
            self.assertTrue(temporary_path.exists())
            self.assertTrue(temporary.shares_storage_with(temporary))
            self.assertTrue(temporary.temporary_store)
            temporary.close()
            temporary.close()
            self.assertTrue(temporary.closed)
            self.assertTrue(temporary.temporary_store)
            self.assertEqual(temporary.path, temporary_path)
            self.assertFalse(temporary_path.exists())
            self.assertFalse(temporary_parent.exists())

            persistent_path = root / "survives.sqlite3"
            persistent = Ledger.create(persistent_path)
            persistent.transfer(
                tick=0,
                debit_account="a",
                credit_account="b",
                amount_cents=1,
                kind="kind",
                reference="persisted",
            )
            persistent.close()
            self.assertTrue(persistent_path.is_file())
            self.assertGreater(persistent_path.stat().st_size, 0)

            memory = Ledger()
            other_memory = Ledger()
            self.assertFalse(memory.shares_storage_with(other_memory))
            self.assertFalse(memory.shares_storage_with(persistent))

            reused_path = root / "reused.sqlite3"
            first = Ledger.create(reused_path)
            first.close()
            reused_path.unlink()
            replacement = Ledger.create(reused_path)
            self.addCleanup(replacement.close)
            self.assertFalse(first.shares_storage_with(replacement))

            memory.close()
            memory.close()
            self.assertTrue(memory.closed)
            for operation in (
                lambda: memory.entries,
                lambda: tuple(memory.iter_entries()),
                lambda: memory.entry_count(),
                lambda: memory.account_net_cents(),
                lambda: memory.total_flow_cents(),
                lambda: memory.assert_balanced(),
                lambda: memory.logical_sha256(),
                lambda: memory.balance_snapshot(),
                lambda: memory.append_many(()),
            ):
                with self.subTest(operation=operation):
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        operation()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                with memory.transaction():
                    pass
            other_memory.close()

    def test_persistent_python_object_retention_is_bounded(self) -> None:
        with _temporary_directory(self) as directory:
            ledger = Ledger.create(Path(directory) / "bounded.sqlite3")
            self.addCleanup(ledger.close)
            gc.collect()
            baseline = sum(type(value) is LedgerEntry for value in gc.get_objects())
            with ledger.transaction():
                for batch in range(200):
                    ledger.append_many(
                        _entry(
                            batch * 25 + offset,
                            reference=f"bounded:{batch}:{offset}",
                        )
                        for offset in range(25)
                    )
            gc.collect()
            retained = sum(type(value) is LedgerEntry for value in gc.get_objects())
            self.assertLessEqual(retained - baseline, 2)
            self.assertEqual(ledger.entry_count(), 5_000)


class LedgerSealTests(unittest.TestCase):
    def _sealed_ledger(
        self,
        root: Path,
        *,
        name: str = "sealed.sqlite3",
    ) -> tuple[Path, object]:
        database_path = root / name
        ledger = Ledger.create(database_path)
        ledger.append_many(_canonical_entries())
        seal = ledger.seal({"label": "α", "nested": {"value": 3}})
        self.assertTrue(ledger.closed)
        return database_path, seal

    def test_seal_and_verify_are_exact_read_only_and_hold_a_read_lock(self) -> None:
        with _temporary_directory(self) as directory:
            database_path, seal = self._sealed_ledger(Path(directory))
            manifest_path = database_path.with_name(database_path.name + ".seal.json")
            self.assertEqual(seal.database_path, database_path.resolve())
            self.assertEqual(seal.manifest_path, manifest_path.resolve())
            self.assertEqual(seal.entry_count, 3)
            self.assertEqual(seal.logical_sha256, GOLDEN_LOGICAL_SHA256)
            self.assertEqual(seal.minimum_tick, 0)
            self.assertEqual(seal.maximum_tick, INT64_MAX)
            self.assertEqual(seal.metadata, {"label": "α", "nested": {"value": 3}})
            self.assertEqual(
                seal.file_sha256,
                hashlib.sha256(database_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(seal.journal_mode, "delete")
            self.assertEqual(seal.synchronous, 3)
            self.assertLess(seal.cache_size, 0)
            self.assertFalse(Path(str(database_path) + "-journal").exists())

            before_bytes = database_path.read_bytes()
            before_stat = database_path.stat()
            original_hash = Ledger._file_sha256
            observed_lock = []

            def hash_while_checking_lock(path: Path) -> str:
                contender = sqlite3.connect(path, isolation_level=None, timeout=0.0)
                try:
                    with self.assertRaises(sqlite3.OperationalError):
                        contender.execute("BEGIN EXCLUSIVE")
                    observed_lock.append(True)
                finally:
                    contender.close()
                return original_hash(path)

            with patch.object(
                Ledger,
                "_file_sha256",
                staticmethod(hash_while_checking_lock),
            ):
                verified = Ledger.verify(database_path)
            self.assertEqual(verified, seal)
            self.assertEqual(observed_lock, [True])
            self.assertEqual(database_path.read_bytes(), before_bytes)
            after_stat = database_path.stat()
            self.assertEqual(after_stat.st_size, before_stat.st_size)
            self.assertEqual(after_stat.st_mtime_ns, before_stat.st_mtime_ns)
            self.assertFalse(Path(str(database_path) + "-journal").exists())

    def test_seal_rejects_nonpersistent_and_active_transaction_storage(self) -> None:
        memory = Ledger()
        self.addCleanup(memory.close)
        with self.assertRaisesRegex(ValueError, "in-memory"):
            memory.seal()

        with _temporary_directory(self) as directory:
            temporary = Ledger.temporary(directory=directory)
            self.addCleanup(temporary.close)
            with self.assertRaisesRegex(ValueError, "temporary"):
                temporary.seal()

            persistent = Ledger.create(Path(directory) / "active.sqlite3")
            self.addCleanup(persistent.close)
            with persistent.transaction():
                with self.assertRaisesRegex(RuntimeError, "active transaction"):
                    persistent.seal()
            persistent._connection.execute("BEGIN IMMEDIATE")
            try:
                with self.assertRaisesRegex(RuntimeError, "active transaction"):
                    persistent.seal()
            finally:
                persistent._connection.execute("ROLLBACK")

    def test_seal_metadata_rejects_non_utf8_text_without_poisoning_storage(
        self,
    ) -> None:
        with _temporary_directory(self) as directory:
            ledger = Ledger.create(Path(directory) / "metadata.sqlite3")
            self.addCleanup(ledger.close)
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                ledger.seal({"invalid": "\ud800"})
            self.assertFalse(ledger.closed)
            self.assertEqual(ledger.entry_count(), 0)
            ledger.append_many((_entry(1),))
            self.assertEqual(ledger.entry_count(), 1)

    def test_seal_never_returns_success_for_a_replaced_database_path(self) -> None:
        with _temporary_directory(self) as directory:
            database_path = Path(directory) / "replaced.sqlite3"
            ledger = Ledger.create(database_path)
            ledger.append_many((_entry(1),))
            original_hash = Ledger._file_sha256
            first_call = True

            def replace_before_hash(path: Path) -> str:
                nonlocal first_call
                if first_call:
                    first_call = False
                    path.write_bytes(b"not a SQLite ledger")
                return original_hash(path)

            with patch.object(
                Ledger,
                "_file_sha256",
                staticmethod(replace_before_hash),
            ):
                with self.assertRaises(LedgerVerificationError):
                    ledger.seal()
            self.assertTrue(ledger.closed)

    def test_verify_wraps_read_only_open_failures(self) -> None:
        with _temporary_directory(self) as directory:
            database_path, _ = self._sealed_ledger(
                Path(directory),
                name="open-failure.sqlite3",
            )
            with patch(
                "microtx_sim.core.ledger.sqlite3.connect",
                side_effect=sqlite3.OperationalError("forced open failure"),
            ):
                with self.assertRaisesRegex(
                    LedgerVerificationError,
                    "opened read-only",
                ):
                    Ledger.verify(database_path)

    def test_verify_detects_manifest_replacement_during_database_hash(self) -> None:
        with _temporary_directory(self) as directory:
            database_path, seal = self._sealed_ledger(
                Path(directory),
                name="manifest-race.sqlite3",
            )
            original_hash = Ledger._file_sha256

            def replace_manifest_during_hash(path: Path) -> str:
                seal.manifest_path.write_text("{}", encoding="utf-8")
                return original_hash(path)

            with patch.object(
                Ledger,
                "_file_sha256",
                staticmethod(replace_manifest_during_hash),
            ):
                with self.assertRaisesRegex(
                    LedgerVerificationError,
                    "manifest changed",
                ):
                    Ledger.verify(database_path)

    def test_verify_rejects_schema_and_journal_contract_drift(self) -> None:
        with _temporary_directory(self) as directory:
            root = Path(directory)
            for index, (mutation, expected) in enumerate(
                (
                    ("DROP TRIGGER ledger_sealed_update", "schema contract"),
                    ("PRAGMA journal_mode = WAL", "journal mode|sidecar"),
                )
            ):
                with self.subTest(mutation=mutation):
                    database_path, seal = self._sealed_ledger(
                        root,
                        name=f"contract-{index}.sqlite3",
                    )
                    connection = sqlite3.connect(
                        database_path,
                        isolation_level=None,
                    )
                    try:
                        connection.execute(mutation)
                    finally:
                        connection.close()
                    manifest = json.loads(seal.manifest_path.read_text("utf-8"))
                    manifest["file_sha256"] = hashlib.sha256(
                        database_path.read_bytes()
                    ).hexdigest()
                    seal.manifest_path.write_text(
                        json.dumps(
                            manifest,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(LedgerVerificationError, expected):
                        Ledger.verify(database_path)

    def test_atomic_manifest_publication_never_clobbers_an_existing_file(
        self,
    ) -> None:
        with _temporary_directory(self) as directory:
            manifest_path = Path(directory) / "occupied.seal.json"
            manifest_path.write_bytes(b"existing manifest")
            with self.assertRaises(FileExistsError):
                Ledger._write_manifest_atomic(manifest_path, {"replacement": True})
            self.assertEqual(manifest_path.read_bytes(), b"existing manifest")
            self.assertEqual(
                tuple(manifest_path.parent.glob(f".{manifest_path.name}.*.tmp")),
                (),
            )

    def test_verify_rejects_unsealed_and_partially_sealed_artifacts(self) -> None:
        with _temporary_directory(self) as directory:
            root = Path(directory)
            unsealed_path = root / "unsealed.sqlite3"
            unsealed = Ledger.create(unsealed_path)
            unsealed.append_many((_entry(1),))
            unsealed.close()
            with self.assertRaisesRegex(LedgerVerificationError, "incomplete"):
                Ledger.verify(unsealed_path)

            partial_path = root / "partial.sqlite3"
            partial = Ledger.create(partial_path)
            partial._connection.execute(
                "INSERT INTO ledger_seal(singleton, payload_json) VALUES (1, '{}')"
            )
            partial.close()
            with self.assertRaisesRegex(LedgerVerificationError, "incomplete"):
                Ledger.verify(partial_path)

    def test_verify_rejects_manifest_physical_logical_and_truncation_tamper(
        self,
    ) -> None:
        with _temporary_directory(self) as directory:
            database_path, seal = self._sealed_ledger(Path(directory))
            manifest_path = seal.manifest_path
            original_database = database_path.read_bytes()
            original_manifest = manifest_path.read_bytes()

            manifest = json.loads(original_manifest.decode("utf-8"))
            manifest["file_sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LedgerVerificationError, "SHA-256"):
                Ledger.verify(database_path)

            manifest_path.write_bytes(original_manifest)
            database_path.write_bytes(original_database + b"physical-only-tamper")
            with self.assertRaises(LedgerVerificationError):
                Ledger.verify(database_path)

            database_path.write_bytes(original_database)
            manifest_path.write_bytes(original_manifest)
            database_path.write_bytes(original_database[: len(original_database) // 2])
            with self.assertRaises(LedgerVerificationError):
                Ledger.verify(database_path)

            database_path.write_bytes(original_database)
            manifest_path.write_bytes(original_manifest)
            connection = sqlite3.connect(database_path, isolation_level=None)
            try:
                trigger_sql = connection.execute(
                    "SELECT sql FROM sqlite_schema "
                    "WHERE name = 'ledger_sealed_update'"
                ).fetchone()[0]
                connection.execute("DROP TRIGGER ledger_sealed_update")
                connection.execute(
                    "UPDATE ledger_entry SET amount_cents = amount_cents + 1 "
                    "WHERE sequence = 1"
                )
                connection.execute(trigger_sql)
            finally:
                connection.close()
            manifest = json.loads(original_manifest.decode("utf-8"))
            manifest["file_sha256"] = hashlib.sha256(
                database_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LedgerVerificationError, "logical_sha256"):
                Ledger.verify(database_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
