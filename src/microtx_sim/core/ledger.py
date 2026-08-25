from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
from typing import Any

from ..types import LedgerBackend


INT64_MAX = 2**63 - 1
_SCHEMA_VERSION = 1
_APPLICATION_ID = 0x4D54584C  # ``MTXL``
_CACHE_SIZE_KIB = 2_048
_REFERENCE_QUERY_CHUNK = 400
_ITERATION_CHUNK = 256
_LOGICAL_HASH_DOMAIN = b"microtx-sim-ledger-logical-v1\x00"
_SCHEMA_HASH_DOMAIN = b"microtx-sim-ledger-schema-v1\x00"
_EXPECTED_SCHEMA_SHA256 = (
    "f1387313d7bc8e0c6a7d60682c445aebcde13519d85fc2b1c78f2dfda9807f04"
)
_SEAL_FORMAT = "microtx-sim-ledger-seal-v1"
_SEAL_SUFFIX = ".seal.json"
_SQLITE_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")


class LedgerVerificationError(ValueError):
    """Raised when a sealed ledger cannot be verified exactly."""


class LedgerStorageError(RuntimeError):
    """Raised when SQLite cannot satisfy the declared storage contract."""


def _require_int64(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer")
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} cannot be negative")
        raise ValueError(f"{name} must be positive")
    if value > INT64_MAX:
        raise OverflowError(f"{name} is outside signed int64")
    return value


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in string")
    if not value:
        raise ValueError(f"{name} cannot be empty")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8 text") from exc
    return value


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """A single balanced transfer expressed in exact signed-int64 cents.

    ``debit_account`` is the source of funds and ``credit_account`` the
    destination. External injections must still name a counter-account (for
    example ``external:income``) so every movement remains auditable.
    """

    tick: int
    debit_account: str
    credit_account: str
    amount_cents: int
    kind: str
    reference: str

    def __post_init__(self) -> None:
        _require_int64(self.tick, name="tick", minimum=0)
        _require_int64(self.amount_cents, name="amount_cents", minimum=1)
        debit = _require_text(self.debit_account, name="debit_account")
        credit = _require_text(self.credit_account, name="credit_account")
        _require_text(self.kind, name="kind")
        _require_text(self.reference, name="reference")
        if debit == credit:
            raise ValueError("source and destination accounts must differ")


@dataclass(frozen=True, slots=True)
class LedgerBalanceSnapshot:
    """Backend-independent logical state used by paired-world balance checks."""

    entries: tuple[LedgerEntry, ...]
    logical_sha256: str

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or any(
            type(entry) is not LedgerEntry for entry in self.entries
        ):
            raise TypeError("ledger balance entries must be a LedgerEntry tuple")
        if (
            type(self.logical_sha256) is not str
            or len(self.logical_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.logical_sha256)
        ):
            raise ValueError("ledger logical_sha256 must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class LedgerSeal:
    """Verified finalization metadata for one persistent ledger artifact."""

    database_path: Path
    manifest_path: Path
    schema_version: int
    entry_count: int
    logical_sha256: str
    file_sha256: str
    minimum_tick: int | None
    maximum_tick: int | None
    metadata_json: str
    sqlite_version: str
    journal_mode: str
    synchronous: int
    cache_size: int

    @property
    def metadata(self) -> dict[str, object]:
        value = json.loads(self.metadata_json)
        if type(value) is not dict:  # pragma: no cover - constructor is internal.
            raise LedgerVerificationError("sealed ledger metadata is not an object")
        return value


class Ledger:
    """Append-only SQLite transfer ledger with an in-memory default.

    ``Ledger()`` uses an in-memory SQLite database. ``Ledger.create(path)``
    creates a fresh persistent database, while ``Ledger.temporary()`` owns and
    removes a temporary database directory when closed. The compatibility
    ``entries`` property deliberately materialises the complete history; kernel
    code should prefer the streaming and indexed methods.
    """

    __slots__ = (
        "_backend",
        "_cache_size",
        "_closed",
        "_connection",
        "_journal_mode",
        "_path",
        "_savepoint_serial",
        "_sealed",
        "_storage_token",
        "_synchronous",
        "_temporary_directory",
        "_transaction_depth",
    )

    def __init__(self) -> None:
        connection = sqlite3.connect(
            ":memory:",
            isolation_level=None,
            timeout=5.0,
        )
        self._install(
            connection,
            backend=LedgerBackend.MEMORY,
            path=None,
            temporary_directory=None,
        )

    @classmethod
    def create(cls, path: str | Path) -> "Ledger":
        """Create a fresh persistent ledger, refusing any existing artifact."""

        database_path = Path(path).expanduser().resolve()
        manifest_path = cls._manifest_path(database_path)
        reserved_artifacts = (
            database_path,
            manifest_path,
            *cls._sidecar_paths(database_path),
        )
        for artifact in reserved_artifacts:
            if os.path.lexists(artifact):
                raise FileExistsError(f"ledger artifact already exists: {artifact}")
        if not database_path.parent.exists():
            raise FileNotFoundError(
                f"ledger parent directory does not exist: {database_path.parent}"
            )
        if not database_path.parent.is_dir():
            raise NotADirectoryError(database_path.parent)

        descriptor = os.open(
            database_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.close(descriptor)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                database_path,
                isolation_level=None,
                timeout=30.0,
            )
            ledger = cls.__new__(cls)
            ledger._install(
                connection,
                backend=LedgerBackend.SQLITE,
                path=database_path,
                temporary_directory=None,
            )
            return ledger
        except BaseException:
            if connection is not None:
                connection.close()
            for sidecar in cls._sidecar_paths(database_path):
                with suppress(OSError):
                    sidecar.unlink(missing_ok=True)
            database_path.unlink(missing_ok=True)
            raise

    @classmethod
    def temporary(
        cls,
        *,
        directory: str | Path | None = None,
    ) -> "Ledger":
        """Create a file-backed ledger whose complete directory is owned here."""

        parent = None if directory is None else str(Path(directory).expanduser().resolve())
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="microtx-sim-ledger-",
            dir=parent,
        )
        database_path = Path(temporary_directory.name) / "ledger.sqlite3"
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                database_path,
                isolation_level=None,
                timeout=30.0,
            )
            ledger = cls.__new__(cls)
            ledger._install(
                connection,
                backend=LedgerBackend.SQLITE,
                path=database_path.resolve(),
                temporary_directory=temporary_directory,
            )
            return ledger
        except BaseException:
            if connection is not None:
                connection.close()
            temporary_directory.cleanup()
            raise

    def _install(
        self,
        connection: sqlite3.Connection,
        *,
        backend: LedgerBackend,
        path: Path | None,
        temporary_directory: tempfile.TemporaryDirectory[str] | None,
    ) -> None:
        self._backend = backend
        self._path = path
        self._temporary_directory = temporary_directory
        self._connection = connection
        self._closed = False
        self._sealed = False
        self._transaction_depth = 0
        self._savepoint_serial = 0
        self._storage_token = object()
        try:
            self._configure_connection()
            self._create_schema()
            if path is not None:
                status = path.stat()
                self._storage_token = (
                    "sqlite-file",
                    int(status.st_dev),
                    int(status.st_ino),
                    int(status.st_ctime_ns),
                )
        except BaseException:
            connection.close()
            self._closed = True
            raise

    def _configure_connection(self) -> None:
        connection = self._connection
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA mmap_size = 0")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute(f"PRAGMA cache_size = {-_CACHE_SIZE_KIB}")
        connection.execute("PRAGMA synchronous = EXTRA")
        journal_row = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        journal_mode = "" if journal_row is None else str(journal_row[0]).lower()
        if self._path is not None and journal_mode != "delete":
            raise LedgerStorageError(
                f"persistent ledger requires DELETE journal mode; got {journal_mode!r}"
            )
        synchronous_row = connection.execute("PRAGMA synchronous").fetchone()
        cache_row = connection.execute("PRAGMA cache_size").fetchone()
        if synchronous_row is None or cache_row is None:
            raise LedgerStorageError(
                "SQLite did not report effective durability settings"
            )
        self._journal_mode = journal_mode
        self._synchronous = int(synchronous_row[0])
        self._cache_size = int(cache_row[0])
        if self._synchronous != 3:
            raise LedgerStorageError(
                "SQLite did not enable EXTRA synchronous durability"
            )
        if self._cache_size >= 0 or abs(self._cache_size) > _CACHE_SIZE_KIB:
            raise LedgerStorageError("SQLite did not enable the bounded ledger cache")

    def _create_schema(self) -> None:
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            statements = (
                f"""
                CREATE TABLE ledger_entry (
                    sequence INTEGER PRIMARY KEY,
                    tick INTEGER NOT NULL
                        CHECK(typeof(tick) = 'integer')
                        CHECK(tick >= 0 AND tick <= {INT64_MAX}),
                    debit_account TEXT COLLATE BINARY NOT NULL
                        CHECK(typeof(debit_account) = 'text')
                        CHECK(length(CAST(debit_account AS BLOB)) > 0),
                    credit_account TEXT COLLATE BINARY NOT NULL
                        CHECK(typeof(credit_account) = 'text')
                        CHECK(length(CAST(credit_account AS BLOB)) > 0),
                    amount_cents INTEGER NOT NULL
                        CHECK(typeof(amount_cents) = 'integer')
                        CHECK(amount_cents > 0 AND amount_cents <= {INT64_MAX}),
                    kind TEXT COLLATE BINARY NOT NULL
                        CHECK(typeof(kind) = 'text')
                        CHECK(length(CAST(kind AS BLOB)) > 0),
                    reference TEXT COLLATE BINARY NOT NULL UNIQUE
                        CHECK(typeof(reference) = 'text')
                        CHECK(length(CAST(reference AS BLOB)) > 0),
                    CHECK(debit_account <> credit_account)
                ) STRICT
                """,
                """
                CREATE INDEX ledger_entry_kind_sequence
                    ON ledger_entry(kind COLLATE BINARY, sequence)
                """,
                """
                CREATE TABLE ledger_seal (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    payload_json TEXT COLLATE BINARY NOT NULL
                        CHECK(typeof(payload_json) = 'text')
                        CHECK(length(CAST(payload_json AS BLOB)) > 0)
                ) STRICT
                """,
                """
                CREATE TRIGGER ledger_sealed_insert
                BEFORE INSERT ON ledger_entry
                WHEN EXISTS (SELECT 1 FROM ledger_seal WHERE singleton = 1)
                BEGIN
                    SELECT RAISE(ABORT, 'ledger is sealed');
                END
                """,
                """
                CREATE TRIGGER ledger_sealed_update
                BEFORE UPDATE ON ledger_entry
                WHEN EXISTS (SELECT 1 FROM ledger_seal WHERE singleton = 1)
                BEGIN
                    SELECT RAISE(ABORT, 'ledger is sealed');
                END
                """,
                """
                CREATE TRIGGER ledger_sealed_delete
                BEFORE DELETE ON ledger_entry
                WHEN EXISTS (SELECT 1 FROM ledger_seal WHERE singleton = 1)
                BEGIN
                    SELECT RAISE(ABORT, 'ledger is sealed');
                END
                """,
            )
            for statement in statements:
                connection.execute(statement)
            connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @property
    def backend(self) -> LedgerBackend:
        return self._backend

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def temporary_store(self) -> bool:
        return self._temporary_directory is not None

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Return a detached, ordered compatibility snapshot of all entries."""

        return tuple(self.iter_entries())

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ledger is closed")

    def _ensure_writable(self) -> None:
        self._ensure_open()
        if self._sealed:
            raise RuntimeError("ledger is sealed")

    @contextmanager
    def transaction(self) -> Iterator["Ledger"]:
        """Open an explicit transaction, nesting subsequent scopes as savepoints."""

        self._ensure_writable()
        connection = self._connection
        if not connection.in_transaction:
            connection.execute("BEGIN IMMEDIATE")
            self._transaction_depth += 1
            try:
                yield self
            except BaseException:
                if connection.in_transaction:
                    with suppress(sqlite3.DatabaseError):
                        connection.execute("ROLLBACK")
                raise
            else:
                try:
                    connection.execute("COMMIT")
                except BaseException:
                    if connection.in_transaction:
                        with suppress(sqlite3.DatabaseError):
                            connection.execute("ROLLBACK")
                    raise
            finally:
                self._transaction_depth -= 1
            return

        savepoint = f"ledger_sp_{self._savepoint_serial}"
        self._savepoint_serial += 1
        connection.execute(f"SAVEPOINT {savepoint}")
        self._transaction_depth += 1
        try:
            yield self
        except BaseException:
            if connection.in_transaction:
                with suppress(sqlite3.DatabaseError):
                    connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                if connection.in_transaction:
                    with suppress(sqlite3.DatabaseError):
                        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            try:
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            except BaseException:
                if connection.in_transaction:
                    with suppress(sqlite3.DatabaseError):
                        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    if connection.in_transaction:
                        with suppress(sqlite3.DatabaseError):
                            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise
        finally:
            self._transaction_depth -= 1

    @contextmanager
    def root_transaction(self) -> Iterator["Ledger"]:
        """Commit one durable root transaction or reject an existing outer scope."""

        self._ensure_writable()
        if self._transaction_depth or self._connection.in_transaction:
            raise RuntimeError(
                "root ledger transaction requires no active outer transaction"
            )
        with self.transaction():
            yield self

    def transfer(
        self,
        *,
        tick: int,
        debit_account: str,
        credit_account: str,
        amount_cents: int,
        kind: str,
        reference: str,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            tick=tick,
            debit_account=debit_account,
            credit_account=credit_account,
            amount_cents=amount_cents,
            kind=kind,
            reference=reference,
        )
        self.append_many((entry,))
        return entry

    def append_many(
        self,
        entries: Iterable[LedgerEntry],
    ) -> tuple[LedgerEntry, ...]:
        """Validate and append a complete batch in insertion order or not at all."""

        self._ensure_writable()
        with self.transaction():
            materialized = tuple(entries)
            for entry in materialized:
                if type(entry) is not LedgerEntry:
                    raise TypeError("ledger batches must contain LedgerEntry values")
                self._validate_entry(entry)

            references = tuple(entry.reference for entry in materialized)
            self._raise_internal_reference_duplicate(references)
            if not materialized:
                return ()

            conflicts = self._conflicting_references_materialized(references)
            if conflicts:
                duplicate = next(
                    reference for reference in references if reference in conflicts
                )
                raise ValueError(f"duplicate ledger reference: {duplicate}")

            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM ledger_entry"
            ).fetchone()
            if row is None:
                raise RuntimeError("SQLite did not return the ledger sequence")
            first_sequence = int(row[0]) + 1
            if first_sequence > INT64_MAX - len(materialized) + 1:
                raise OverflowError("ledger sequence would exceed signed int64")
            try:
                for offset, entry in enumerate(materialized):
                    self._connection.execute(
                        """
                        INSERT INTO ledger_entry (
                            sequence,
                            tick,
                            debit_account,
                            credit_account,
                            amount_cents,
                            kind,
                            reference
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            first_sequence + offset,
                            entry.tick,
                            entry.debit_account,
                            entry.credit_account,
                            entry.amount_cents,
                            entry.kind,
                            entry.reference,
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                if (
                    getattr(exc, "sqlite_errorcode", None)
                    == sqlite3.SQLITE_CONSTRAINT_UNIQUE
                    and "ledger_entry.reference" in str(exc)
                ):
                    raise ValueError(
                        f"duplicate ledger reference: {entry.reference}"
                    ) from exc
                raise
        return materialized

    def extend(self, entries: Iterable[LedgerEntry]) -> None:
        self.append_many(entries)

    @staticmethod
    def _validate_entry(entry: LedgerEntry) -> None:
        _require_int64(entry.tick, name="tick", minimum=0)
        _require_int64(entry.amount_cents, name="amount_cents", minimum=1)
        debit = _require_text(entry.debit_account, name="debit_account")
        credit = _require_text(entry.credit_account, name="credit_account")
        _require_text(entry.kind, name="kind")
        _require_text(entry.reference, name="reference")
        if debit == credit:
            raise ValueError("source and destination accounts must differ")

    @staticmethod
    def _raise_internal_reference_duplicate(references: tuple[str, ...]) -> None:
        seen: set[str] = set()
        for reference in references:
            if reference in seen:
                raise ValueError(f"duplicate ledger reference: {reference}")
            seen.add(reference)

    def conflicting_references(
        self,
        references: Iterable[str],
    ) -> frozenset[str]:
        """Return references already present using the backend's unique index."""

        self._ensure_open()
        materialized = tuple(references)
        for reference in materialized:
            _require_text(reference, name="reference")
        return self._conflicting_references_materialized(materialized)

    def validate_references(self, references: Iterable[str]) -> None:
        """Reject duplicates within a candidate batch or in existing storage."""

        self._ensure_open()
        materialized = tuple(references)
        for reference in materialized:
            _require_text(reference, name="reference")
        self._raise_internal_reference_duplicate(materialized)
        conflicts = self._conflicting_references_materialized(materialized)
        if conflicts:
            duplicate = next(
                reference for reference in materialized if reference in conflicts
            )
            raise ValueError(f"duplicate ledger reference: {duplicate}")

    def _conflicting_references_materialized(
        self,
        references: tuple[str, ...],
    ) -> frozenset[str]:
        conflicts: set[str] = set()
        for start in range(0, len(references), _REFERENCE_QUERY_CHUNK):
            chunk = references[start : start + _REFERENCE_QUERY_CHUNK]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                "SELECT reference FROM ledger_entry "
                f"WHERE reference COLLATE BINARY IN ({placeholders})",
                chunk,
            )
            conflicts.update(str(row[0]) for row in rows)
        return frozenset(conflicts)

    @staticmethod
    def _validate_kind(kind: str | None) -> str | None:
        if kind is None:
            return None
        return _require_text(kind, name="kind")

    def iter_entries(self, kind: str | None = None) -> Iterator[LedgerEntry]:
        """Stream immutable entries in their explicit append sequence."""

        self._ensure_open()
        selected_kind = self._validate_kind(kind)
        high_water_row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM ledger_entry"
        ).fetchone()
        if high_water_row is None:
            raise RuntimeError("SQLite did not return the ledger high-water sequence")
        high_water = int(high_water_row[0])
        if selected_kind is None:
            cursor = self._connection.execute(
                """
                SELECT tick, debit_account, credit_account, amount_cents, kind,
                       reference
                FROM ledger_entry
                WHERE sequence <= ?
                ORDER BY sequence
                """,
                (high_water,),
            )
        else:
            cursor = self._connection.execute(
                """
                SELECT tick, debit_account, credit_account, amount_cents, kind,
                       reference
                FROM ledger_entry
                WHERE kind = ? COLLATE BINARY AND sequence <= ?
                ORDER BY sequence
                """,
                (selected_kind, high_water),
            )

        def generate() -> Iterator[LedgerEntry]:
            try:
                while rows := cursor.fetchmany(_ITERATION_CHUNK):
                    for row in rows:
                        yield LedgerEntry(
                            tick=int(row[0]),
                            debit_account=str(row[1]),
                            credit_account=str(row[2]),
                            amount_cents=int(row[3]),
                            kind=str(row[4]),
                            reference=str(row[5]),
                        )
            finally:
                cursor.close()

        return generate()

    def entry_count(self, kind: str | None = None) -> int:
        self._ensure_open()
        selected_kind = self._validate_kind(kind)
        if selected_kind is None:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM ledger_entry"
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM ledger_entry WHERE kind = ? COLLATE BINARY",
                (selected_kind,),
            ).fetchone()
        if row is None:
            raise RuntimeError("SQLite did not return the ledger entry count")
        return int(row[0])

    def account_net_cents(self) -> dict[str, int]:
        self._ensure_open()
        net: defaultdict[str, int] = defaultdict(int)
        cursor = self._connection.execute(
            """
            SELECT debit_account, credit_account, amount_cents
            FROM ledger_entry
            ORDER BY sequence
            """
        )
        try:
            while rows := cursor.fetchmany(_ITERATION_CHUNK):
                for debit_account, credit_account, amount_cents in rows:
                    amount = int(amount_cents)
                    net[str(debit_account)] -= amount
                    net[str(credit_account)] += amount
        finally:
            cursor.close()
        return dict(net)

    def total_flow_cents(self, *, kind: str | None = None) -> int:
        self._ensure_open()
        selected_kind = self._validate_kind(kind)
        if selected_kind is None:
            cursor = self._connection.execute(
                "SELECT amount_cents FROM ledger_entry ORDER BY sequence"
            )
        else:
            cursor = self._connection.execute(
                """
                SELECT amount_cents
                FROM ledger_entry
                WHERE kind = ? COLLATE BINARY
                ORDER BY sequence
                """,
                (selected_kind,),
            )
        total = 0
        try:
            while rows := cursor.fetchmany(_ITERATION_CHUNK):
                for (amount_cents,) in rows:
                    total += int(amount_cents)
        finally:
            cursor.close()
        return total

    def assert_balanced(self, *, full: bool = False) -> None:
        """Assert structural balance, optionally including full storage checks.

        A schema-valid transfer row is intrinsically balanced because it stores
        one amount with one source and one destination. The default check is
        therefore constant-space and does not rescan an append-only history.
        ``full=True`` additionally runs SQLite integrity checking and recomputes
        every account net in Python.
        """

        self._ensure_open()
        if type(full) is not bool:
            raise TypeError("full must be a built-in boolean")
        if not full:
            return
        self._validate_schema_identity(self._connection)
        self._assert_storage_integrity(self._connection)
        self._logical_summary_from_connection(self._connection)
        if sum(self.account_net_cents().values()) != 0:
            raise AssertionError("ledger is not balanced")

    @staticmethod
    def _assert_storage_integrity(connection: sqlite3.Connection) -> None:
        rows = tuple(str(row[0]) for row in connection.execute("PRAGMA integrity_check"))
        if rows != ("ok",):
            detail = "; ".join(rows) if rows else "no result"
            raise LedgerVerificationError(
                f"SQLite ledger integrity check failed: {detail}"
            )

    @staticmethod
    def _update_logical_hash(
        digest: Any,
        sequence: int,
        entry: LedgerEntry,
    ) -> None:
        digest.update(struct.pack(">Qqq", sequence, entry.tick, entry.amount_cents))
        for value in (
            entry.debit_account,
            entry.credit_account,
            entry.kind,
            entry.reference,
        ):
            encoded = value.encode("utf-8")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)

    @classmethod
    def _logical_summary_from_entries(
        cls,
        entries: Iterable[LedgerEntry],
    ) -> tuple[int, str, int | None, int | None]:
        digest = hashlib.sha256(_LOGICAL_HASH_DOMAIN)
        count = 0
        minimum_tick: int | None = None
        maximum_tick: int | None = None
        for count, entry in enumerate(entries, start=1):
            cls._update_logical_hash(digest, count, entry)
            minimum_tick = (
                entry.tick if minimum_tick is None else min(minimum_tick, entry.tick)
            )
            maximum_tick = (
                entry.tick if maximum_tick is None else max(maximum_tick, entry.tick)
            )
        return count, digest.hexdigest(), minimum_tick, maximum_tick

    def logical_sha256(self) -> str:
        self._ensure_open()
        return self._logical_summary_from_connection(self._connection)[1]

    def balance_snapshot(self) -> LedgerBalanceSnapshot:
        self._ensure_open()
        entries = self.entries
        logical_sha256 = self._logical_summary_from_entries(entries)[1]
        return LedgerBalanceSnapshot(
            entries=entries,
            logical_sha256=logical_sha256,
        )

    def shares_storage_with(self, other: object) -> bool:
        """Return whether another facade addresses the same physical store."""

        if not isinstance(other, Ledger):
            return False
        return bool(
            self is other
            or self._connection is other._connection
            or self._storage_token == other._storage_token
        )

    @staticmethod
    def _canonical_json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _normalise_metadata(
        cls,
        metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise TypeError("ledger seal metadata must be a mapping")
        result = dict(metadata)
        if any(type(key) is not str for key in result):
            raise TypeError("ledger seal metadata keys must be built-in strings")
        try:
            encoded = cls._canonical_json(result)
            encoded.encode("utf-8")
            decoded = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError("ledger seal metadata must be finite JSON data") from exc
        if type(decoded) is not dict:  # pragma: no cover - result starts as a dict.
            raise ValueError("ledger seal metadata must be a JSON object")
        return decoded

    @staticmethod
    def _manifest_path(database_path: Path) -> Path:
        return database_path.with_name(database_path.name + _SEAL_SUFFIX)

    @staticmethod
    def _sidecar_paths(database_path: Path) -> tuple[Path, ...]:
        return tuple(
            Path(str(database_path) + suffix)
            for suffix in _SQLITE_SIDECAR_SUFFIXES
        )

    @classmethod
    def _assert_no_sidecars(cls, database_path: Path) -> None:
        for sidecar in cls._sidecar_paths(database_path):
            if os.path.lexists(sidecar):
                raise LedgerVerificationError(
                    f"unexpected SQLite sidecar prevents exact verification: {sidecar}"
                )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _write_manifest_atomic(cls, path: Path, payload: dict[str, object]) -> None:
        rendered = (cls._canonical_json(payload) + "\n").encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            # A hard-link publish is atomic and, unlike os.replace(), cannot
            # clobber a manifest created by another process after preflight.
            os.link(temporary_path, path)
            temporary_path.unlink()
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise

    def seal(
        self,
        metadata: Mapping[str, object] | None = None,
    ) -> LedgerSeal:
        """Finalize, close, hash, and manifest a persistent ledger artifact."""

        self._ensure_writable()
        if self._path is None:
            raise ValueError("an in-memory ledger cannot be sealed as a file artifact")
        if self._temporary_directory is not None:
            raise ValueError("an owned temporary ledger cannot be sealed persistently")
        if self._transaction_depth or self._connection.in_transaction:
            raise RuntimeError("cannot seal a ledger inside an active transaction")
        database_path = self._path
        manifest_path = self._manifest_path(database_path)
        if os.path.lexists(manifest_path):
            raise FileExistsError(f"ledger seal already exists: {manifest_path}")
        self._assert_no_sidecars(database_path)
        normalised_metadata = self._normalise_metadata(metadata)

        with self.transaction():
            existing = self._connection.execute(
                "SELECT payload_json FROM ledger_seal WHERE singleton = 1"
            ).fetchone()
            if existing is not None:
                raise RuntimeError("ledger is already sealed")
            self._validate_persistent_contract(self._connection)
            self._assert_storage_integrity(self._connection)
            count, logical_hash, minimum_tick, maximum_tick = (
                self._logical_summary_from_connection(self._connection)
            )
            core: dict[str, object] = {
                "format": _SEAL_FORMAT,
                "schema_version": _SCHEMA_VERSION,
                "application_id": _APPLICATION_ID,
                "backend": self._backend.value,
                "entry_count": count,
                "logical_sha256": logical_hash,
                "minimum_tick": minimum_tick,
                "maximum_tick": maximum_tick,
                "metadata": normalised_metadata,
                "sqlite_version": sqlite3.sqlite_version,
                "journal_mode": self._journal_mode,
                "synchronous": self._synchronous,
                "cache_size": self._cache_size,
            }
            core_json = self._canonical_json(core)
            self._connection.execute(
                "INSERT INTO ledger_seal(singleton, payload_json) VALUES (1, ?)",
                (core_json,),
            )

        self._sealed = True
        self.close()
        self._assert_no_sidecars(database_path)
        file_hash = self._file_sha256(database_path)
        manifest: dict[str, object] = {
            "database_filename": database_path.name,
            "file_sha256": file_hash,
            "core": core,
        }
        self._write_manifest_atomic(manifest_path, manifest)
        # Verification is deliberately part of finalization. It closes the gap
        # between releasing the writer connection and publishing the manifest:
        # a replaced or modified path is reported as a failed seal, never
        # returned as a successfully completed artifact.
        return self.verify(
            database_path,
            seal_path=manifest_path,
        )

    @classmethod
    def verify(
        cls,
        path: str | Path,
        *,
        seal_path: str | Path | None = None,
    ) -> LedgerSeal:
        """Verify a sealed database read-only and return its trusted metadata."""

        database_path = Path(path).expanduser().resolve()
        manifest_path = (
            cls._manifest_path(database_path)
            if seal_path is None
            else Path(seal_path).expanduser().resolve()
        )
        if not database_path.is_file():
            raise LedgerVerificationError(
                f"ledger database is missing: {database_path}"
            )
        if not manifest_path.is_file():
            raise LedgerVerificationError(
                "ledger seal manifest is missing; artifact is incomplete"
            )
        cls._assert_no_sidecars(database_path)
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LedgerVerificationError("ledger seal manifest is unreadable") from exc
        cls._validate_manifest_shape(manifest, database_path)
        uri = database_path.as_uri() + "?mode=ro"
        try:
            connection = sqlite3.connect(
                uri,
                uri=True,
                isolation_level=None,
                timeout=5.0,
            )
        except (OSError, sqlite3.DatabaseError) as exc:
            raise LedgerVerificationError(
                "ledger database cannot be opened read-only"
            ) from exc
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.execute(f"PRAGMA cache_size = {-_CACHE_SIZE_KIB}")
            connection.execute("BEGIN")
            cls._validate_persistent_contract(connection)
            cls._assert_storage_integrity(connection)
            seal_row = connection.execute(
                "SELECT payload_json FROM ledger_seal WHERE singleton = 1"
            ).fetchone()
            if seal_row is None:
                raise LedgerVerificationError(
                    "ledger database has no finalization seal; artifact is incomplete"
                )
            core = manifest["core"]
            core_json = cls._canonical_json(core)
            if str(seal_row[0]) != core_json:
                raise LedgerVerificationError(
                    "ledger internal seal does not match its manifest"
                )
            count, logical_hash, minimum_tick, maximum_tick = (
                cls._logical_summary_from_connection(connection)
            )
            comparisons = {
                "entry_count": count,
                "logical_sha256": logical_hash,
                "minimum_tick": minimum_tick,
                "maximum_tick": maximum_tick,
            }
            for name, actual in comparisons.items():
                if core.get(name) != actual:
                    raise LedgerVerificationError(
                        f"ledger {name} does not match its finalization seal"
                    )
            expected_file_hash = manifest["file_sha256"]
            actual_file_hash = cls._file_sha256(database_path)
            if actual_file_hash != expected_file_hash:
                raise LedgerVerificationError("ledger database file SHA-256 mismatch")
            cls._assert_no_sidecars(database_path)
            try:
                current_manifest_bytes = manifest_path.read_bytes()
            except OSError as exc:
                raise LedgerVerificationError(
                    "ledger seal manifest changed during verification"
                ) from exc
            if current_manifest_bytes != manifest_bytes:
                raise LedgerVerificationError(
                    "ledger seal manifest changed during verification"
                )
        except LedgerVerificationError:
            raise
        except sqlite3.DatabaseError as exc:
            raise LedgerVerificationError("ledger database cannot be verified") from exc
        except OSError as exc:
            raise LedgerVerificationError("ledger database cannot be hashed") from exc
        except (TypeError, ValueError, OverflowError) as exc:
            raise LedgerVerificationError("ledger logical rows cannot be verified") from exc
        finally:
            if connection.in_transaction:
                with suppress(sqlite3.DatabaseError):
                    connection.execute("ROLLBACK")
            with suppress(sqlite3.DatabaseError):
                connection.close()
        return cls._seal_from_manifest(
            database_path,
            manifest_path,
            manifest,
        )

    @classmethod
    def _logical_summary_from_connection(
        cls,
        connection: sqlite3.Connection,
    ) -> tuple[int, str, int | None, int | None]:
        cursor = connection.execute(
            """
            SELECT sequence, tick, debit_account, credit_account, amount_cents,
                   kind, reference
            FROM ledger_entry
            ORDER BY sequence
            """
        )

        def entries() -> Iterator[LedgerEntry]:
            expected_sequence = 1
            try:
                while rows := cursor.fetchmany(_ITERATION_CHUNK):
                    for row in rows:
                        sequence = int(row[0])
                        if sequence != expected_sequence:
                            raise LedgerVerificationError(
                                "ledger append sequence is not contiguous"
                            )
                        expected_sequence += 1
                        yield LedgerEntry(
                            tick=int(row[1]),
                            debit_account=str(row[2]),
                            credit_account=str(row[3]),
                            amount_cents=int(row[4]),
                            kind=str(row[5]),
                            reference=str(row[6]),
                        )
            finally:
                cursor.close()

        return cls._logical_summary_from_entries(entries())

    @classmethod
    def _validate_schema_identity(cls, connection: sqlite3.Connection) -> None:
        application_row = connection.execute("PRAGMA application_id").fetchone()
        version_row = connection.execute("PRAGMA user_version").fetchone()
        if application_row is None or int(application_row[0]) != _APPLICATION_ID:
            raise LedgerVerificationError("ledger application id is invalid")
        if version_row is None or int(version_row[0]) != _SCHEMA_VERSION:
            raise LedgerVerificationError("ledger schema version is unsupported")
        if cls._schema_sha256(connection) != _EXPECTED_SCHEMA_SHA256:
            raise LedgerVerificationError("ledger schema contract is invalid")

    @classmethod
    def _validate_persistent_contract(
        cls,
        connection: sqlite3.Connection,
    ) -> None:
        journal_row = connection.execute("PRAGMA journal_mode").fetchone()
        journal_mode = "" if journal_row is None else str(journal_row[0]).lower()
        if journal_mode != "delete":
            raise LedgerVerificationError(
                f"ledger journal mode must be delete; got {journal_mode!r}"
            )
        cls._validate_schema_identity(connection)

    @staticmethod
    def _schema_sha256(connection: sqlite3.Connection) -> str:
        rows = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE sql IS NOT NULL
            ORDER BY type COLLATE BINARY, name COLLATE BINARY
            """
        )
        digest = hashlib.sha256(_SCHEMA_HASH_DOMAIN)
        for row in rows:
            if len(row) != 4 or any(type(value) is not str for value in row):
                raise LedgerVerificationError(
                    "ledger schema contains a non-text object definition"
                )
            for value in row:
                encoded = " ".join(value.split()).encode("utf-8")
                digest.update(struct.pack(">Q", len(encoded)))
                digest.update(encoded)
        return digest.hexdigest()

    @classmethod
    def _validate_manifest_shape(
        cls,
        manifest: object,
        database_path: Path,
    ) -> None:
        if type(manifest) is not dict or set(manifest) != {
            "database_filename",
            "file_sha256",
            "core",
        }:
            raise LedgerVerificationError("ledger seal manifest has an invalid shape")
        if (
            type(manifest["database_filename"]) is not str
            or manifest["database_filename"] != database_path.name
        ):
            raise LedgerVerificationError(
                "ledger seal manifest names a different database"
            )
        file_hash = manifest["file_sha256"]
        if (
            type(file_hash) is not str
            or len(file_hash) != 64
            or any(character not in "0123456789abcdef" for character in file_hash)
        ):
            raise LedgerVerificationError("ledger file SHA-256 is invalid")
        core = manifest["core"]
        required_core = {
            "format",
            "schema_version",
            "application_id",
            "backend",
            "entry_count",
            "logical_sha256",
            "minimum_tick",
            "maximum_tick",
            "metadata",
            "sqlite_version",
            "journal_mode",
            "synchronous",
            "cache_size",
        }
        if type(core) is not dict or set(core) != required_core:
            raise LedgerVerificationError("ledger seal core has an invalid shape")
        if type(core["format"]) is not str or core["format"] != _SEAL_FORMAT:
            raise LedgerVerificationError("ledger seal format is unsupported")
        if (
            type(core["schema_version"]) is not int
            or core["schema_version"] != _SCHEMA_VERSION
        ):
            raise LedgerVerificationError("ledger seal schema version is unsupported")
        if (
            type(core["application_id"]) is not int
            or core["application_id"] != _APPLICATION_ID
        ):
            raise LedgerVerificationError("ledger seal application id is invalid")
        if (
            type(core["backend"]) is not str
            or core["backend"] != LedgerBackend.SQLITE.value
        ):
            raise LedgerVerificationError("sealed ledger backend must be sqlite")
        if (
            type(core["entry_count"]) is not int
            or core["entry_count"] < 0
            or core["entry_count"] > INT64_MAX
        ):
            raise LedgerVerificationError("ledger sealed entry count is invalid")
        logical_hash = core["logical_sha256"]
        if (
            type(logical_hash) is not str
            or len(logical_hash) != 64
            or any(character not in "0123456789abcdef" for character in logical_hash)
        ):
            raise LedgerVerificationError("ledger logical SHA-256 is invalid")
        for name in ("minimum_tick", "maximum_tick"):
            tick = core[name]
            if tick is not None and (
                type(tick) is not int or tick < 0 or tick > INT64_MAX
            ):
                raise LedgerVerificationError(f"ledger sealed {name} is invalid")
        if (core["minimum_tick"] is None) != (core["maximum_tick"] is None):
            raise LedgerVerificationError("ledger sealed tick range is incomplete")
        if (core["entry_count"] == 0) != (core["minimum_tick"] is None):
            raise LedgerVerificationError(
                "ledger sealed tick range does not match its entry count"
            )
        if (
            core["minimum_tick"] is not None
            and core["minimum_tick"] > core["maximum_tick"]
        ):
            raise LedgerVerificationError("ledger sealed tick range is reversed")
        if type(core["metadata"]) is not dict:
            raise LedgerVerificationError("ledger sealed metadata is not an object")
        if (
            type(core["sqlite_version"]) is not str
            or not core["sqlite_version"]
        ):
            raise LedgerVerificationError("ledger sealed SQLite version is invalid")
        if (
            type(core["journal_mode"]) is not str
            or core["journal_mode"] != "delete"
        ):
            raise LedgerVerificationError("ledger sealed journal mode is invalid")
        if type(core["synchronous"]) is not int or core["synchronous"] != 3:
            raise LedgerVerificationError(
                "ledger sealed synchronous setting is not EXTRA"
            )
        if (
            type(core["cache_size"]) is not int
            or core["cache_size"] >= 0
            or abs(core["cache_size"]) > _CACHE_SIZE_KIB
        ):
            raise LedgerVerificationError("ledger sealed cache size is invalid")
        try:
            cls._canonical_json(core)
        except (TypeError, ValueError) as exc:
            raise LedgerVerificationError("ledger seal contains invalid JSON data") from exc

    @classmethod
    def _seal_from_manifest(
        cls,
        database_path: Path,
        manifest_path: Path,
        manifest: dict[str, object],
    ) -> LedgerSeal:
        core = manifest["core"]
        if type(core) is not dict:  # pragma: no cover - validated at the boundary.
            raise LedgerVerificationError("ledger seal core is not an object")
        metadata = core["metadata"]
        return LedgerSeal(
            database_path=database_path,
            manifest_path=manifest_path,
            schema_version=int(core["schema_version"]),
            entry_count=int(core["entry_count"]),
            logical_sha256=str(core["logical_sha256"]),
            file_sha256=str(manifest["file_sha256"]),
            minimum_tick=(
                None if core["minimum_tick"] is None else int(core["minimum_tick"])
            ),
            maximum_tick=(
                None if core["maximum_tick"] is None else int(core["maximum_tick"])
            ),
            metadata_json=cls._canonical_json(metadata),
            sqlite_version=str(core["sqlite_version"]),
            journal_mode=str(core["journal_mode"]),
            synchronous=int(core["synchronous"]),
            cache_size=int(core["cache_size"]),
        )

    def close(self) -> None:
        """Close storage exactly once and clean up any owned temporary files."""

        if self._closed:
            return
        if self._transaction_depth:
            raise RuntimeError("cannot close a ledger inside an active transaction")
        connection = self._connection
        try:
            if connection.in_transaction:
                with suppress(sqlite3.DatabaseError):
                    connection.execute("ROLLBACK")
        finally:
            try:
                connection.close()
            finally:
                self._closed = True
                self._transaction_depth = 0
                if self._temporary_directory is not None:
                    self._temporary_directory.cleanup()

    def __enter__(self) -> "Ledger":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


__all__ = [
    "INT64_MAX",
    "Ledger",
    "LedgerBalanceSnapshot",
    "LedgerEntry",
    "LedgerSeal",
    "LedgerStorageError",
    "LedgerVerificationError",
]
