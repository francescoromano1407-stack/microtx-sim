from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """A single balanced transfer expressed in integer cents.

    `debit_account` is the source of funds and `credit_account` the destination.
    External injections must still name a counter-account (for example
    `external:income`) so every movement remains auditable.
    """

    tick: int
    debit_account: str
    credit_account: str
    amount_cents: int
    kind: str
    reference: str

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("tick cannot be negative")
        if self.amount_cents <= 0:
            raise ValueError("ledger transfers must be positive")
        if not self.debit_account or not self.credit_account:
            raise ValueError("both accounts are required")
        if self.debit_account == self.credit_account:
            raise ValueError("source and destination accounts must differ")


class Ledger:
    """Append-only double-entry transfer log.

    The simulation state remains the operational source of balances. The ledger
    provides a compact, independently recomputable audit trail.
    """

    __slots__ = ("_entries", "_seen_references")

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._seen_references: set[str] = set()

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

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
        if reference in self._seen_references:
            raise ValueError(f"duplicate ledger reference: {reference}")
        entry = LedgerEntry(
            tick=tick,
            debit_account=debit_account,
            credit_account=credit_account,
            amount_cents=int(amount_cents),
            kind=kind,
            reference=reference,
        )
        self._entries.append(entry)
        self._seen_references.add(reference)
        return entry

    def extend(self, entries: Iterable[LedgerEntry]) -> None:
        for entry in entries:
            self.transfer(
                tick=entry.tick,
                debit_account=entry.debit_account,
                credit_account=entry.credit_account,
                amount_cents=entry.amount_cents,
                kind=entry.kind,
                reference=entry.reference,
            )

    def account_net_cents(self) -> dict[str, int]:
        net: defaultdict[str, int] = defaultdict(int)
        for entry in self._entries:
            net[entry.debit_account] -= entry.amount_cents
            net[entry.credit_account] += entry.amount_cents
        return dict(net)

    def total_flow_cents(self, *, kind: str | None = None) -> int:
        return sum(
            entry.amount_cents
            for entry in self._entries
            if kind is None or entry.kind == kind
        )

    def assert_balanced(self) -> None:
        if sum(self.account_net_cents().values()) != 0:
            raise AssertionError("ledger is not balanced")

