from __future__ import annotations

from enum import Enum, IntEnum, auto


class ProvenanceStatus(str, Enum):
    CALIBRATED = "CALIBRATED"
    ANCHORED = "ANCHORED"
    ILLUSTRATIVE = "ILLUSTRATIVE"
    SYNTHETIC = "SYNTHETIC"


class LedgerBackend(str, Enum):
    """Authoritative storage used for double-entry accounting records."""

    MEMORY = "memory"
    SQLITE = "sqlite"


class Motive(IntEnum):
    COMPETITION = 0
    COLLECTION = 1
    SOCIAL = 2
    EXPLORATION = 3
    RELAXATION = 4


class SpendSegment(str, Enum):
    NON_PAYER = "non_payer"
    MINNOW = "minnow"
    DOLPHIN = "dolphin"
    WHALE = "whale"


class MonetisationMechanism(IntEnum):
    POWER_SALE = 0
    RANDOM_REWARD = 1
    ARTIFICIAL_SCARCITY = 2
    SOCIAL_PRESSURE = 3
    PRICE_OBFUSCATION = 4
    PAYMENT_FRICTION_REMOVAL = 5


class HarmDimension(IntEnum):
    FINANCIAL_STRESS = 0
    ESSENTIAL_SPEND_DISPLACEMENT = 1
    DEBT = 2
    UNAUTHORISED_SPEND = 3
    LOSS_OF_CONTROL = 4
    FUNCTIONING_IMPAIRMENT = 5
    REGRET = 6


class FirmAction(str, Enum):
    HOLD = "hold"
    RELEASE_CONTENT = "release_content"
    ADJUST_MONETISATION = "adjust_monetisation"
    BUY_RESEARCH = "buy_research"
    INVEST_COMPLIANCE = "invest_compliance"
    ACQUIRE_USERS = "acquire_users"
    PROPOSE_COLLABORATION = "propose_collaboration"
    PROPOSE_COLLUSION = "propose_collusion"
    EVADE = "evade"
    APPLY_SUBSIDY = "apply_subsidy"


class EventKind(IntEnum):
    FIRM_DECISION = auto()
    CONTENT_RELEASE = auto()
    PUBLISH_RANKING = auto()
    AUDIT_DUE = auto()
    AUDIT_RESOLUTION = auto()
    SUBSIDY_REVIEW = auto()
    AGREEMENT_REVIEW = auto()
    REFUND_RESOLUTION = auto()


class InformationSource(str, Enum):
    PERSONAL_EXPERIENCE = "personal_experience"
    PUBLIC_RANKING = "public_ranking"
    COMPANY_TELEMETRY = "company_telemetry"
    PAID_RESEARCH = "paid_research"
    COMPLAINT = "complaint"
    AUDIT_EVIDENCE = "audit_evidence"
    DISCLOSURE = "disclosure"
