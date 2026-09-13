from enum import Enum


class ERSTacticalAction(str, Enum):
    HOLD = "HOLD"
    RECHARGE = "RECHARGE"
    BOOST = "BOOST"
    OVERTAKE = "OVERTAKE"


class ActionType(str, Enum):
    HARVEST = "HARVEST"
    HOLD = "HOLD"
    PARTIAL_DEPLOY = "PARTIAL_DEPLOY"
    FULL_DEPLOY = "FULL_DEPLOY"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DecisionMode(str, Enum):
    VERIFIED = "VERIFIED"
    FAST = "FAST"


class DecisionStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    SAFE_HOLD = "SAFE_HOLD"


class CandidateStatus(str, Enum):
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    ILLEGAL = "ILLEGAL"
