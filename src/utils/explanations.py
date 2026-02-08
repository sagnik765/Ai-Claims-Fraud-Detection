from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math


@dataclass
class AmountStats:
    count: int
    mean: float
    median: float
    p75: float
    p90: float
    p95: float


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_values[f]
    d = k - f
    return sorted_values[f] * (1 - d) + sorted_values[c] * d


def compute_amount_stats(records: List[Dict[str, Any]], field: str = "claim_amount") -> Optional[AmountStats]:
    values: List[float] = []
    for record in records:
        val = _safe_float(record.get(field))
        if val is None:
            continue
        values.append(val)

    if not values:
        return None

    values.sort()
    mean = sum(values) / max(1, len(values))
    median = _percentile(values, 0.5)
    p75 = _percentile(values, 0.75)
    p90 = _percentile(values, 0.90)
    p95 = _percentile(values, 0.95)
    return AmountStats(count=len(values), mean=mean, median=median, p75=p75, p90=p90, p95=p95)


def amount_rationale(amount: Optional[float], stats: Optional[AmountStats]) -> str:
    if amount is None:
        return "Claim amount not provided."
    if stats is None:
        return "Insufficient data to benchmark claim amount."

    if amount >= stats.p95:
        return (
            f"Claim amount ${amount:,.0f} is in the top 5% of observed claims "
            f"(p95=${stats.p95:,.0f}); higher scrutiny is typical."
        )
    if amount >= stats.p90:
        return (
            f"Claim amount ${amount:,.0f} exceeds the 90th percentile "
            f"(p90=${stats.p90:,.0f}); validate estimates and documentation."
        )
    if amount >= stats.p75:
        return (
            f"Claim amount ${amount:,.0f} is above the typical range "
            f"(p75=${stats.p75:,.0f}); review scope and pricing assumptions."
        )
    if amount <= stats.median:
        return (
            f"Claim amount ${amount:,.0f} is within the typical range "
            f"(median=${stats.median:,.0f})."
        )
    return (
        f"Claim amount ${amount:,.0f} is near the median range "
        f"(median=${stats.median:,.0f})."
    )


def decline_risk_reasons(record: Dict[str, Any], score: float, threshold: float) -> List[str]:
    reasons: List[str] = []
    if score >= threshold:
        reasons.append(f"Model score {score:.2f} exceeds threshold {threshold:.2f}.")
    if float(record.get("late_reported", 0) or 0) > 0:
        reasons.append("Loss reported late; verify loss date and coverage timing.")
    if float(record.get("policy_age_days", 0) or 0) < 30:
        reasons.append("Policy recently bound; higher risk window.")
    if float(record.get("prior_claims_count", 0) or 0) >= 2:
        reasons.append("Multiple prior claims; check claim history consistency.")
    if float(record.get("multiple_parties", 0) or 0) > 0:
        reasons.append("Multiple parties involved; validate liability and police reports.")
    if float(record.get("injury_reported", 0) or 0) > 0:
        reasons.append("Injury reported; confirm medical documentation and causality.")
    if float(record.get("total_loss", 0) or 0) > 0:
        reasons.append("Total loss indicated; verify valuation and salvage process.")
    return reasons


def decision_support(score: float, threshold: float) -> Dict[str, Any]:
    if score >= threshold + 0.25:
        return {"recommendation": "escalate", "reason": "Very high risk score."}
    if score >= threshold:
        return {"recommendation": "review", "reason": "Score above fraud threshold."}
    if score >= threshold * 0.6:
        return {"recommendation": "standard_review", "reason": "Moderate risk score."}
    return {"recommendation": "standard", "reason": "Low risk score."}


def genai_rationale(
    record: Dict[str, Any],
    score: float,
    threshold: float,
    stats: Optional[AmountStats],
    disclaimer: str,
) -> Dict[str, Any]:
    reasons = decline_risk_reasons(record, score, threshold)
    amount = _safe_float(record.get("claim_amount"))
    amount_note = amount_rationale(amount, stats)
    decision = decision_support(score, threshold)

    if reasons:
        reason_text = "; ".join(reasons)
        summary = (
            f"Decision support: {decision['recommendation']} ({decision['reason']}). "
            f"Key risk indicators: {reason_text}. {amount_note}"
        )
    else:
        summary = (
            f"Decision support: {decision['recommendation']} ({decision['reason']}). "
            f"No major risk indicators detected. {amount_note}"
        )

    return {
        "summary": summary,
        "decline_risk_reasons": reasons,
        "amount_rationale": amount_note,
        "decision_support": decision,
        "disclaimer": disclaimer,
        "genai_mode": "template",
    }
