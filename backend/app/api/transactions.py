"""
backend/app/api/transactions.py
=================================
AbuseRing Sentinel — Transactions Endpoint
"""
from __future__ import annotations

from fastapi import APIRouter, Request, Query
from typing import Optional

router = APIRouter(tags=["Transactions"])


@router.get("/transactions")
async def list_transactions(
    request: Request,
    customer_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """
    Returns paginated transactions with optional customer filter.
    Each transaction row includes the customer's current risk score.
    """
    data: "DataStore" = request.app.state.data  # type: ignore
    ml   = request.app.state.ml

    if customer_id:
        txns = data.get_transactions_for_customer(customer_id, limit=limit)
        total = len(data.get_transactions_for_customer(customer_id, limit=10_000))
    else:
        txns = data.get_recent_transactions(limit=limit + offset)[offset:]
        total = len(data.transactions)

    # Enrich with risk score
    enriched = []
    for txn in txns:
        cid = str(txn.get("customer_id", ""))
        txn["risk_score"] = ml.predict(cid)
        txn["risk_level"] = ml.get_risk_level(txn["risk_score"])
        enriched.append(txn)

    return {
        "total":       total,
        "limit":       limit,
        "offset":      offset,
        "transactions": enriched,
    }
