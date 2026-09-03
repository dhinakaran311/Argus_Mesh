"""
backend/app/agents/analyst.py
================================
AbuseRing Sentinel — Analyst Node

Synthesizes all gathered evidence into a structured investigation report
using Groq's llama-3.3-70b-versatile LLM.

Steps:
  1. Build a rich evidence context string
  2. Call Groq with a structured-output prompt
  3. Parse JSON response
  4. Store new case in Qdrant (learning loop)
  5. Emit two SSE events: "reasoning" + "complete"
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .state import InvestigationState
from .tools import store_investigation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior fraud analyst at Razorpay. 
Your job is to analyze evidence about a suspected coordinated abuse ring and produce a structured investigation report.
You must be precise, evidence-driven, and never hallucinate facts not present in the evidence.
Always return valid JSON and nothing else."""

_INVESTIGATION_PROMPT = """
## CLUSTER EVIDENCE

### Cluster Facts
{facts}

### Graph Topology
{graph_data}

### ML Risk Assessment
{ml_data}

### Similar Historical Cases
{similar_cases}

---

## TASK

Based on the above evidence, produce a structured JSON investigation report with EXACTLY these fields:

{{
  "risk_level": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "final_risk_score": <float between 0 and 1>,
  "summary": "<2-3 sentence plain-English summary of the ring and why it's suspicious>",
  "key_evidence": [
    "<evidence point 1>",
    "<evidence point 2>",
    "<evidence point 3>",
    "<evidence point 4 (optional)>",
    "<evidence point 5 (optional)>"
  ],
  "recommended_action": "<specific action: BLOCK_ALL | ESCALATE | MONITOR | REVIEW>",
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}}

Rules:
- risk_level and final_risk_score must be consistent (CRITICAL ≥ 0.80, HIGH ≥ 0.60, MEDIUM ≥ 0.30)
- key_evidence must be specific, citing numbers from the data (e.g. "91% return rate vs 8% baseline")
- recommended_action must be a single clear directive
- Return ONLY the JSON object, no markdown, no code blocks, no explanation
"""


def _fmt(obj: object) -> str:
    if obj is None:
        return "No data available"
    if isinstance(obj, dict):
        return json.dumps(obj, indent=2, default=str)
    if isinstance(obj, list):
        return json.dumps(obj[:5], indent=2, default=str)
    return str(obj)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a string, even if wrapped in prose."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: return a safe degraded report
    return {
        "risk_level":          "HIGH",
        "final_risk_score":    0.75,
        "summary":             "Investigation completed but report parsing failed. Manual review recommended.",
        "key_evidence":        ["Automated parsing error — please review raw evidence"],
        "recommended_action":  "ESCALATE",
        "confidence":          "LOW",
    }


def analyst_node(state: InvestigationState) -> InvestigationState:
    """LangGraph node: synthesize evidence with Groq LLM."""
    cluster_id = state["cluster_id"]
    events = list(state.get("events", []))
    _services = state.get("_services", {})  # type: ignore
    groq_key  = _services.get("groq_api_key", "")
    groq_model = _services.get("groq_model", "llama-3.3-70b-versatile")
    vectors   = _services.get("vectors")

    # -- Emit "reasoning" step -----------------------------------------------
    events.append({
        "step":    "reasoning",
        "message": "Groq is synthesising evidence into an investigation report...",
        "data":    None,
    })

    # -- Build prompt --------------------------------------------------------
    prompt_text = _INVESTIGATION_PROMPT.format(
        facts=_fmt(state.get("facts")),
        graph_data=_fmt(state.get("graph_data")),
        ml_data=_fmt(state.get("ml_data")),
        similar_cases=_fmt(state.get("similar_cases")),
    )

    # -- Call Groq -----------------------------------------------------------
    report = {}
    try:
        llm = ChatGroq(
            api_key=groq_key,
            model=groq_model,
            temperature=0.1,
            max_tokens=1024,
        )
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])
        raw_text = response.content
        log.info(f"  [analyst] Groq response ({len(raw_text)} chars)")
        report = _extract_json(raw_text)
    except Exception as e:
        log.error(f"  [analyst] Groq call failed: {e}")
        report = {
            "risk_level":       "HIGH",
            "final_risk_score": float(
                (state.get("ml_data") or {}).get("avg_ml_score", 0.75)
            ),
            "summary":          f"Automated investigation for cluster {cluster_id}. LLM unavailable — evidence summary only.",
            "key_evidence":     [
                f"Cluster size: {(state.get('facts') or {}).get('cluster_size', 'unknown')}",
                f"Avg return rate: {(state.get('facts') or {}).get('avg_return_rate', 0.0):.1%}",
                f"Shared devices: {(state.get('graph_data') or {}).get('shared_devices', 0)}",
            ],
            "recommended_action": "ESCALATE",
            "confidence":         "LOW",
        }

    # Enrich report with metadata
    report["cluster_id"]    = cluster_id
    report["generated_at"]  = datetime.utcnow().isoformat()
    report["similar_cases"] = state.get("similar_cases") or []

    # -- Store in Qdrant (learning loop) ------------------------------------
    if vectors and report.get("summary"):
        case_id = f"CASE-{cluster_id[:8].upper()}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        try:
            store_investigation(
                case_id=case_id,
                summary=report["summary"],
                metadata={
                    "case_id":          case_id,
                    "ring_type":        (state.get("facts") or {}).get("ring_type", "UNKNOWN"),
                    "risk_score":       report.get("final_risk_score", 0.0),
                    "risk_level":       report.get("risk_level", "HIGH"),
                    "outcome":          report.get("recommended_action", ""),
                    "cluster_size":     (state.get("facts") or {}).get("cluster_size", 0),
                    "avg_return_rate":  (state.get("facts") or {}).get("avg_return_rate", 0.0),
                },
                vectors=vectors,
            )
            log.info(f"  [analyst] stored case: {case_id}")
        except Exception as e:
            log.warning(f"  [analyst] case storage failed: {e}")

    # -- Emit "complete" event -----------------------------------------------
    events.append({
        "step":    "complete",
        "message": "Investigation complete",
        "data":    report,
    })

    return {
        **state,
        "report": report,
        "events": events,
    }
