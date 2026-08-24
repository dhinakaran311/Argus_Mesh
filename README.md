# AbuseRing Sentinel 🛡️

> **Coordinated payment-abuse ring detection for the Razorpay ecosystem**  
> ML-powered fraud intelligence with graph analytics, LLM-assisted investigation, and a real-time risk dashboard.

---

## Overview

AbuseRing Sentinel detects **coordinated fraud rings** — groups of customers who work together to exploit payment systems through refund fraud, promo abuse, and return manipulation. Unlike single-account fraud systems, this platform identifies the **network structure** of abuse: shared devices, shared IPs, synchronised account creation, and abnormal transaction velocity across the ring.

### Key Capabilities

| Capability | Description |
|---|---|
| 🔍 **Ring Detection** | Graph-based clustering finds connected customer networks sharing devices / IPs |
| 🤖 **ML Risk Scoring** | XGBoost model trained on 25+ behavioural features with temporal validation |
| 🕸️ **Graph Analytics** | Neo4j stores the customer–device–IP–transaction graph for Cypher-based queries |
| 🧠 **AI Investigation** | Groq LLM agent answers natural-language questions about suspected rings |
| 📊 **Live Dashboard** | Next.js frontend with ring explorer, risk gauges, and network visualisations |
| 🗄️ **Vector Search** | Qdrant stores case embeddings for similarity-based investigation retrieval |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Next.js 14 Frontend                │
│   Dashboard · Ring Explorer · AI Investigation Chat  │
└────────────────────────┬────────────────────────────┘
                         │ REST / SSE
┌────────────────────────▼────────────────────────────┐
│                  FastAPI Backend                      │
│  /customers  /rings  /agent  /health                 │
│                                                      │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────┐  │
│  │ Risk Service │  │ Graph Service │  │  Agent   │  │
│  │  (XGBoost)   │  │  (Neo4j)      │  │  (Groq)  │  │
│  └──────┬───────┘  └───────┬───────┘  └────┬─────┘  │
└─────────┼──────────────────┼───────────────┼────────┘
          │                  │               │
   ┌──────▼──────┐   ┌───────▼──────┐  ┌────▼──────┐
   │  Supabase   │   │  Neo4j Aura  │  │  Qdrant   │
   │ (PostgreSQL)│   │   (Graph DB) │  │ (Vectors) │
   └─────────────┘   └──────────────┘  └───────────┘
```

**Tech Stack:**

| Layer | Technology |
|---|---|
| Backend | Python 3.11 · FastAPI · Uvicorn |
| ML | XGBoost · scikit-learn · pandas · numpy |
| Relational DB | Supabase (PostgreSQL) |
| Graph DB | Neo4j Aura |
| Vector Store | Qdrant Cloud |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Frontend | Next.js 14 · TypeScript |
| Infrastructure | Docker Compose |

---

## Dataset

The project ships with a **synthetic dataset generator** that produces a realistic Indian payment ecosystem with injected coordinated abuse rings.

### Generated Files (`data/raw/`)

| File | Description | Scale (sample) | Scale (full) |
|---|---|---|---|
| `customers.csv` | Customer profiles | 1,000 | 10,000 |
| `transactions.csv` | Payment transactions (Jan–Nov 2024) | ~18,670 | ~200,000 |
| `orders.csv` | Orders from successful transactions | ~12,400 | ~150,000 |
| `returns.csv` | Return / refund events | ~2,300 | ~22,000 |
| `devices.csv` | Device fingerprints | 350 | 3,500 |
| `ips.csv` | IP address metadata | 420 | 4,200 |
| `merchants.csv` | Merchant profiles | 11 | 51 |
| `abuse_labels.csv` | Ground truth (one row per customer) | 1,000 | 10,000 |

### Design Principles

- **Realistic overlaps**: Legitimate *families* also share devices (2–4 accounts). A single shared-device signal is **not** sufficient to flag a ring — the model must combine multiple signals.
- **Temporal split**: Jan–Sep → train, Oct → val, Nov → held-out test.
- **Abuse prevalence**: ~7.4% ring members, 3 ring types (`REFUND_RING`, `PROMO_ABUSE`, `RETURN_FRAUD`).
- **Sharp behavioral signals**: Ring return rate ~88% vs. normal ~9.5%.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (for full stack)
- Cloud credentials (see `.env.example`)

### 1 — Generate the Dataset

```bash
# Install script dependencies
pip install -r requirements-scripts.txt

# Generate sample dataset (~30 seconds)
python scripts/generate_data.py --scale sample --output data/raw

# Validate the output
python scripts/validate_data.py --input data/raw
```

**Expected output:**
```
Validation: ✅  ALL CHECKS PASSED
```

For the full 10,000-customer dataset:
```bash
python scripts/generate_data.py --scale full --output data/raw
```

### 2 — Configure Environment

```bash
cp .env.example .env
# Fill in your Supabase, Neo4j, Qdrant, Groq, and HuggingFace credentials
```

### 3 — Run the Stack

```bash
docker compose up
```

| Service | URL |
|---|---|
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Frontend | http://localhost:3000 |

---

## Project Structure

```
abusering-sentinel/
├── scripts/
│   ├── generate_data.py        # Synthetic dataset generator
│   └── validate_data.py        # Dataset quality validator
├── data/
│   ├── raw/                    # Generated CSVs (git-ignored)
│   ├── processed/              # Feature-engineered parquet files
│   └── sample/                 # Small sample kept in git
├── ml/
│   ├── features/               # Feature engineering
│   ├── training/               # XGBoost training pipeline
│   ├── evaluation/             # Metrics and cost analysis
│   ├── models/                 # Predictor class + artifacts
│   └── notebooks/              # EDA notebooks
├── backend/
│   └── app/
│       ├── api/                # FastAPI route handlers
│       ├── services/           # Business logic layer
│       ├── agents/             # LLM investigation agent
│       ├── models/             # Pydantic schemas
│       └── db/                 # DB client wrappers
├── graph/
│   ├── schema/                 # Neo4j Cypher schema + constraints
│   └── queries/                # Ring detection Cypher queries
├── frontend/                   # Next.js 14 dashboard
├── docs/                       # Architecture docs
├── docker-compose.yml
├── .env.example
└── requirements-scripts.txt
```

---

## Development Roadmap

| Phase | Status | Description |
|---|---|---|
| **Day 1** — Data Layer | ✅ **Complete** | Dataset generator, validator, project scaffold |
| **Day 2** — ML Pipeline | 🔄 In Progress | Feature engineering, XGBoost training, evaluation |
| **Day 3** — Backend | ⬜ Planned | FastAPI REST API, DB integrations, risk service |
| **Day 4** — Graph Layer | ⬜ Planned | Neo4j schema, Cypher queries, ring detection |
| **Day 5** — AI Agent | ⬜ Planned | Groq LLM agent with tool-calling for investigation |
| **Day 6** — Frontend | ⬜ Planned | Next.js dashboard, ring explorer, network graph |
| **Day 7** — Polish | ⬜ Planned | Tests, docs, performance tuning |

---

## Data Privacy

All data in this repository is **100% synthetic**. No real customer, transaction, or payment data is used at any stage. The generator uses a fixed random seed (`42`) for full reproducibility.

---

## License

MIT © 2024 — Built as a Razorpay fraud intelligence research project.
