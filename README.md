# Argus Mesh 🛡️

> **Coordinated payment-abuse ring detection for the Razorpay ecosystem**  
> XGBoost ML · Neo4j graph intelligence · Groq LLM investigation agents · Real-time SSE dashboard

---

## What Is This?

Argus Mesh detects **coordinated fraud rings** — groups of customers acting in concert to exploit payment systems through return fraud, promo abuse, and refund manipulation. Unlike single-account fraud detectors, it identifies the **network structure** of abuse: clusters of accounts sharing devices and IP addresses, created in tight time windows, with abnormally synchronised transaction behaviour.

The system combines three independent signals into a single risk verdict:

| Signal | Source | Contribution |
|--------|--------|-------------|
| **ML behavioural score** | XGBoost (41 features, SHAP explanations) | 40% |
| **Graph topology score** | Neo4j (device/IP sharing, cluster size, burst) | 30% |
| **Return rate signal** | Raw transaction data | 30% |

When a ring is flagged, a LangGraph + Groq LLM agent automatically synthesises all three signals into a structured investigation report — with a risk verdict, key evidence bullets, and a recommended action (`BLOCK_ALL`, `ESCALATE`, `MONITOR`, `REVIEW`).

---

## Live Demo

| Service | URL |
|---------|-----|
| Frontend dashboard | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger / OpenAPI | http://localhost:8000/docs |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Next.js 16 Frontend                          │
│  Dashboard · Rings Ledger · Case File · AI Investigation · Graph │
└───────────────────────────┬──────────────────────────────────────┘
                            │  REST + Server-Sent Events (SSE)
┌───────────────────────────▼──────────────────────────────────────┐
│                       FastAPI Backend                            │
│                                                                  │
│  /api/dashboard   /api/clusters   /api/graph/{id}               │
│  /api/investigate (SSE)   /api/rag   /api/model   /api/health   │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  ML Service │  │ Graph Service│  │  LangGraph Pipeline   │  │
│  │  XGBoost    │  │  Neo4j       │  │  investigator_node    │  │
│  │  SHAP       │  │  Cypher      │  │  retrieval_node (RAG) │  │
│  │  DataStore  │  │  queries     │  │  analyst_node (Groq)  │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬────────────┘  │
└─────────┼────────────────┼─────────────────────┼───────────────┘
          │                │                      │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌───────────▼────────┐
   │  DataStore  │  │ Neo4j Aura  │  │    Qdrant Cloud    │
   │  (CSV RAM)  │  │  Graph DB   │  │  (Vector / RAG)    │
   └─────────────┘  └─────────────┘  └────────────────────┘
                                              │
                                     ┌────────▼───────┐
                                     │   Groq Cloud   │
                                     │ compound-beta  │
                                     └────────────────┘
```

---

## Tech Stack

### Backend
| Component | Technology | Version |
|-----------|-----------|---------|
| Web framework | FastAPI + Uvicorn | 0.115.0 |
| Streaming | sse-starlette (SSE) | 2.1.3 |
| ML | XGBoost + scikit-learn + SHAP | 2.0.3 / 1.5.1 / 0.45.1 |
| Data | pandas + numpy + pyarrow | 2.2.2 / 1.26.4 / 16.1.0 |
| Graph DB | Neo4j Python driver | 5.21.0 |
| Vector DB | Qdrant client | 1.10.1 |
| LLM framework | LangChain + LangGraph | 0.2.14 / 0.2.4 |
| LLM provider | langchain-groq (compound-beta) | 0.1.9 |
| Config | pydantic-settings | 2.3.4 |
| Rate limiting | slowapi | 0.1.10 |

### Frontend
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Next.js (App Router) | 16.3.4 |
| Language | TypeScript | 5.x |
| Graph viz | React Flow | 11.11.4 |
| Charts | Recharts | 3.10.1 |
| State | Zustand | 5.0.15 |
| Icons | Lucide React | 1.39.0 |
| Styling | Tailwind CSS v4 + CSS custom properties | 4.x |

### Cloud Services
| Service | Provider | Role |
|---------|----------|------|
| Graph database | Neo4j AuraDB (free tier) | Customer–device–IP–merchant graph |
| Vector store | Qdrant Cloud | Case embeddings for RAG |
| LLM inference | Groq (compound-beta) | Investigation report generation |
| Embeddings | HuggingFace Inference API | `all-MiniLM-L6-v2` (384-dim) |

---

## Dataset

The project ships with a **fully synthetic** dataset simulating 10,000 customers in an Indian payment ecosystem, with injected coordinated abuse rings.

### Raw Files (`data/raw/`)

| File | Rows | Description |
|------|------|-------------|
| `customers.csv` | 10,000 | Customer profiles — city, email domain, account creation timestamp, risk_score, return_rate |
| `transactions.csv` | ~200,000 | Payment events (Jan–Nov 2024) sorted by timestamp DESC |
| `orders.csv` | ~159,000 | Orders from successful transactions |
| `returns.csv` | ~41,500 | Return / refund events |
| `devices.csv` | 3,500 | Device fingerprints with `accounts_count` |
| `ips.csv` | 4,200 | IP metadata — city, ISP |
| `merchants.csv` | 11 | Merchant profiles |
| `abuse_labels.csv` | 10,000 | Ground truth — `is_abuse`, `cluster_id`, `ring_type` |

### Ring Types

| Type | Description | Typical return rate |
|------|-------------|-------------------|
| `RETURN_FRAUD` | Abuse of return/refund policies at scale | ~88–91% |
| `REFUND_RING` | Coordinated refund claims before delivery | ~85% |
| `PROMO_ABUSE` | Multi-account exploitation of promo codes | variable |

### Design Principles

- **Realistic legitimate sharing**: Families legitimately share devices (2–4 accounts). A shared device alone does not flag a ring — the model must combine multiple signals.
- **Temporal train/val/test split**: Jan–Sep 2024 → train, Oct → val, Nov → held-out test.
- **Abuse prevalence**: ~7.4% ring members across the full dataset.
- **Sharp behavioural signal**: Ring members average ~88% return rate vs ~9.5% for normal customers.
- **Reproducible**: Fixed seed `42` — all outputs are deterministic.

---

## ML Pipeline

### Feature Engineering (41 features)

Features are computed per customer from their full transaction/order/return history:

| Category | Features |
|----------|---------|
| **Volume** | `num_transactions`, `num_orders`, `num_returns`, `total_gmv` |
| **Return behaviour** | `return_rate`, `avg_days_to_return`, `return_velocity_7d` |
| **Temporal** | `account_age_days`, `days_since_last_txn`, `burst_creation_flag` |
| **Network** | `shared_device_count`, `shared_ip_count`, `co_cluster_size` |
| **Risk signals** | `high_value_return_ratio`, `cross_merchant_abuse_flag`, `promo_usage_rate` |

### Model

- **Algorithm**: XGBoost gradient boosting classifier
- **Threshold**: 0.68 (optimised for asymmetric cost: FP cost = ₹100, FN cost = ₹3,000)
- **Explainability**: SHAP values computed per prediction — top features surfaced in the investigation report
- **Artifacts**: Saved as `.pkl`, loaded at startup into `MLService`

### Scoring in the API

Every investigation call runs `ml.predict_batch()` across all ring members and `ml.explain()` per member to get per-feature SHAP importance, which is aggregated across the cluster and included in the Groq prompt.

---

## Investigation Pipeline (LangGraph)

When `POST /api/investigate` is called with a `cluster_id`, a **LangGraph state machine** streams 7 SSE events to the frontend:

```
starting → facts → graph → ml → rag → reasoning → complete
```

### Nodes

#### 1. `investigator_node`
Runs three tool calls in sequence:

- **`get_cluster_facts`** — Looks up cluster members from Neo4j (UUID device IDs) or DataStore (RING-XXX labels), then scores each member with XGBoost. Returns cluster size, abuse count, return rates, ring type, member locations.
- **`get_graph_topology`** — Queries Neo4j `_ENTITY_GRAPH_QUERY` to get shared devices, IPs, merchants, and per-member connections.
- **`get_ml_explanation`** — Runs `predict_batch` + SHAP across all members. Returns top-5 most important features globally across the cluster.

#### 2. `retrieval_node` (RAG)
Builds a natural-language summary of the cluster evidence and embeds it via HuggingFace `all-MiniLM-L6-v2`. Searches Qdrant for the top-3 most similar historical investigation cases. Retrieved cases are injected into the Groq prompt as context.

#### 3. `analyst_node`
Calls **Groq `compound-beta`** with a structured prompt containing all gathered evidence. Parses the JSON response (handles compound-beta's reasoning preamble) into a typed investigation report:

```json
{
  "risk_level": "CRITICAL",
  "final_risk_score": 0.9987,
  "summary": "A coordinated promo abuse ring of 78 accounts...",
  "key_evidence": ["78/78 members confirmed abuse (100%)", "..."],
  "recommended_action": "BLOCK_ALL",
  "confidence": "HIGH"
}
```

After generation, the case is stored back into Qdrant for future RAG retrieval (learning loop).

### Critical Implementation Detail

`_services` (containing `neo4j`, `vectors`, `groq_api_key`, `groq_model`, `data`, `ml`) **must be declared in `InvestigationState`** (the LangGraph TypedDict). LangGraph strips any key not declared in the schema when passing state between nodes — if omitted, all services are `None` in every node, causing the LLM fallback to fire.

---

## API Reference

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Service health check — Neo4j, Qdrant, ML model, DataStore |
| `GET` | `/api/dashboard` | Aggregate stats — total customers, abuse rate, ring count, risk distribution |
| `GET` | `/api/clusters?limit=50` | Top rings ranked by combined risk score |
| `GET` | `/api/clusters/{id}` | Full cluster detail — members, stats, risk breakdown |
| `GET` | `/api/graph/{id}` | React Flow graph data — nodes (customer/device/IP) + edges |
| `POST` | `/api/investigate` | **SSE stream** — AI investigation of a ring cluster |
| `GET` | `/api/transactions` | Recent transactions with pagination |
| `GET` | `/api/model/metrics` | XGBoost evaluation metrics |
| `GET` | `/api/model/features` | SHAP global feature importance |
| `GET` | `/api/rag/search` | Semantic search over past investigations |

### Investigation SSE Stream

```bash
curl -X POST http://localhost:8000/api/investigate \
  -H "Content-Type: application/json" \
  -d '{"cluster_id": "4e7ccaff-419d-44dc-8f07-97fd8cfe9fd2"}'
```

Response is `text/event-stream`. Each `data:` line is a JSON object:

```
data: {"step": "starting",   "message": "Initialising...",     "data": null}
data: {"step": "facts",      "message": "Retrieved facts...",   "data": {...}}
data: {"step": "graph",      "message": "Graph traversal...",   "data": {...}}
data: {"step": "ml",         "message": "ML scoring...",        "data": {...}}
data: {"step": "rag",        "message": "Found N similar...",   "data": {...}}
data: {"step": "reasoning",  "message": "Groq synthesising...", "data": null}
data: {"step": "complete",   "message": "Investigation done",   "data": {report}}
```

### Authentication

In production (`ENVIRONMENT != development`), all endpoints requiring LLM inference are protected by `X-API-Key` header matching `SECRET_KEY` in `.env`. Rate limited to **5 requests per minute per IP** via `slowapi`.

---

## Frontend Pages

| Route | Description |
|-------|-------------|
| `/` | Homepage / entry point |
| `/rings` | Rings ledger — all clusters ranked by combined risk, with skeleton loader |
| `/rings/[id]` | Case file — cluster stats, member list, React Flow entity graph (lazy loaded) |
| `/investigate` | AI investigation — SSE-streamed live case log + structured report |
| `/graph` | Full-screen network graph explorer |
| `/transactions` | Transaction ledger with risk scores |
| `/model` | ML model metrics, feature importance chart |

### Loading States

All pages with Neo4j/LLM latency show polished loading UIs:
- **Rings list** (`loading.tsx`): 12 shimmer skeleton table rows while server fetches clusters
- **Ring detail**: Full skeleton with shimmer stats rows + pulse-dots "Fetching case data…" in the card; graph area shows "Building connection graph…" then transitions to the React Flow canvas
- **Investigation**: Live SSE step-by-step log with ✓ done / ● working / ○ pending states

---

## Project Structure

```
Argus_Mesh/
├── .env.example                    # All required environment variables
├── .env                            # Your actual credentials (git-ignored)
├── docker-compose.yml              # Backend + frontend Docker services
│
├── backend/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── app/
│       ├── main.py                 # FastAPI app + lifespan (startup/shutdown)
│       ├── config.py               # pydantic-settings — all env vars
│       ├── agents/
│       │   ├── state.py            # InvestigationState TypedDict (MUST include _services)
│       │   ├── orchestrator.py     # LangGraph graph: investigator → retrieval → analyst
│       │   ├── investigator.py     # Node 1: facts + graph + ML evidence gathering
│       │   ├── retrieval.py        # Node 2: Qdrant RAG — similar case search
│       │   ├── analyst.py          # Node 3: Groq LLM synthesis → structured report
│       │   └── tools.py            # Tool functions: get_cluster_facts, get_graph_topology,
│       │                           #   get_ml_explanation, search_similar_cases, store_investigation
│       ├── api/
│       │   ├── clusters.py         # GET /clusters, GET /clusters/{id}, GET /graph/{id}
│       │   ├── investigate.py      # POST /investigate (SSE, rate-limited, auth-gated)
│       │   ├── dashboard.py        # GET /dashboard
│       │   ├── transactions.py     # GET /transactions
│       │   ├── model.py            # GET /model/metrics, GET /model/features
│       │   ├── rag.py              # GET /rag/search
│       │   └── health.py           # GET /health
│       ├── db/
│       │   ├── neo4j.py            # Neo4j driver wrapper + all Cypher queries
│       │   ├── qdrant.py           # Qdrant client wrapper
│       │   ├── datastore.py        # In-memory CSV store (pandas DataFrames + O(1) indexes)
│       │   └── supabase.py         # Reserved (future migration)
│       ├── models/
│       │   ├── cluster.py          # ClusterSummary, ClusterDetail, ReactFlowGraph
│       │   ├── customer.py         # CustomerSummary
│       │   ├── investigation.py    # InvestigationRequest
│       │   └── risk.py             # Risk level enums
│       └── services/
│           ├── graph_service.py    # GraphService — cluster list, detail, React Flow builder
│           ├── ml_service.py       # MLService — XGBoost load, predict, predict_batch, explain
│           ├── vector_service.py   # VectorService — Qdrant setup, find_similar, store
│           ├── embed_service.py    # EmbedService — HuggingFace text embeddings
│           └── risk_engine.py      # RiskEngine — score → risk level thresholds
│
├── frontend/
│   ├── next.config.ts              # /api/* → http://localhost:8000/api/* proxy rewrite
│   ├── app/
│   │   ├── globals.css             # Design system: "The Evidence Wall" dark theme
│   │   │                           #   includes shimmer skeleton + pulse-dots animations
│   │   ├── layout.tsx              # Root layout — sidebar navigation
│   │   ├── rings/
│   │   │   ├── page.tsx            # Server component — cluster list table
│   │   │   ├── loading.tsx         # Next.js loading.tsx — shimmer skeleton rows
│   │   │   └── [id]/
│   │   │       └── page.tsx        # Client component — case file + lazy graph
│   │   ├── investigate/
│   │   │   └── page.tsx            # SSE investigation stream + report card
│   │   ├── graph/page.tsx
│   │   ├── transactions/page.tsx
│   │   └── model/page.tsx
│   ├── components/
│   │   ├── CorkboardGraph.tsx      # React Flow canvas — customer/device/IP nodes
│   │   ├── RiskStamp.tsx           # Risk level badge (CRITICAL/HIGH/MEDIUM/LOW)
│   │   └── layout/Sidebar.tsx      # Navigation sidebar
│   └── lib/
│       ├── api.ts                  # Typed fetch wrappers — all backend endpoints
│       └── sse.ts                  # SSE client — streamInvestigation()
│
└── data/
    ├── raw/                        # CSVs (customers, transactions, devices, etc.)
    └── processed/                  # features.parquet + feature_names.json
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Accounts for: [Neo4j AuraDB](https://neo4j.com/cloud/aura/), [Qdrant Cloud](https://cloud.qdrant.io/), [Groq](https://console.groq.com/), [HuggingFace](https://huggingface.co/)

### 1 — Clone and configure

```bash
git clone https://github.com/dhinakaran311/Argus_Mesh.git
cd Argus_Mesh
cp .env.example .env
# Edit .env with your credentials
```

### 2 — Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3 — Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4 — Docker (full stack)

```bash
docker compose up
```

---

## Environment Variables

See `.env.example` for the full list. Required variables:

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key — get from console.groq.com |
| `GROQ_MODEL` | Model ID, e.g. `compound-beta` |
| `NEO4J_URI` | Neo4j AuraDB connection URI (`neo4j+s://...`) |
| `NEO4J_USERNAME` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `HUGGINGFACE_API_KEY` | HuggingFace API key (for embedding inference) |
| `SECRET_KEY` | API key for `/investigate` endpoint (set `changeme` to skip auth in dev) |
| `ENVIRONMENT` | `development` or `production` |

---

## Key Design Decisions

### Why Neo4j for the graph?
The customer–device–IP relationship is a natural graph. Cypher queries like "find all customers sharing a device with at least 10 accounts" are trivial in Cypher and painful in SQL. AuraDB free tier is sufficient for 10K customers.

### Why in-memory DataStore instead of Supabase live queries?
All CSVs fit comfortably in RAM (~200MB for 200K transactions). Serving 200,000 transactions from pandas in-memory with pre-built O(1) dict indexes is ~100× faster than round-tripping to Supabase for every request. Supabase is reserved for a future real-data migration.

### Why LangGraph over a raw LLM call?
The investigation is a multi-step pipeline with tool calls, RAG retrieval, and state propagation. LangGraph's node/edge model makes the flow explicit and testable. Each node receives and returns the full `InvestigationState`, which is the clean contract for passing context between steps.

### Why SSE instead of WebSockets?
The investigation is a one-directional stream (server → client). SSE is simpler, works over plain HTTP/1.1, and is natively supported by `EventSource` in browsers. No need for WebSocket handshake overhead.

### Why compound-beta and not llama-3.3-70b-versatile?
`compound-beta` is Groq's compound AI system that uses Llama 4 Scout for reasoning + Llama 3.3 70B for routing. It's available on this account's free tier. `llama-3.3-70b-versatile` is not available — using it causes a 404 from Groq's API and triggers the LLM fallback.

---

## Known Limitations

- **Neo4j AuraDB latency**: Free-tier AuraDB is hosted remotely. Each Cypher query takes 2–5 seconds from India. The investigation pipeline requires 2 Neo4j queries per run. Loading UIs mask this delay.
- **HuggingFace Inference API**: The free-tier inference endpoint can be slow or rate-limited. Embedding failures fall back to a zero vector, so RAG returns no similar cases.
- **Graph data for some clusters**: Not all UUID device IDs returned by the top-clusters query have entity graph data in Neo4j (depends on which devices were seeded). These return `{nodes:[], edges:[]}` gracefully.
- **Qdrant cold start**: RAG returns 0 results until the first few investigations are stored in Qdrant's collection.

---

## Data Privacy

All data in this repository is **100% synthetic**. Generated with `seed=42` using randomised Indian names, cities, email domains, and behavioural patterns. No real customer, transaction, or payment data is used anywhere in the project.

---

## License

MIT © 2024 — Built as a Razorpay fraud intelligence research project.
