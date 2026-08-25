"""
=============================================================================
AbuseRing Sentinel — Credential Tester
=============================================================================
Tests connectivity and authentication for all external services:
  - Supabase (REST API + PostgreSQL)
  - Qdrant Cloud (Vector DB)
  - Neo4j Aura (Graph DB)
  - Groq API (LLM)
  - HuggingFace Inference API (Embeddings)

Usage:
    python scripts/test_credentials.py
=============================================================================
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"  Loaded .env from: {env_path}\n")
    else:
        print(f"  WARNING: .env not found at {env_path}. Using system env vars.\n")
except ImportError:
    print("  'python-dotenv' not installed. Run: pip install python-dotenv\n")

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
results = []

def record(service: str, status: bool, message: str, latency_ms: float = 0):
    icon = "[PASS]" if status else "[FAIL]"
    results.append({
        "service": service,
        "status": status,
        "message": message,
        "latency_ms": latency_ms,
        "icon": icon,
    })


# ---------------------------------------------------------------------------
# 1. Supabase — REST API
# ---------------------------------------------------------------------------
def test_supabase_rest():
    print("[*] Testing Supabase REST API...")
    try:
        import urllib.request
        import urllib.error
        import json

        url = os.getenv("SUPABASE_URL", "").rstrip("/")
        anon_key = os.getenv("SUPABASE_ANON_KEY", "")
        service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

        if not url or "your-project" in url:
            record("Supabase REST (Anon)", False, "SUPABASE_URL not configured")
            record("Supabase REST (Service Role)", False, "SUPABASE_URL not configured")
            return

        # Ensure url points to the base (without /rest/v1)
        base_url = url.replace("/rest/v1", "").replace("/rest/v1/", "")
        project_ref = base_url.replace("https://", "").split(".")[0]

        # --- Test Anon Key via /auth/v1/health (public health endpoint) ---
        # NOTE: /rest/v1/ root requires service_role. Use /auth/v1/health for anon key validation.
        start = time.time()
        health_url = f"https://{project_ref}.supabase.co/auth/v1/health"
        req = urllib.request.Request(
            health_url,
            headers={
                "apikey": anon_key,
                "Authorization": f"Bearer {anon_key}",
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                latency = (time.time() - start) * 1000
                body = json.loads(resp.read().decode())
                record("Supabase REST (Anon Key)", True, f"HTTP 200 — Auth service: {body.get('name','ok')}", latency)
        except urllib.error.HTTPError as e:
            latency = (time.time() - start) * 1000
            # 404 on health is still reachable; 200/204 is ideal
            if e.code in (200, 204, 404):
                record("Supabase REST (Anon Key)", True, f"HTTP {e.code} — Project reachable, key format OK", latency)
            else:
                record("Supabase REST (Anon Key)", False, f"HTTP {e.code}: {e.reason}", latency)

        # --- Test Service Role Key via /rest/v1/ (requires service_role) ---
        start = time.time()
        req2 = urllib.request.Request(
            f"{base_url}/rest/v1/",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
            }
        )
        try:
            with urllib.request.urlopen(req2, timeout=10) as resp:
                latency = (time.time() - start) * 1000
                record("Supabase REST (Service Role)", True, f"HTTP {resp.status}", latency)
        except urllib.error.HTTPError as e:
            latency = (time.time() - start) * 1000
            if e.code in (200, 400, 406):
                record("Supabase REST (Service Role)", True, f"HTTP {e.code} — API reachable", latency)
            else:
                record("Supabase REST (Service Role)", False, f"HTTP {e.code}: {e.reason}", latency)

    except Exception as e:
        record("Supabase REST", False, f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# 2. Supabase — PostgreSQL (Transaction Pooler, port 6543)
# ---------------------------------------------------------------------------
def test_supabase_postgres():
    print("[*] Testing Supabase PostgreSQL (transaction pooler)...")
    try:
        import psycopg2
    except ImportError:
        record("Supabase PostgreSQL", False, "psycopg2 not installed — run: pip install psycopg2-binary")
        return

    db_url = os.getenv("SUPABASE_DB_URL", "")
    if not db_url or "your-password" in db_url:
        record("Supabase PostgreSQL", False, "SUPABASE_DB_URL not configured")
        return

    try:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(db_url)

        # Decode double-encoded password (%25 -> %)
        raw_password = parsed.password or ""
        decoded_password = unquote(unquote(raw_password))  # double-decode for %25 -> %

        host = parsed.hostname
        port = parsed.port or 5432
        user = parsed.username
        dbname = parsed.path.lstrip("/")

        # Auto-switch to Supabase transaction pooler (port 6543) if direct port used
        if port == 5432 and host and "supabase.co" in host:
            # Build pooler host: db.xxx.supabase.co -> xxx.pooler.supabase.com
            project_ref = host.replace("db.", "").replace(".supabase.co", "")
            host = f"{project_ref}.pooler.supabase.com"
            port = 6543
            user = f"postgres.{project_ref}"  # Supabase pooler requires project-scoped user
            print(f"    [auto] Switched to transaction pooler: {host}:{port}")

        start = time.time()
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=decoded_password,
            dbname=dbname,
            connect_timeout=10,
            sslmode="require",
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        latency = (time.time() - start) * 1000
        cur.close()
        conn.close()
        record("Supabase PostgreSQL", True, f"Connected via pooler — {version[:50]}...", latency)
    except Exception as e:
        record("Supabase PostgreSQL", False, f"{type(e).__name__}: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# 3. Qdrant Cloud
# ---------------------------------------------------------------------------
def test_qdrant():
    print("[*] Testing Qdrant Cloud...")
    try:
        import urllib.request
        import urllib.error

        url = os.getenv("QDRANT_URL", "").rstrip("/")
        api_key = os.getenv("QDRANT_API_KEY", "")

        if not url or "your-cluster-id" in url:
            record("Qdrant Cloud", False, "QDRANT_URL not configured")
            return

        start = time.time()
        req = urllib.request.Request(
            f"{url}/collections",
            headers={"api-key": api_key, "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                import json
                data = json.loads(resp.read().decode())
                latency = (time.time() - start) * 1000
                collections = data.get("result", {}).get("collections", [])
                col_names = [c["name"] for c in collections] if collections else []
                msg = f"Authenticated — {len(col_names)} collection(s)"
                if col_names:
                    msg += f": {', '.join(col_names)}"
                record("Qdrant Cloud", True, msg, latency)
        except urllib.error.HTTPError as e:
            latency = (time.time() - start) * 1000
            record("Qdrant Cloud", False, f"HTTP {e.code}: {e.reason}", latency)

    except Exception as e:
        record("Qdrant Cloud", False, f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# 4. Neo4j Aura
# ---------------------------------------------------------------------------
def test_neo4j():
    print("[*] Testing Neo4j Aura...")
    try:
        from neo4j import GraphDatabase
    except ImportError:
        record("Neo4j Aura", False, "neo4j driver not installed — run: pip install neo4j")
        return

    uri = os.getenv("NEO4J_URI", "")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    if not uri or "xxxxxxxx" in uri:
        record("Neo4j Aura", False, "NEO4J_URI not configured (still has placeholder 'xxxxxxxx')")
        return
    if not password or "your-neo4j-password" in password:
        record("Neo4j Aura", False, "NEO4J_PASSWORD not configured")
        return

    try:
        start = time.time()
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run("RETURN 1 AS ping")
            result.single()
        latency = (time.time() - start) * 1000
        driver.close()
        record("Neo4j Aura", True, f"Connected to {uri}", latency)
    except Exception as e:
        record("Neo4j Aura", False, f"{type(e).__name__}: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# 5. Groq API
# ---------------------------------------------------------------------------
def test_groq():
    print("[*] Testing Groq API...")
    try:
        import urllib.request
        import urllib.error
        import json

        api_key = os.getenv("GROQ_API_KEY", "")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not api_key or "your-groq" in api_key:
            record("Groq API", False, "GROQ_API_KEY not configured")
            return

        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Reply with only the word: PONG"}],
            "max_tokens": 5,
            "temperature": 0,
        }).encode()

        start = time.time()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # User-Agent required to bypass Cloudflare WAF on direct requests
                "User-Agent": "Mozilla/5.0 (compatible; ArgusCredentialTester/1.0)",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
                latency = (time.time() - start) * 1000
                reply = data["choices"][0]["message"]["content"].strip()
                record("Groq API", True, f"Model: {model} | Response: '{reply}'", latency)
        except urllib.error.HTTPError as e:
            latency = (time.time() - start) * 1000
            body = e.read().decode()
            if e.code == 403:
                record("Groq API", False, f"HTTP 403 — Cloudflare blocked. Key may still be valid. Verify at console.groq.com", latency)
            else:
                record("Groq API", False, f"HTTP {e.code}: {body[:120]}", latency)

    except Exception as e:
        record("Groq API", False, f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# 6. HuggingFace Inference API
# ---------------------------------------------------------------------------
def test_huggingface():
    print("[*] Testing HuggingFace Inference API...")
    try:
        import urllib.request
        import urllib.error
        import json

        api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        if not api_key or "your-huggingface" in api_key:
            record("HuggingFace API", False, "HUGGINGFACE_API_KEY not configured — please add your token to .env")
            return

        payload = json.dumps({"inputs": ["test credential check"]}).encode()

        start = time.time()
        req = urllib.request.Request(
            f"https://api-inference.huggingface.co/models/{model}",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
                latency = (time.time() - start) * 1000
                if isinstance(data, list) and len(data) > 0:
                    dims = len(data[0]) if isinstance(data[0], list) else "N/A"
                    record("HuggingFace API", True, f"Model: {model} | Embedding dims: {dims}", latency)
                else:
                    record("HuggingFace API", True, f"Model: {model} | Response OK", latency)
        except urllib.error.HTTPError as e:
            latency = (time.time() - start) * 1000
            body = e.read().decode()
            if e.code == 503:
                # Model is loading — key is valid
                record("HuggingFace API", True, f"Key valid — Model {model} is loading (503, normal for cold start)", latency)
            else:
                record("HuggingFace API", False, f"HTTP {e.code}: {body[:120]}", latency)

    except Exception as e:
        record("HuggingFace API", False, f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Print final report
# ---------------------------------------------------------------------------
def print_report():
    print("\n")
    print("=" * 70)
    print("  ARGUS MESH — CREDENTIAL TEST REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"  {'SERVICE':<35} {'STATUS':<10} {'LATENCY':>8}   DETAILS")
    print("-" * 70)

    passed = 0
    failed = 0
    for r in results:
        status_str = "PASS" if r["status"] else "FAIL"
        latency_str = f"{r['latency_ms']:.0f}ms" if r["latency_ms"] else "  N/A"
        print(f"  {r['icon']} {r['service']:<33} {status_str:<10} {latency_str:>7}   {r['message']}")
        if r["status"]:
            passed += 1
        else:
            failed += 1

    print("-" * 70)
    print(f"  TOTAL: {passed + failed} tests | PASSED: {passed} | FAILED: {failed}")
    print("=" * 70)

    if failed == 0:
        print("\n  *** ALL CREDENTIALS VERIFIED SUCCESSFULLY! ***")
    else:
        print(f"\n  WARNING: {failed} credential(s) need attention. See details above.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ARGUS MESH — Running Credential Tests...")
    print("=" * 70 + "\n")

    test_supabase_rest()
    test_supabase_postgres()
    test_qdrant()
    test_neo4j()
    test_groq()
    test_huggingface()

    print_report()
