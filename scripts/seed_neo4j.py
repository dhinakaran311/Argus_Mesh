"""
scripts/seed_neo4j.py
=====================
AbuseRing Sentinel — Neo4j Seeding Entry Point

Thin wrapper that invokes graph/graph_builder.py.
Run this from the project root after all CSVs have been generated.

Usage:
    python scripts/seed_neo4j.py
    python scripts/seed_neo4j.py --nodes-only    # nodes only, skip relationships

Prerequisites:
    pip install neo4j tqdm
    .env must contain NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    data/raw/*.csv must exist (run scripts/generate_data.py first)
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from graph.graph_builder import build_graph


def main():
    parser = argparse.ArgumentParser(
        description="Seed Neo4j Aura with the AbuseRing Sentinel graph data"
    )
    parser.add_argument(
        "--nodes-only",
        action="store_true",
        help="Only create nodes (Customer/Device/IP/Merchant), skip relationships. "
             "Useful for quick connectivity testing.",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  AbuseRing Sentinel — Neo4j Seeding Script")
    print("=" * 60)
    print()

    counts = build_graph(skip_relationships=args.nodes_only)

    print()
    print("=" * 60)
    print("  Final Graph Counts:")
    print("=" * 60)
    for label, count in counts.items():
        print(f"  {label:<20} {count:>10,}")
    print()
    print("  Next step: Day 5 — Supabase Schema + Backend Skeleton")
    print("=" * 60)


if __name__ == "__main__":
    main()
