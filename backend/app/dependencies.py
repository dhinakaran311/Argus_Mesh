"""
backend/app/dependencies.py
=============================
AbuseRing Sentinel — FastAPI Dependency Injection

All singleton services are stored on app.state during startup.
Route handlers access them via request.app.state.

This module provides FastAPI Depends() helpers for routes that
prefer explicit injection over request.app.state access.
"""
from __future__ import annotations

from fastapi import Request

from .config import Settings, get_settings
from .db.neo4j import Neo4jClient
from .db.qdrant import QdrantDB
from .db.supabase import DataStore
from .services.ml_service import MLService
from .services.graph_service import GraphService
from .services.vector_service import VectorService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_neo4j(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


def get_qdrant(request: Request) -> QdrantDB:
    return request.app.state.qdrant


def get_data(request: Request) -> DataStore:
    return request.app.state.data


def get_ml(request: Request) -> MLService:
    return request.app.state.ml


def get_graph_service(request: Request) -> GraphService:
    return request.app.state.graph


def get_vector_service(request: Request) -> VectorService:
    return request.app.state.vectors
