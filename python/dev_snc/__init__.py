"""Developmental Multicenter SNC (v1).

An embodied, multicenter learning layer on top of the SNC substrate: specialized
centers connected by adaptive, structurally-plastic pathways, evaluated in a small
digital nursery. See docs/developmental-multicenter-snc.md (plan) and
docs/nursery-v1.md (this prototype).
"""
from .agent import AgentConfig, DevelopmentalAgent
from .centers import Center, MulticenterGraph, Pathway
from .nursery import Nursery, Obj
from .tasks import run_forgetting, run_naming

__all__ = [
    "AgentConfig", "DevelopmentalAgent",
    "Center", "MulticenterGraph", "Pathway",
    "Nursery", "Obj",
    "run_naming", "run_forgetting",
]
