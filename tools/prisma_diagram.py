"""tools/prisma_diagram.py — Generate PRISMA flow diagram in Mermaid and DOT formats"""
from __future__ import annotations
from typing import Dict


def generate_prisma_mermaid(flow: Dict[str, int]) -> str:
    identified = flow.get("identified", 0)
    screened = flow.get("screened", 0)
    eligibility = flow.get("eligibility", 0)
    included = flow.get("included", 0)
    excluded_screening = max(0, identified - screened)
    excluded_eligibility = max(0, eligibility - included)
    return "\n".join([
        "flowchart TD",
        f'    A["\U0001f50d Records identified\\n(n = {identified})"]',
        f'    B["\U0001f4cb Records screened\\n(n = {screened})"]',
        f'    C["❌ Excluded at screening\\n(n = {excluded_screening})"]',
        f'    D["\U0001f4c4 Full-text assessed\\n(n = {eligibility})"]',
        f'    E["❌ Full-text excluded\\n(n = {excluded_eligibility})"]',
        f'    F["✅ Studies included\\n(n = {included})"]',
        "    A --> B",
        "    B -->|not excluded| D",
        "    B -->|excluded| C",
        "    D -->|not excluded| F",
        "    D -->|excluded| E",
        "    style A fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f",
        "    style B fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f",
        "    style D fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f",
        "    style C fill:#fee2e2,stroke:#ef4444,color:#7f1d1d",
        "    style E fill:#fee2e2,stroke:#ef4444,color:#7f1d1d",
        "    style F fill:#dcfce7,stroke:#22c55e,color:#14532d",
    ])


def generate_prisma_dot(flow: Dict[str, int]) -> str:
    identified = flow.get("identified", 0)
    screened = flow.get("screened", 0)
    eligibility = flow.get("eligibility", 0)
    included = flow.get("included", 0)
    excluded_screening = max(0, identified - screened)
    excluded_eligibility = max(0, eligibility - included)
    return f"""digraph PRISMA {{
    rankdir=TD;
    node [shape=box, style="rounded,filled", fontsize=11];
    A [label="Records identified\\n(n={identified})", fillcolor="#dbeafe", color="#3b82f6"];
    B [label="Records screened\\n(n={screened})", fillcolor="#dbeafe", color="#3b82f6"];
    C [label="Excluded at screening\\n(n={excluded_screening})", fillcolor="#fee2e2", color="#ef4444"];
    D [label="Full-text assessed\\n(n={eligibility})", fillcolor="#dbeafe", color="#3b82f6"];
    E [label="Full-text excluded\\n(n={excluded_eligibility})", fillcolor="#fee2e2", color="#ef4444"];
    F [label="Studies included\\n(n={included})", fillcolor="#dcfce7", color="#22c55e"];
    A -> B; B -> C [label="excluded"]; B -> D [label="eligible"];
    D -> E [label="excluded"]; D -> F [label="included"];
    {{ rank=same; B; C; }} {{ rank=same; D; E; }}
}}"""
