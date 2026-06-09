"""Model routing cheap-first com fallback.

Classifica queries como simples ou complexas para rotear entre modelos Groq:
- Simples  → llama-3.1-8b-instant  (ultra-rápido, gratuito)
- Complexas → llama-3.3-70b-versatile (mais capaz, ainda gratuito no free tier)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


# Palavras-chave que indicam complexidade (requerem raciocínio mais profundo)
_COMPLEX_KEYWORDS = {
    "explique", "explica", "expliquem",
    "compare", "comparar", "comparação",
    "analise", "analisar", "análise",
    "projete", "projetar",
    "descreva", "descrever",
    "diferença", "diferenças",
    "vantagem", "vantagens", "desvantagem", "desvantagens",
    "quando usar", "como funciona", "por que",
    "demonstre", "mostre", "exemplifique",
    "quais são as", "liste todos", "resuma",
    "passo a passo", "tutorial", "guia",
    "implemente", "criar", "construir",
}


# ------------------------------------------------------------------ TODO 6
def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium).

    Estratégia heurística em 3 camadas:
    1. Query muito curta e direta → simple
    2. Contém palavras-chave de raciocínio → complex
    3. Query longa (>= 80 chars) → complex por padrão
    """
    cheap_model = os.environ.get("CHEAP_MODEL", "llama-3.1-8b-instant")
    premium_model = os.environ.get("PREMIUM_MODEL", "llama-3.3-70b-versatile")

    query_lower = query.lower().strip()
    words = re.findall(r"\w+", query_lower)

    # Short question ending in '?' usually indicates a simple factual lookup
    if len(query) < 60 and query.strip().endswith("?") and len(words) <= 10:
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason=f"Query curta ({len(query)} chars) e direta — modelo rápido suficiente",
        )

    # Check for keywords that suggest complex reasoning
    matched = _COMPLEX_KEYWORDS.intersection(set(words))
    # Also check for common bigrams
    for bigram_keyword in ["como funciona", "por que", "quando usar", "passo a passo", "quais são"]:
        if bigram_keyword in query_lower:
            matched.add(bigram_keyword)

    if matched:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"Palavras-chave complexas detectadas: {', '.join(sorted(matched))}",
        )

    # Long queries typically need more context and reasoning capacity
    if len(query) >= 80:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"Query longa ({len(query)} chars) — modelo premium para melhor qualidade",
        )

    # Default to simple if no complex indicators are found
    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="Query direta sem indicadores de complexidade — modelo rápido",
    )
