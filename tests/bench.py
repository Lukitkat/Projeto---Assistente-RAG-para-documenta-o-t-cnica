"""Script de benchmark para medir custo e latência do pipeline RAG.

Executa 50 queries e mede impacto de cada estratégia de otimização.
Uso: python tests/bench.py

Requisitos: .env configurado, corpus indexado.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Adiciona root ao path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.cache import ExactCache, SemanticCache
from src.pipeline.rag import build_rag_pipeline
from src.pipeline.routing import classify_complexity

# 50 queries de benchmark sobre documentação Python
BENCH_QUERIES = [
    # Simples (factual)
    "O que é Python?",
    "O que é uma lista?",
    "O que é uma tupla?",
    "O que é um dicionário?",
    "O que é um conjunto (set)?",
    "O que é uma string?",
    "O que é um inteiro?",
    "O que é um float?",
    "O que é None?",
    "O que é True e False?",
    "O que é uma variável?",
    "O que é um comentário em Python?",
    "O que é um módulo?",
    "O que é um pacote?",
    "O que é pip?",
    # Médias
    "Como criar uma função em Python?",
    "Como usar if e else?",
    "Como fazer um loop for?",
    "Como fazer um loop while?",
    "Como importar um módulo?",
    "Como abrir um arquivo em Python?",
    "Como tratar exceções com try/except?",
    "Como usar list comprehension?",
    "Como usar dict comprehension?",
    "Como criar uma classe?",
    # Complexas
    "Explique como funcionam os decorators em Python",
    "Compare list comprehension com for loop tradicional",
    "Como funciona herança múltipla em Python?",
    "Explique o conceito de generators e quando usá-los",
    "Compare map(), filter() e list comprehension",
    "Explique o modelo de dados do Python e a diferença entre identidade e igualdade",
    "Como funciona o gerenciamento de memória em Python?",
    "Explique closures e funções de primeira classe",
    "Compare métodos de classe, estáticos e de instância",
    "Descreva o protocolo de iteração em Python",
    # Paráfrases (para testar semantic cache)
    "Para que serve Python?",  # paráfrase de "O que é Python?"
    "O que são listas em Python?",  # paráfrase de "O que é uma lista?"
    "Como definir funções?",  # paráfrase de "Como criar uma função em Python?"
    "Explique decoradores Python",  # paráfrase de "Explique como funcionam os decorators"
    "O que são generators?",  # paráfrase de "Explique o conceito de generators"
    # Mistas
    "Como usar *args e **kwargs?",
    "O que é lambda em Python?",
    "Como funciona enumerate()?",
    "O que é zip() e como usar?",
    "Como usar sorted() com key?",
    "O que é slice notation?",
    "Como funciona o operador := (walrus)?",
    "O que são type hints?",
    "Como usar f-strings?",
    "O que é o módulo pathlib?",
    "O que é o módulo pathlib?",
]


def _call_with_retry(pipeline, query: str, max_retries: int = 5) -> tuple[dict, float]:
    """Chama pipeline.answer com tratamento de RateLimitError para não falhar o benchmark.
    Retorna (resultado, tempo_em_ms). O tempo medido não inclui os sleeps.
    """
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            res = pipeline.answer(query)
            t_ms = (time.perf_counter() - t0) * 1000
            return res, t_ms
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str:
                print(f"[Rate limit! Sleep 10s...]", end=" ", flush=True)
                time.sleep(10)
            else:
                raise
    raise RuntimeError("Max retries exceeded for rate limit.")


def run_benchmark():
    """Executa benchmark completo e imprime tabela de resultados."""
    print("=" * 60)
    print("BENCHMARK: PyDoc Assistant RAG Pipeline")
    print("=" * 60)
    print(f"Queries: {len(BENCH_QUERIES)}")
    print()

    # Inicializar
    print("Inicializando pipeline...")
    t0 = time.perf_counter()
    pipeline = build_rag_pipeline()
    print(f"Pipeline pronto em {(time.perf_counter()-t0)*1000:.0f}ms")
    print(f"Chunks indexados: {pipeline.collection.count()}")
    print()

    exact_cache = ExactCache()
    semantic_cache = SemanticCache(threshold=0.93)

    # Estratégia 1: Baseline (sem cache, sem routing — sempre premium)
    import os
    premium_model = os.environ.get("PREMIUM_MODEL", "llama-3.3-70b-versatile")
    cheap_model = os.environ.get("CHEAP_MODEL", "llama-3.1-8b-instant")

    latencies_baseline = []
    latencies_with_cache = []
    latencies_with_routing = []

    cache_exact_hits = 0
    cache_semantic_hits = 0
    cheap_routes = 0
    premium_routes = 0

    print("Rodando 50 queries...")
    for i, query in enumerate(BENCH_QUERIES, 1):
        print(f"  [{i:02d}/50] {query[:50]:<50}", end=" ", flush=True)

        # --- Baseline: sempre premium, sem cache
        original_model = pipeline.llm_model
        pipeline.llm_model = premium_model

        _, t_baseline = _call_with_retry(pipeline, query)
        latencies_baseline.append(t_baseline)
        pipeline.llm_model = original_model

        # --- Com cache
        t_start = time.perf_counter()
        cached = exact_cache.get(query)
        if cached:
            cache_exact_hits += 1
            t_cache = (time.perf_counter() - t_start) * 1000
            latencies_with_cache.append(t_cache)
        else:
            sem_cached = semantic_cache.get(query)
            if sem_cached:
                cache_semantic_hits += 1
                t_cache = (time.perf_counter() - t_start) * 1000
                latencies_with_cache.append(t_cache)
            else:
                result, t_cache = _call_with_retry(pipeline, query)
                latencies_with_cache.append(t_cache)
                exact_cache.put(query, result["answer"])
                semantic_cache.put(query, result["answer"])

        # --- Com routing
        decision = classify_complexity(query)
        pipeline.llm_model = decision.model
        if decision.complexity == "simple":
            cheap_routes += 1
        else:
            premium_routes += 1

        _, t_routing = _call_with_retry(pipeline, query)
        latencies_with_routing.append(t_routing)
        pipeline.llm_model = original_model

        print(f"baseline={t_baseline:.0f}ms cache={t_cache:.0f}ms routing={t_routing:.0f}ms")

    # Resultados
    import numpy as np
    print()
    print("=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    p95_baseline = float(np.percentile(latencies_baseline, 95))
    p95_cache = float(np.percentile(latencies_with_cache, 95))
    p95_routing = float(np.percentile(latencies_with_routing, 95))

    avg_baseline = sum(latencies_baseline) / len(latencies_baseline)
    avg_cache = sum(latencies_with_cache) / len(latencies_with_cache)
    avg_routing = sum(latencies_with_routing) / len(latencies_with_routing)

    cache_reduction = (1 - avg_cache / avg_baseline) * 100
    routing_reduction = (1 - avg_routing / avg_baseline) * 100

    print(f"{'Estratégia':<30} {'Média (ms)':>12} {'P95 (ms)':>10} {'Redução':>8}")
    print("-" * 65)
    print(f"{'Baseline (premium always)':<30} {avg_baseline:>12.0f} {p95_baseline:>10.0f} {'—':>8}")
    print(f"{'+ Cache (exact + semantic)':<30} {avg_cache:>12.0f} {p95_cache:>10.0f} {cache_reduction:>7.0f}%")
    print(f"{'+ Routing cheap-first':<30} {avg_routing:>12.0f} {p95_routing:>10.0f} {routing_reduction:>7.0f}%")
    print()
    print(f"Cache hits — Exact: {cache_exact_hits}, Semantic: {cache_semantic_hits}")
    print(f"Routing — Cheap: {cheap_routes}/{len(BENCH_QUERIES)}, Premium: {premium_routes}/{len(BENCH_QUERIES)}")

    goal = cache_reduction >= 50 or routing_reduction >= 50
    print()
    if goal:
        print("✅ META ATINGIDA: ≥50% de redução!")
    else:
        print("⚠️  Meta de ≥50% de redução ainda não atingida. Ajuste threshold do semantic cache.")

    print()
    print("Cole estes valores na tabela do README.")


if __name__ == "__main__":
    run_benchmark()
