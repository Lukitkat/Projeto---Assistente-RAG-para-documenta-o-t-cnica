"""RAG pipeline — chunk, embed, index, retrieve, generate.

Usa Groq como LLM provider e fastembed (ONNX, sem PyTorch) para embeddings locais.
Fallback: sentence-transformers se fastembed não disponível.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ------------------------------------------------------------------ Embedding function local
def _build_embedding_function(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingFunction:
    """Cria função de embedding local.

    Tenta em ordem:
    1. fastembed (ONNX, leve, sem PyTorch)
    2. sentence-transformers (requer PyTorch + VC++ Redistributable)
    Levanta RuntimeError se nenhum funcionar.
    """
    # Tentativa 1: fastembed (preferido — mais leve)
    try:
        from fastembed import TextEmbedding as FastTextEmbedding

        class FastEmbedFunction(EmbeddingFunction):
            def __init__(self, model_name: str) -> None:
                self._model = FastTextEmbedding(model_name=model_name, cache_dir=".cache/fastembed")

            def name(self) -> str:
                return f"fastembed:{model_name}"

            def __call__(self, input: Documents) -> Embeddings:
                vecs = list(self._model.embed(list(input)))
                return [v.tolist() for v in vecs]

        print(f"[embedding] Usando fastembed ({model_name})")
        return FastEmbedFunction(model_name=model_name)

    except Exception as e_fast:
        print(f"[embedding] fastembed falhou ({e_fast}), tentando sentence-transformers...")

    # Tentativa 2: sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer

        st_model_name = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

        class SentenceTransformerFunction(EmbeddingFunction):
            def __init__(self, model_name: str) -> None:
                self._model = SentenceTransformer(model_name)

            def name(self) -> str:
                return f"sentence-transformers:{st_model_name}"

            def __call__(self, input: Documents) -> Embeddings:
                vecs = self._model.encode(list(input), normalize_embeddings=True)
                return vecs.tolist()

        print(f"[embedding] Usando sentence-transformers ({st_model_name})")
        return SentenceTransformerFunction(model_name=st_model_name)

    except Exception as e_st:
        print(f"[embedding] sentence-transformers falhou ({e_st})")

    raise RuntimeError(
        "Nenhum provider de embeddings disponível.\n"
        "Instale as dependências:\n"
        "  1. Microsoft Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
        "  2. Reinicie o terminal e tente novamente."
    )


def _make_groq_client() -> Groq:
    """Inicializa cliente Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Configure GROQ_API_KEY no .env. Crie sua chave grátis em https://console.groq.com"
        )
    return Groq(api_key=api_key)


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local + Groq LLM + embeddings locais."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.groq_client = _make_groq_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        embed_model_name = embed_model or os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

        self.embed_fn = _build_embedding_function(model_name=embed_model_name)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    # ------------------------------------------------------------------ TODO 1
    def ingest_and_index(self) -> int:
        """Lê PDFs de `corpus_dir`, faz chunking e indexa em Chroma.

        Retorna número de chunks indexados.
        """
        # Extract text from all PDFs in the corpus folder
        docs: list[dict] = []
        pdf_files = list(self.corpus_dir.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(
                f"Nenhum PDF encontrado em '{self.corpus_dir}'. "
                "Adicione pelo menos 1 arquivo PDF antes de rodar o pipeline."
            )

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(str(pdf_path))
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():  # ignora páginas em branco
                        docs.append({
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_num,
                        })
            except Exception as e:
                print(f"[WARN] Erro ao ler {pdf_path.name}: {e}")

        if not docs:
            raise RuntimeError("PDFs encontrados mas nenhum texto extraível. Use OCR antes.")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks: list[dict] = []

        for doc in docs:
            parts = splitter.split_text(doc["text"])
            for i, part in enumerate(parts):
                chunk_id = f"{doc['source']}-p{doc['page']}-{i}-{uuid.uuid4().hex[:6]}"
                chunks.append({
                    "id": chunk_id,
                    "text": part,
                    "source": doc["source"],
                    "page": doc["page"],
                })

        # Add chunks to the vector store in batches
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start: start + batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
            )

        return self.collection.count()

    # ------------------------------------------------------------------ TODO 2
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares à query."""
        results = self.collection.query(query_texts=[query], n_results=k)

        hits: list[dict] = []
        if not results["documents"] or not results["documents"][0]:
            return hits

        docs_list = results["documents"][0]
        metas_list = results["metadatas"][0]
        dists_list = results["distances"][0]

        for text, meta, dist in zip(docs_list, metas_list, dists_list):
            hits.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "distance": dist,
            })

        return hits

    # ------------------------------------------------------------------ TODO 3
    def answer(self, question: str, k: int = 5) -> dict:
        """Pipeline completo: retrieve + augment + generate. Retorna {answer, sources}."""
        hits = self.retrieve(question, k=k)

        if not hits:
            return {
                "answer": "No relevant information found in the corpus to answer your question.",
                "sources": [],
            }

        # Format the context string including source page references
        context_parts = []
        for hit in hits:
            header = f"[{hit['source']}:page {hit['page']}]"
            context_parts.append(f"{header}\n{hit['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # Inject context and question into the prompt template
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # Fetch generation from the LLM
        response = self.groq_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )

        answer_text = response.choices[0].message.content or "No answer generated."
        sources = [(h["source"], h["page"]) for h in hits]

        return {"answer": answer_text, "sources": sources}


PROMPT_TEMPLATE = """You are a technical assistant specializing in Python documentation.
Answer ONLY based on the context below.
If the information is not in the context, say "Not found in the corpus".
Always cite the source using the format [file:page].
Answer in English.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda não indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
