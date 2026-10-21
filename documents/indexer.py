"""JARVIS Lite — Project Document Indexer & Semantic Vector Retrieval.

Indexes document chunks into ChromaDB with project-scoped collections,
and answers natural language queries with verified file and page citations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from ai.model_router import router
from ai.ollama_client import ollama_client
from core.logging import get_logger
from documents.loader import document_loader

logger = get_logger("documents.indexer")
CHROMA_DOCS_DIR = Path("data/chroma_docs")


class DocumentIndexer:
    """Manages project-scoped ChromaDB collections and citation-backed document Q&A."""

    def __init__(self) -> None:
        CHROMA_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DOCS_DIR),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

    def _get_collection_name(self, project: str) -> str:
        clean = re.sub(r"[^\w-]", "_", project.lower()).strip("_")
        return f"project_{clean or 'default'}"[:60]

    def _get_collection(self, project: str) -> Any:
        col_name = self._get_collection_name(project)
        return self._client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def _compute_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors via custom trained neural model or deterministic fallback."""
        try:
            from ai.custom_model_engine import custom_embedding_engine

            return custom_embedding_engine.embed_documents(texts)
        except Exception:
            try:
                embeddings: list[list[float]] = []
                for text in texts:
                    vec = await ollama_client.generate_embedding(text)
                    if vec:
                        embeddings.append(vec)
                    else:
                        embeddings.append(self._fallback_embedding(text))
                return embeddings
            except Exception:
                return [self._fallback_embedding(t) for t in texts]

    def _fallback_embedding(self, text: str, dim: int = 384) -> list[float]:
        """Generate a deterministic normalized float vector based on string hash."""
        import hashlib
        import math
        vec = [0.0] * dim
        tokens = text.lower().split()
        for token in tokens:
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 5) for v in vec]

    async def index_file(self, file_path: Path, project: str = "default") -> int:
        """Parse file, compute embeddings, and insert chunks into the project collection."""
        chunks = document_loader.load(file_path)
        if not chunks:
            return 0

        coll = self._get_collection(project)
        texts = [c.text for c in chunks]
        embeddings = await self._compute_embeddings(texts)

        ids: list[str] = []
        metas: list[dict[str, Any]] = []

        for c in chunks:
            cid = f"{c.file_name}_p{c.page_number}_c{c.chunk_index}"
            ids.append(cid)
            metas.append({
                "source_path": c.file_path,
                "file_name": c.file_name,
                "page_number": c.page_number or 1,
                "chunk_index": c.chunk_index,
                "has_injection_risk": c.has_injection_risk,
            })

        coll.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )

        logger.info(
            "file_indexed_successfully",
            file_name=file_path.name,
            project=project,
            chunks_count=len(chunks),
        )
        return len(chunks)

    async def query_documents(
        self,
        query: str,
        project: str = "default",
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """Perform semantic search across indexed project documents."""
        coll = self._get_collection(project)
        count = coll.count()
        if count == 0:
            return []

        q_embeds = await self._compute_embeddings([query])
        res = coll.query(
            query_embeddings=q_embeds,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if res.get("distances") else []

        for i, doc in enumerate(docs):
            m = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 0.0
            results.append({
                "text": doc,
                "file_name": m.get("file_name", "unknown"),
                "page_number": m.get("page_number", 1),
                "source_path": m.get("source_path", ""),
                "distance": dist,
                "citation": f"[{m.get('file_name', 'unknown')}, Page {m.get('page_number', 1)}]",
            })

        return results

    async def answer_with_citations(self, query: str, project: str = "default") -> str:
        """Answer a user question based on indexed documents with mandatory file + page citations."""
        matches = await self.query_documents(query=query, project=project, top_k=4)
        if not matches:
            return f"I could not find any relevant documents indexed under project '{project}' to answer your question."

        context_lines: list[str] = []
        for m in matches:
            context_lines.append(f"Source: {m['citation']}\nExcerpt:\n{m['text']}")

        context_block = "\n\n---\n\n".join(context_lines)

        prompt = f"""You are JARVIS Lite, answering a question using verified excerpts from the user's local documents.
Rules:
1. Every claim you make MUST cite the exact source in the format: [Filename, Page X].
2. Do not invent facts not present in the excerpts.

Relevant Document Excerpts:
{context_block}

User Question: {query}
"""

        try:
            ans = await router.chat(
                messages=[
                    {"role": "system", "content": "You are a precise technical document assistant that always cites source files and page numbers."},
                    {"role": "user", "content": prompt},
                ],
                task_type="chat",
                temperature=0.1,
            )
            unique_citations = list(dict.fromkeys(m["citation"] for m in matches if m.get("citation")))
            if unique_citations and not any(cit.lower() in ans.lower() for cit in unique_citations):
                ans = f"{ans.strip()}\n\nSources:\n" + "\n".join(f"- {c}" for c in unique_citations)
            return ans
        except Exception:
            # Deterministic fallback synthesizing highest scoring excerpt citation
            top = matches[0]
            clean_snippet = top["text"].strip().replace("\n", " ")[:200]
            return f"According to {top['citation']}: \"{clean_snippet}...\""


document_indexer = DocumentIndexer()
