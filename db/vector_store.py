import os
from functools import lru_cache
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "dataset_metadata"
_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


@lru_cache(maxsize=1)
def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_EMBED_FN,
    )


def _build_doc(table: str, column: str, profile_entry: dict) -> str:
    dtype = profile_entry.get("dtype", "unknown")
    null_pct = profile_entry.get("null_pct", 0)
    unique = profile_entry.get("unique_count", 0)
    return (
        f"table: {table}, column: {column}, type: {dtype}, "
        f"null_pct: {null_pct}%, unique_values: {unique}"
    )


def index_dataset_in_chroma(table: str, columns: list[str], profile: dict) -> None:
    collection = _get_collection()
    ids, docs, metas = [], [], []
    for col in columns:
        doc_id = f"{table}::{col}"
        doc = _build_doc(table, col, profile.get(col, {}))
        ids.append(doc_id)
        docs.append(doc)
        metas.append({"table": table, "column": col})
    collection.upsert(ids=ids, documents=docs, metadatas=metas)


def search_metadata(query: str, n_results: int = 10) -> list[dict]:
    collection = _get_collection()
    try:
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"doc": d, "meta": m} for d, m in zip(docs, metas)]
    except Exception:
        return []


def remove_dataset_from_chroma(table: str, columns: list[str]) -> None:
    collection = _get_collection()
    ids = [f"{table}::{col}" for col in columns]
    collection.delete(ids=ids)
