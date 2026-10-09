"""RAG layer (plan T2.1-T2.3): load -> chunk -> embed -> index -> retrieve.

* `PatientRecordLoader` is a LangChain document loader that turns each patient
  record into one Document per section (demographics, discharge, medications,
  appointments, responses), tagged with `patient_id` metadata.
* Long-term memory (past reminders, responses, escalations) is indexed too,
  as `section="history"` documents, refreshed after every run.
* The index is ChromaDB by default (FAISS with VECTOR_STORE=faiss), persisted
  to VECTOR_DIR and snapshotted to storage (S3) as vectorstore/index.tar.gz.
* Retrieval is ALWAYS filtered by `patient_id` metadata, so one patient's
  prompt can never be grounded in another patient's records.

    python -m src.rag build                      # (re)build the index
    python -m src.rag query P0001 "medications"  # inspect retrieval
"""
import argparse
import logging
import os
import shutil
import tarfile
import tempfile
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .memory import list_memories
from .storage import get_storage, load_patients

log = logging.getLogger(__name__)

SNAPSHOT_KEY = "vectorstore/index.tar.gz"
_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


# --- Loading -------------------------------------------------------------------


def _patient_sections(p: dict) -> dict[str, str]:
    d, dis = p["demographics"], p["discharge"]
    meds = "\n".join(
        f"- {m['name']} {m['dose']}, {m['frequency']} ({m['change']})" for m in p["medications"]
    ) or "- none"
    appts = "\n".join(
        f"- {a['appointment_id']}: {a['type']} on {a['date']}"
        f"{' [CRITICAL]' if a['critical'] else ''} -- status: {a['status']}"
        for a in p["appointments"]
    )
    responses = "\n".join(
        f"- {r['date']} via {r['channel']}: {r['status']}"
        + (f' -- "{r["message"]}"' if r["message"] else "")
        for r in p["responses"]
    ) or "- no reminders sent yet"
    return {
        "demographics": (
            f"Patient {p['patient_id']}: {d['name']}, {d['age']}-year-old {d['gender']}. "
            f"Preferred channel: {d['preferred_channel']}. Language: {d['language']}."
        ),
        "discharge": (
            f"Discharged {dis['date']} with {dis['diagnosis']}. {dis['summary']} "
            f"Warning signs to watch for: {', '.join(dis['warning_signs'])}."
        ),
        "medications": f"Discharge medications:\n{meds}",
        "appointments": f"Follow-up appointments:\n{appts}",
        "responses": f"Reminder response log:\n{responses}",
    }


def history_text(memory: dict) -> str:
    lines = [f"Follow-up history for {memory['patient_id']} from previous runs:"]
    for r in memory["reminders"]:
        lines.append(f"- reminder {r['created_at']} via {r['channel']} for {r.get('appointment_id')}: {r['text']}")
    for r in memory["responses"]:
        lines.append(f"- patient response {r['date']} via {r['channel']}: {r['status']} {r['message']}".rstrip())
    for e in memory["escalations"]:
        lines.append(f"- risk flag {e['risk_flag']} on {e['created_at']}: {'; '.join(e['reasons'])}")
    return "\n".join(lines)


class PatientRecordLoader(BaseLoader):
    """LangChain loader over patient records (+ long-term memory) in storage."""

    def __init__(self, patients: list[dict] | None = None, memories: list[dict] | None = None, storage=None):
        self.storage = storage or get_storage()
        self.patients = patients
        self.memories = memories

    def lazy_load(self) -> Iterator[Document]:
        patients = self.patients if self.patients is not None else load_patients(self.storage)
        for p in patients:
            for section, text in _patient_sections(p).items():
                yield Document(page_content=text, metadata={"patient_id": p["patient_id"], "section": section})
        memories = self.memories if self.memories is not None else list_memories(self.storage)
        for m in memories:
            yield history_document(m)


def history_document(memory: dict) -> Document:
    return Document(page_content=history_text(memory),
                    metadata={"patient_id": memory["patient_id"], "section": "history"})


def chunk(documents: list[Document]) -> list[Document]:
    return _splitter.split_documents(documents)


# --- Embeddings & vector store ---------------------------------------------------


class ChromaDefaultEmbeddings(Embeddings):
    """ChromaDB's bundled all-MiniLM-L6-v2 (ONNX) -- local, no API key needed."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    if config.EMBEDDINGS == "fake":
        return DeterministicFakeEmbedding(size=256)
    return ChromaDefaultEmbeddings()


def _chunk_ids(chunks: list[Document]) -> list[str]:
    counters: dict[tuple, int] = {}
    ids = []
    for c in chunks:
        key = (c.metadata["patient_id"], c.metadata["section"])
        counters[key] = counters.get(key, -1) + 1
        ids.append(f"{key[0]}:{key[1]}:{counters[key]}")
    return ids


class PatientIndex:
    """Thin wrapper so the rest of the app doesn't care if it's Chroma or FAISS."""

    def __init__(self, store, kind: str, directory: str):
        self.store, self.kind, self.directory = store, kind, directory

    def retrieve(self, patient_id: str, query: str, k: int = config.RETRIEVER_K) -> list[Document]:
        docs = self.store.similarity_search(query, k=k, filter={"patient_id": patient_id})
        # Belt and braces: never let another patient's chunk through.
        return [d for d in docs if d.metadata.get("patient_id") == patient_id]

    def _history_ids(self, patient_id: str) -> list[str]:
        prefix = f"{patient_id}:history:"
        if self.kind == "chroma":
            return [i for i in self.store.get(where={"patient_id": patient_id})["ids"] if i.startswith(prefix)]
        return [i for i in self.store.index_to_docstore_id.values() if i.startswith(prefix)]

    def upsert_history(self, memory: dict) -> None:
        old = self._history_ids(memory["patient_id"])
        if old:
            self.store.delete(ids=old)
        chunks = chunk([history_document(memory)])
        self.store.add_documents(chunks, ids=_chunk_ids(chunks))
        self.persist()

    def persist(self) -> None:
        if self.kind == "faiss":
            self.store.save_local(self.directory)
        # Chroma's PersistentClient writes through to disk on every change.

    def snapshot_to_storage(self, storage=None) -> None:
        storage = storage or get_storage()
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "index.tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(self.directory, arcname=".")
            storage.upload_file(archive, SNAPSHOT_KEY)


def _open_store(kind: str, directory: str, embeddings: Embeddings, chunks: list[Document] | None):
    if kind == "faiss":
        from langchain_community.vectorstores import FAISS

        if chunks is not None:
            return FAISS.from_documents(chunks, embeddings, ids=_chunk_ids(chunks))
        return FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)

    from langchain_chroma import Chroma

    store = Chroma(collection_name="patients", embedding_function=embeddings, persist_directory=directory)
    if chunks is not None:
        store.add_documents(chunks, ids=_chunk_ids(chunks))
    return store


def build_index(patients: list[dict] | None = None, storage=None, directory: str | None = None,
                snapshot: bool = True) -> PatientIndex:
    storage = storage or get_storage()
    directory = directory or config.VECTOR_DIR
    kind = config.VECTOR_STORE
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)

    docs = PatientRecordLoader(patients=patients, storage=storage).load()
    chunks = chunk(docs)
    per_patient: dict[str, int] = {}
    for c in chunks:
        per_patient[c.metadata["patient_id"]] = per_patient.get(c.metadata["patient_id"], 0) + 1
    log.info("Indexing %d chunks for %d patients (%s)", len(chunks), len(per_patient), kind)

    index = PatientIndex(_open_store(kind, directory, get_embeddings(), chunks), kind, directory)
    index.persist()
    if snapshot:
        index.snapshot_to_storage(storage)
    return index


def load_index(storage=None, directory: str | None = None) -> PatientIndex:
    """Opens the persisted index; restores it from the S3 snapshot or builds it if missing."""
    storage = storage or get_storage()
    directory = directory or config.VECTOR_DIR
    kind = config.VECTOR_STORE
    if not os.path.isdir(directory) or not os.listdir(directory):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "index.tar.gz")
            if storage.download_file(SNAPSHOT_KEY, archive):
                os.makedirs(directory, exist_ok=True)
                with tarfile.open(archive) as tar:
                    tar.extractall(directory, filter="data")
                log.info("Restored vector index from %s", storage.describe())
            else:
                return build_index(storage=storage, directory=directory)
    return PatientIndex(_open_store(kind, directory, get_embeddings(), None), kind, directory)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    q = sub.add_parser("query")
    q.add_argument("patient_id")
    q.add_argument("text")
    args = parser.parse_args()

    if args.cmd == "build":
        index = build_index()
        print(f"Index built at {index.directory} and snapshotted to {get_storage().describe()}/{SNAPSHOT_KEY}")
    else:
        for doc in load_index().retrieve(args.patient_id, args.text):
            print(f"[{doc.metadata['patient_id']} / {doc.metadata['section']}]\n{doc.page_content}\n")


if __name__ == "__main__":
    main()
