"""Tests run fully offline: local storage in a temp dir, fake embeddings, and
the LLM replaced per-test via monkeypatch. No API key or AWS account needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["EMBEDDINGS"] = "fake"
os.environ["S3_BUCKET"] = ""
os.environ["LLM_CACHE"] = "false"


@pytest.fixture
def storage(tmp_path, monkeypatch):
    from src import config, storage as storage_mod

    monkeypatch.setattr(config, "LOCAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(config, "VECTOR_DIR", str(tmp_path / "vectorstore"))
    storage_mod.get_storage.cache_clear()
    yield storage_mod.get_storage()
    storage_mod.get_storage.cache_clear()


@pytest.fixture
def patients(storage):
    from src.storage import save_patients
    from src.synthetic_data import generate_patients

    data = generate_patients(30, seed=7)
    save_patients(data, storage)
    return data
