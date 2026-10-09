"""Key/value JSON storage: S3 when S3_BUCKET is set, a local folder otherwise.

Layout (same keys in both backends):
  patients/<patient_id>.json        synthetic patient records (input)
  memory/<patient_id>.json          long-term memory: reminders, responses, escalations
  logs/<date>/<run_id>/<step>.json  every agent output, for review and evaluation
  vectorstore/index.tar.gz          snapshot of the persisted vector index
"""
import json
import os
from functools import lru_cache

from . import config


class LocalStorage:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def describe(self) -> str:
        return f"local:{self.root}"

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def read_json(self, key: str, default=None):
        path = self._path(key)
        if not os.path.exists(path):
            return default
        with open(path) as f:
            return json.load(f)

    def write_json(self, key: str, value) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f, indent=2, default=str)

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not os.path.isdir(base):
            return []
        keys = []
        for dirpath, _, files in os.walk(base):
            for name in files:
                full = os.path.join(dirpath, name)
                keys.append(os.path.relpath(full, self.root).replace(os.sep, "/"))
        return sorted(keys)

    def upload_file(self, local_path: str, key: str) -> None:
        dest = self._path(key)
        if os.path.abspath(local_path) == dest:
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(local_path, "rb") as src, open(dest, "wb") as out:
            out.write(src.read())

    def download_file(self, key: str, local_path: str) -> bool:
        src = self._path(key)
        if not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        with open(src, "rb") as f, open(local_path, "wb") as out:
            out.write(f.read())
        return True


class S3Storage:
    def __init__(self, bucket: str, prefix: str, region: str):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        # No keys passed: boto3 uses the EC2 instance's IAM role (or the local
        # AWS CLI profile when developing).
        self.s3 = boto3.client("s3", region_name=region)

    def describe(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def read_json(self, key: str, default=None):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self._key(key))
        except self.s3.exceptions.NoSuchKey:
            return default
        return json.loads(obj["Body"].read())

    def write_json(self, key: str, value) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key(key),
            Body=json.dumps(value, indent=2, default=str).encode(),
            ContentType="application/json",
        )

    def list_keys(self, prefix: str) -> list[str]:
        strip = len(self._key(""))
        keys = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            keys.extend(obj["Key"][strip:] for obj in page.get("Contents", []))
        return sorted(keys)

    def upload_file(self, local_path: str, key: str) -> None:
        self.s3.upload_file(local_path, self.bucket, self._key(key))

    def download_file(self, key: str, local_path: str) -> bool:
        from botocore.exceptions import ClientError

        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        try:
            self.s3.download_file(self.bucket, self._key(key), local_path)
        except ClientError:
            return False
        return True


@lru_cache(maxsize=1)
def get_storage():
    if config.S3_BUCKET:
        return S3Storage(config.S3_BUCKET, config.S3_PREFIX, config.AWS_REGION)
    return LocalStorage(config.LOCAL_DATA_DIR)


# --- Patient records -----------------------------------------------------------


def load_patients(storage=None) -> list[dict]:
    storage = storage or get_storage()
    return [
        storage.read_json(key)
        for key in storage.list_keys("patients/")
        if key.endswith(".json")
    ]


def load_patient(patient_id: str, storage=None) -> dict | None:
    storage = storage or get_storage()
    return storage.read_json(f"patients/{patient_id}.json")


def save_patients(patients: list[dict], storage=None) -> None:
    storage = storage or get_storage()
    for patient in patients:
        storage.write_json(f"patients/{patient['patient_id']}.json", patient)
