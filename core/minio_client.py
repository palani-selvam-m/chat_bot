from minio import Minio
from pathlib import Path

class MinioUploader:
    def __init__(self,
                 endpoint="localhost:9000",
                 access_key="admin",
                 secret_key="admin123",
                 bucket="lexichat"):
        self.client = Minio(endpoint,
                            access_key=access_key,
                            secret_key=secret_key,
                            secure=False)
        self.bucket = bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def upload_file(self, file_path: Path, object_name: str = None) -> str:
        object_name = object_name or file_path.name
        self.client.fput_object(self.bucket, object_name, str(file_path))
        return f"{self.bucket}/{object_name}"

    def download_file(self, object_name: str, dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, object_name, str(dest_path))
        return dest_path
