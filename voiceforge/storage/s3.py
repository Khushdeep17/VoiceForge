"""
voiceforge/storage/s3.py

Uploads files to S3 after local generation.
If S3 is not configured or upload fails, the app continues
serving from local /outputs — S3 is additive, never a hard dependency.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

# pulled from environment — never hardcode credentials
_BUCKET   = os.getenv("S3_BUCKET_NAME", "")
_REGION   = os.getenv("AWS_REGION", "us-east-1")
_EXPIRY   = int(os.getenv("S3_PRESIGN_EXPIRY", 3600))   # seconds


def _client():
    return boto3.client("s3", region_name=_REGION)


def is_configured() -> bool:
    """True only if S3_BUCKET_NAME is set in environment."""
    return bool(_BUCKET)


def upload_file(local_path: str, s3_key: Optional[str] = None) -> Optional[str]:
    """Upload a local file to S3.

    Args:
        local_path: Absolute or relative path to the file.
        s3_key:     Destination key inside the bucket.
                    Defaults to  outputs/<filename>.

    Returns:
        S3 URI (s3://bucket/key) on success, None on failure.
    """
    if not is_configured():
        logger.debug("S3 not configured — skipping upload for %s", local_path)
        return None

    path = Path(local_path)
    key = s3_key or f"outputs/{path.name}"

    try:
        _client().upload_file(str(path), _BUCKET, key)
        s3_uri = f"s3://{_BUCKET}/{key}"
        logger.info("Uploaded %s → %s", path.name, s3_uri)
        return s3_uri

    except NoCredentialsError:
        logger.warning("S3 upload skipped — no AWS credentials found")
    except (BotoCoreError, ClientError) as exc:
        logger.warning("S3 upload failed for %s: %s", local_path, exc)

    return None


def upload_outputs_dir(directory: str = "outputs") -> dict[str, Optional[str]]:
    """Upload every file in a local directory to S3.

    Returns a dict mapping local filename → S3 URI (or None if upload failed).
    """
    results: dict[str, Optional[str]] = {}

    for filepath in Path(directory).iterdir():
        if filepath.is_file():
            results[filepath.name] = upload_file(str(filepath))

    return results


def get_presigned_url(s3_key: str, expiry: int = _EXPIRY) -> Optional[str]:
    """Generate a temporary public URL for an S3 object.

    Args:
        s3_key: Key inside the bucket (e.g. 'outputs/audio.wav').
        expiry: URL lifetime in seconds (default 3600).

    Returns:
        HTTPS presigned URL, or None if generation fails.
    """
    if not is_configured():
        return None

    try:
        url = _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": _BUCKET, "Key": s3_key},
            ExpiresIn=expiry,
        )
        return url

    except (BotoCoreError, ClientError, NoCredentialsError) as exc:
        logger.warning("Could not generate presigned URL for %s: %s", s3_key, exc)
        return None


def upload_and_get_url(local_path: str) -> dict:
    """Convenience: upload a file and return both its S3 URI and presigned URL.

    Always returns a dict — callers can safely ignore S3 fields if None.
    """
    s3_uri = upload_file(local_path)
    if not s3_uri:
        return {"s3_uri": None, "presigned_url": None}

    path = Path(local_path)
    key = f"outputs/{path.name}"
    presigned = get_presigned_url(key)

    return {"s3_uri": s3_uri, "presigned_url": presigned}
