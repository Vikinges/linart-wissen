"""Download the embedding model (BAAI/bge-m3) for LinArt Wissen."""
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download  # noqa: E402

dst = Path(__file__).resolve().parent / "models" / "bge-m3"
snapshot_download("BAAI/bge-m3", local_dir=str(dst), ignore_patterns=["onnx/*", "imgs/*", "*.onnx*"])
print("model ready in", dst)
