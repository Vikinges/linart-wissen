# LinArt Wissen 📚

**Ask questions about your own documents — fully local RAG with citations. Your files never leave your PC.**

Point it at folders (manuals, contracts, datasheets, notes). It indexes PDF, DOCX, TXT, MD, HTML and CSV,
and answers questions in Russian, German or English with numbered citations `[1]`, `[2]` with file name and page, linking to the source file.

## Features

- **Hybrid search** – dense embeddings (BAAI/bge-m3, multilingual, GPU) + BM25 keywords, merged with Reciprocal Rank Fusion.
  Finds both "what is meant" and exact part numbers / error codes.
- **Answers with sources** – local LLM (LM Studio / any OpenAI-compatible server) answers only from retrieved passages,
  each claim cited; answers in the language of the question.
- **Cross-language** – ask in Russian about a German or English manual.
- **Incremental indexing** – changed files are re-indexed automatically (default every 10 min) or on demand.
- **Clean web UI** at `http://127.0.0.1:7890` – ask, search, browse indexed documents, manage folders.

## Install (Windows 10/11, Python 3.12)

```powershell
git clone https://github.com/Vikinges/linart-wissen.git
cd linart-wissen
python -m venv venv
venv\Scripts\activate
```

### PyTorch for your GPU

Install PyTorch **before** the requirements, matching your hardware:

```powershell
# AMD Ryzen AI MAX / Radeon 8060S (gfx1151) - ROCm nightly for Windows
pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
# NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CPU only
pip install torch
```

```powershell
pip install -r requirements.txt
python download_models.py      # ~2.3 GB: BAAI/bge-m3
start_wissen.bat               # opens http://127.0.0.1:7890
```

Start LM Studio with a chat model loaded and the local server enabled (`http://127.0.0.1:1234`).
Add your folders in the UI (default: your *Documents* folder). Settings are stored in `data/config.json`.

## How it works

1. Documents are split into overlapping chunks (page-aware for PDFs) and stored in SQLite.
2. Each chunk gets a bge-m3 embedding; a BM25 index is built over the same chunks.
3. A question is searched both ways, results fused (RRF), top-k passages sent to the LLM with strict
   "answer only from the sources, cite them" instructions.

## Part of LinArt Local AI

Small, private, offline-first tools that run entirely on your own PC — no cloud, no account, no telemetry:

| Project | What it does |
|---|---|
| [linart-diktat](https://github.com/Vikinges/linart-diktat) | Push-to-talk dictation + instant translation in any app |
| [linart-wissen](https://github.com/Vikinges/linart-wissen) | Ask questions about your own documents (local RAG with citations) |
| [linart-belege](https://github.com/Vikinges/linart-belege) | Receipts & invoices → bookkeeping (German EÜR) with a local vision LLM |
| [linart-3d-studio](https://github.com/Vikinges/linart-3d-studio) | Text/image → 3D model, plus a 3D-print check, on your own GPU |

Built and tested on an HP ZBook Ultra G1a (AMD Ryzen AI MAX+ PRO 395, Radeon 8060S, 128 GB unified memory) under Windows 11 with PyTorch ROCm, and designed to also run on NVIDIA (CUDA) or CPU.

## Author

**Vladimir Linartas** — founder of [LinArt Labs](https://linart.club), Bavaria, Germany. Self-hosted IoT, embedded systems and local AI.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Third-party models are downloaded separately and keep their own licenses.
