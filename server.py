"""LinArt Wissen - ask your own documents, fully local.

Index: PDF/DOCX/TXT/MD/HTML/CSV from configured folders -> chunks with page numbers.
Search: hybrid = semantic (bge-m3 on the GPU, multilingual RU/DE/EN) + keyword (BM25,
for exact terms like register numbers, part numbers), fused with reciprocal rank fusion.
Answer: local LLM in LM Studio, only from the found passages, with [n] citations.
Run: start_wissen.bat -> http://127.0.0.1:7890
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Optional

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

APP = Path(__file__).resolve().parent
DATA = APP / "data"
DATA.mkdir(exist_ok=True)
CFG_FILE = DATA / "config.json"
DEFAULT_CFG = {
    "folders": [str(Path.home() / "Documents")],
    "extensions": [".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".csv"],
    "llm_url": "http://127.0.0.1:1234/v1",
    "llm_model": "qwen/qwen3-vl-30b",
    "embed_model": str(APP / "models" / "bge-m3"),
    "top_k": 8,
    "rescan_minutes": 10,
}
_mi = APP / ".miopen"
_mi.mkdir(exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(_mi))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(_mi))


def load_cfg() -> dict:
    c = dict(DEFAULT_CFG)
    try:
        c.update(json.loads(CFG_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return c


def save_cfg(c: dict) -> None:
    CFG_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")


CFG = load_cfg()
save_cfg(CFG)

import numpy as np  # noqa: E402

# ---------------------------------------------------------------- DB
DB = DATA / "index.db"
_db_lock = threading.Lock()


def db():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime REAL, size INTEGER,
        pages INTEGER, chunks INTEGER, scanned_pages INTEGER, indexed_at TEXT, error TEXT);
    CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER, text TEXT, emb BLOB);
    CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
    """)

# ---------------------------------------------------------------- text extraction
def extract_pages(p: Path) -> list[tuple[Optional[int], str]]:
    suf = p.suffix.lower()
    if suf == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(p))
        return [(i + 1, pdf[i].get_textpage().get_text_range()) for i in range(len(pdf))]
    if suf == ".docx":
        import docx
        d = docx.Document(str(p))
        parts = [para.text for para in d.paragraphs]
        for t in d.tables:
            for r in t.rows:
                parts.append(" | ".join(c.text for c in r.cells))
        return [(None, "\n".join(parts))]
    if suf in (".html", ".htm"):
        from bs4 import BeautifulSoup
        return [(None, BeautifulSoup(p.read_text(encoding="utf-8", errors="replace"), "html.parser").get_text("\n"))]
    return [(None, p.read_text(encoding="utf-8", errors="replace"))]


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text] if len(text) > 20 else []
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):   # cut at a sentence / line end if possible
            k = max(text.rfind(". ", i + size // 2, j), text.rfind("\n", i + size // 2, j))
            if k > 0:
                j = k + 1
        piece = text[i:j].strip()
        if len(piece) > 20:
            out.append(piece)
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return out


# ---------------------------------------------------------------- embeddings
_embedder = None
_emb_lock = threading.Lock()


def embedder():
    global _embedder
    with _emb_lock:
        if _embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            m = SentenceTransformer(CFG["embed_model"], device=dev)
            if dev == "cuda":
                m = m.half()
            _embedder = m
            print(f"[wissen] embedder ready on {dev}", flush=True)
        return _embedder


def embed(texts: list[str]) -> np.ndarray:
    m = embedder()
    v = m.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return v.astype(np.float16)


# ---------------------------------------------------------------- in-memory search index
IDX = {"ids": np.zeros(0, int), "emb": np.zeros((0, 1024), np.float16), "bm25": None, "tok": [],
       "meta": {}, "version": 0}
_idx_lock = threading.Lock()
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(s: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(s)]


def rebuild_memory_index() -> None:
    from rank_bm25 import BM25Okapi
    with db() as c:
        rows = c.execute("SELECT chunks.id, chunks.doc_id, chunks.page, chunks.text, chunks.emb, docs.path "
                         "FROM chunks JOIN docs ON docs.id = chunks.doc_id ORDER BY chunks.id").fetchall()
    ids = np.array([r["id"] for r in rows], int)
    emb = np.stack([np.frombuffer(r["emb"], np.float16) for r in rows]) if rows else np.zeros((0, 1024), np.float16)
    toks = [tokenize(r["text"]) for r in rows]
    meta = {r["id"]: {"doc_id": r["doc_id"], "page": r["page"], "text": r["text"], "path": r["path"]} for r in rows}
    with _idx_lock:
        IDX.update(ids=ids, emb=emb, tok=toks, meta=meta, bm25=BM25Okapi(toks) if toks else None,
                   version=IDX["version"] + 1)
    print(f"[wissen] memory index: {len(ids)} chunks", flush=True)


# ---------------------------------------------------------------- indexer
STATE = {"indexing": False, "current": None, "done": 0, "total": 0, "last_scan": None, "errors": 0}


def scan_files() -> list[Path]:
    exts = set(e.lower() for e in CFG["extensions"])
    files = []
    for f in CFG["folders"]:
        root = Path(f)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("~$"):
                files.append(p)
    return files


def index_all() -> None:
    if STATE["indexing"]:
        return
    STATE.update(indexing=True, done=0, errors=0)
    try:
        files = scan_files()
        paths = {str(p) for p in files}
        with db() as c:
            known = {r["path"]: r for r in c.execute("SELECT * FROM docs")}
        todo = [p for p in files if str(p) not in known
                or abs(known[str(p)]["mtime"] - p.stat().st_mtime) > 1 or known[str(p)]["size"] != p.stat().st_size]
        removed = [k for k in known if k not in paths]
        STATE["total"] = len(todo)
        with _db_lock, db() as c:
            for k in removed:
                did = known[k]["id"]
                c.execute("DELETE FROM chunks WHERE doc_id=?", (did,))
                c.execute("DELETE FROM docs WHERE id=?", (did,))
        for p in todo:
            STATE["current"] = p.name
            try:
                pages = extract_pages(p)
                pieces, scanned = [], 0
                for page, text in pages:
                    if page is not None and len(text.strip()) < 30:
                        scanned += 1
                        continue
                    for ch in chunk_text(text):
                        pieces.append((page, f"[{p.stem}{f', S. {page}' if page else ''}]\n{ch}"))
                vecs = embed([t for _, t in pieces]) if pieces else np.zeros((0, 1024), np.float16)
                st = p.stat()
                with _db_lock, db() as c:
                    old = c.execute("SELECT id FROM docs WHERE path=?", (str(p),)).fetchone()
                    if old:
                        c.execute("DELETE FROM chunks WHERE doc_id=?", (old["id"],))
                        c.execute("DELETE FROM docs WHERE id=?", (old["id"],))
                    cur = c.execute("INSERT INTO docs(path,mtime,size,pages,chunks,scanned_pages,indexed_at,error) "
                                    "VALUES(?,?,?,?,?,?,?,NULL)",
                                    (str(p), st.st_mtime, st.st_size, len(pages), len(pieces), scanned,
                                     time.strftime("%Y-%m-%d %H:%M")))
                    did = cur.lastrowid
                    c.executemany("INSERT INTO chunks(doc_id,page,text,emb) VALUES(?,?,?,?)",
                                  [(did, pg, tx, v.tobytes()) for (pg, tx), v in zip(pieces, vecs)])
            except Exception as e:
                STATE["errors"] += 1
                print(f"[wissen] {p}: {e}", flush=True)
                traceback.print_exc()
                with _db_lock, db() as c:
                    st = p.stat()
                    c.execute("INSERT OR REPLACE INTO docs(path,mtime,size,pages,chunks,scanned_pages,indexed_at,error) "
                              "VALUES(?,?,?,0,0,0,?,?)", (str(p), st.st_mtime, st.st_size,
                                                          time.strftime("%Y-%m-%d %H:%M"), f"{type(e).__name__}: {e}"))
            STATE["done"] += 1
        if todo or removed or IDX["version"] == 0:
            rebuild_memory_index()
    finally:
        STATE.update(indexing=False, current=None, last_scan=time.strftime("%Y-%m-%d %H:%M:%S"))


def indexer_loop() -> None:
    while True:
        try:
            index_all()
        except Exception:
            traceback.print_exc()
        time.sleep(max(1, int(CFG.get("rescan_minutes", 10))) * 60)


# ---------------------------------------------------------------- search + answer
def search(q: str, k: int) -> list[dict]:
    with _idx_lock:
        ids, emb, bm, meta = IDX["ids"], IDX["emb"], IDX["bm25"], IDX["meta"]
    if not len(ids):
        return []
    qv = embed([q])[0].astype(np.float32)
    sem = emb.astype(np.float32) @ qv
    sem_rank = np.argsort(-sem)[:50]
    kw_rank = np.argsort(-bm.get_scores(tokenize(q)))[:50] if bm is not None else []
    score: dict[int, float] = {}
    for r, i in enumerate(sem_rank):
        score[i] = score.get(i, 0) + 1 / (60 + r)
    for r, i in enumerate(kw_rank):
        score[i] = score.get(i, 0) + 1 / (60 + r)
    best = sorted(score, key=lambda i: -score[i])[:k]
    out = []
    for i in best:
        m = meta[int(ids[i])]
        out.append({"chunk_id": int(ids[i]), "doc_id": m["doc_id"], "page": m["page"], "file": Path(m["path"]).name,
                    "path": m["path"], "text": m["text"], "score": round(float(score[i]), 4),
                    "semantic": round(float(sem[i]), 3)})
    return out


SYSTEM = ("Du bist der Wissensassistent von LinArt Labs. Beantworte die Frage NUR mit den nummerierten Quellen. "
          "Zitiere jede Aussage mit [n]. Wenn die Quellen die Antwort nicht enthalten, sag das klar und rate nicht. "
          "Antworte in der Sprache der Frage (Russisch, Deutsch oder Englisch). Technische Werte (Register, "
          "Adressen, Einheiten, Grenzwerte) exakt übernehmen. Kurz und konkret.")


_DE = re.compile(r"[äöüß]|\b(der|die|das|und|ist|wie|welche[rsmn]?|wird|nicht|für|mit|ein[e]?|ich|wo|was|warum)\b", re.I)


def question_language(q: str) -> str:
    if re.search(r"[а-яё]", q, re.I):
        return "Russisch (русский язык)"
    if _DE.search(q):
        return "Deutsch"
    return "Englisch (English)"


def llm(messages: list, max_tokens: int = 900) -> str:
    body = {"model": CFG["llm_model"], "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
    req = urllib.request.Request(CFG["llm_url"] + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def answer(q: str) -> dict:
    t0 = time.time()
    hits = search(q, int(CFG.get("top_k", 8)))
    if not hits:
        return {"answer": "Индекс пуст или ничего не найдено. / Nichts gefunden.", "sources": [], "elapsed": 0}
    ctx = "\n\n".join(f"[{n}] ({h['file']}{', S. ' + str(h['page']) if h['page'] else ''})\n{h['text']}"
                      for n, h in enumerate(hits, 1))
    text = llm([{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Quellen:\n{ctx}\n\nFrage: {q}\n\n"
                                            f"WICHTIG: Antworte ausschließlich auf {question_language(q)}."}])
    for n, h in enumerate(hits, 1):
        h["n"] = n
        h["url"] = f"/api/file/{h['doc_id']}" + (f"#page={h['page']}" if h["page"] else "")
        h["snippet"] = h.pop("text")[:400]
    return {"answer": text, "sources": hits, "elapsed": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- HTTP
from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402

app = FastAPI(title="LinArt Wissen")


@app.get("/", response_class=HTMLResponse)
def index():
    return (APP / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    with db() as c:
        d = c.execute("SELECT COUNT(*) n, COALESCE(SUM(chunks),0) ch, COALESCE(SUM(scanned_pages),0) sc, "
                      "SUM(error IS NOT NULL) er FROM docs").fetchone()
    llm_ok = False
    try:
        with urllib.request.urlopen(CFG["llm_url"] + "/models", timeout=2):
            llm_ok = True
    except Exception:
        pass
    return {"docs": d["n"], "chunks": d["ch"], "scanned_pages": d["sc"], "doc_errors": d["er"] or 0,
            "folders": CFG["folders"], "llm_ok": llm_ok, "llm_model": CFG["llm_model"], **STATE}


@app.post("/api/ask")
def ask(body: dict = Body(...)):
    q = (body.get("q") or "").strip()
    if not q:
        raise HTTPException(400, "empty question")
    try:
        return answer(q)
    except urllib.error.URLError:
        raise HTTPException(503, "LM Studio nicht erreichbar (Server an Port 1234 starten, Modell laden)")


@app.get("/api/search")
def api_search(q: str, k: int = 8):
    return search(q, k)


@app.get("/api/docs")
def docs():
    with db() as c:
        return [dict(r) | {"name": Path(r["path"]).name} for r in
                c.execute("SELECT id,path,pages,chunks,scanned_pages,indexed_at,error FROM docs ORDER BY path")]


@app.get("/api/file/{doc_id}")
def file(doc_id: int):
    with db() as c:
        r = c.execute("SELECT path FROM docs WHERE id=?", (doc_id,)).fetchone()
    if not r or not Path(r["path"]).exists():
        raise HTTPException(404, "not found")
    p = Path(r["path"])
    media = "application/pdf" if p.suffix.lower() == ".pdf" else None
    return FileResponse(p, media_type=media, filename=p.name, content_disposition_type="inline")


@app.post("/api/folders")
def set_folders(body: dict = Body(...)):
    folders = [f.strip().strip('"') for f in body.get("folders", []) if f.strip()]
    bad = [f for f in folders if not Path(f).is_dir()]
    if bad:
        raise HTTPException(400, "Ordner nicht gefunden: " + ", ".join(bad))
    CFG["folders"] = folders
    save_cfg(CFG)
    threading.Thread(target=index_all, daemon=True).start()
    return {"ok": True, "folders": folders}


@app.post("/api/reindex")
def reindex():
    threading.Thread(target=index_all, daemon=True).start()
    return {"ok": True}


threading.Thread(target=indexer_loop, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    port = int(os.environ.get("WISSEN_PORT", "7890"))
    if os.environ.get("WISSEN_NO_BROWSER") != "1":
        threading.Timer(2.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
