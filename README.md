# 🔍 LangChain RAG Project

> A from-scratch Retrieval-Augmented Generation pipeline — no black boxes.
> Ingest documents → chunk → embed → store in FAISS → retrieve → summarize with an LLM.

[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-powered-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036)](https://groq.com/)
[![Status](https://img.shields.io/badge/status-in%20progress-yellow)]()

Most RAG demos hide the interesting parts behind a framework call. This one wires up
**every stage by hand** — loaders, a chunker, a `sentence-transformers` encoder, a raw
FAISS index, and a retrieval-then-prompt loop — so you can see exactly where a document
goes and why a given chunk came back as the answer.

---

## 🧠 Why this project exists

RAG is easy to *call* and hard to *tune*. Chunk size, overlap, the distance metric, and
`top_k` all quietly decide whether your answers are good. This repo keeps each knob
visible and hand-rolled, so you can change one at a time and watch what happens.

---

## ✨ Features

- **Multi-format ingestion** — PDF, TXT, CSV, Excel, Word, and JSON through LangChain document loaders.
- **Recursive chunking** — structure-aware splitting on `\n\n` → `\n` → space, so paragraphs stay intact.
- **Local embeddings** — `all-MiniLM-L6-v2` runs on CPU, no API key, 384-dimensional vectors.
- **Hand-built FAISS store** — a plain `IndexFlatL2` you can save, reload, and inspect.
- **Retrieve-then-generate** — embed the query, pull top-`k` neighbors, feed them to a Groq LLM as context.
- **Notebook-first workflow** — experiment in Jupyter, promote the parts that work into `src/`.

---

## 🏗️ Architecture

```
  Data/                        ingest              chunk                 embed
┌──────────────────┐      ┌───────────────┐   ┌────────────────┐   ┌──────────────────────┐
│ Data/pdf/*.pdf   │      │               │   │                │   │                      │
│ Data/text_files/ │─────▶│ data_loader   │──▶│  embedding     │──▶│  384-d float32       │
│ *.csv *.xlsx ... │      │  (LangChain)  │   │  1000 / 200    │   │  all-MiniLM-L6-v2    │
└──────────────────┘      └───────────────┘   └────────────────┘   └──────────┬───────────┘
                                                                             │
                                                                             ▼
                                                                  ┌─────────────────────┐
                                                                  │ FAISS IndexFlatL2   │
                                                                  │ faiss_store/        │
                                                                  └──────────┬──────────┘
                                                                             │
   query ──▶ embed ──▶ top-k search ──▶ context ──▶ Groq LLM ──▶ summary ◀───┘
```

| Stage | Module | What it does |
| --- | --- | --- |
| 1. Ingest | `src/data_loader.py` | Walks `Data/` and returns LangChain `Document` objects |
| 2. Chunk | `src/embedding.py` | `RecursiveCharacterTextSplitter` → 1000 chars, 200 overlap |
| 3. Embed | `src/embedding.py` | `sentence-transformers` encodes chunks to 384-d vectors |
| 4. Store | `src/vectorstore.py` | Adds vectors to `IndexFlatL2`, persists index + metadata |
| 5. Retrieve | `src/vectorstore.py` | L2 nearest-neighbour search over the index |
| 6. Generate | `src/search.py` | Stuffs retrieved text into a prompt for a Groq LLM |

---

## 🚀 Quickstart

**1. Create the virtual environment**

```bash
# Git Bash / macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file in the project root (it's git-ignored):

```bash
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com/keys).

**4. Drop your documents into `Data/`**

```
Data/
├── pdf/          # .pdf files
├── text_files/   # .txt files
└── ...           # .csv, .xlsx, .docx, .json
```

**5. Run it**

```bash
python app.py
```

The first run loads every file, chunks and embeds it, then writes
`faiss_store/faiss.index` + `faiss_store/metadata.pkl`. Later runs detect the saved
index and skip straight to loading — so re-running is fast.

---

## 🗂️ Project layout

```
LangChain-RAG-project/
├── app.py                    # Entry point: build-or-load the store, then ask a question
├── src/
│   ├── data_loader.py        # Multi-format ingestion → List[Document]
│   ├── embedding.py          # Chunking + sentence-transformers embeddings
│   ├── vectorstore.py        # FaissVectorStore: build, save, load, search
│   └── search.py             # RAGSearch: retrieve context + call the Groq LLM
├── notebook/
│   ├── document.ipynb        # Ingestion experiments
│   └── rag.ipynb             # Retrieval / prompting experiments
├── Data/                     # ← your documents live here
├── faiss_store/              # Generated at runtime: faiss.index + metadata.pkl
└── requirements.txt
```

---

## ⚙️ Configuration

Every knob lives in a constructor argument — no config files, no magic constants.

| Parameter | Where | Default | Effect |
| --- | --- | --- | --- |
| `chunk_size` | `EmbeddingPipeline`, `FaissVectorStore` | `1000` | Characters per chunk |
| `chunk_overlap` | `EmbeddingPipeline`, `FaissVectorStore` | `200` | Shared characters between neighbours |
| `embedding_model` | `FaissVectorStore` | `all-MiniLM-L6-v2` | Local encoder (must be 384-d for a match) |
| `persist_dir` | `FaissVectorStore` | `faiss_store` | Where the index is saved |
| `top_k` | `query()`, `search_and_summarize()` | `5` | How many chunks become context |
| `llm_model` | `RAGSearch` | `openai/gpt-oss-20b` | Groq model used to write the summary |

Tuning starting points:

- **Answers miss obvious facts** → raise `top_k` (3 → 5 → 10) before you blame the model.
- **Chunks straddle two ideas** → drop `chunk_size` to ~500 and overlap to ~100.
- **Answers ignore document context** → the LLM is over-reading the prompt; make the
  "answer only from the context" instruction stricter.

---

## 🔬 How retrieval actually works here

`IndexFlatL2` compares vectors with **squared Euclidean distance**, so a *smaller*
distance means a *closer* match. That's the opposite of the cosine-similarity scores
you'll see in most tutorials — keep it in mind when you read raw results:

```python
store.query("What is SEO?", top_k=3)
# [{'index': 12, 'distance': 1.0843, 'metadata': {...}},   ← best hit (lowest distance)
#  {'index': 47, 'distance': 1.2261, 'metadata': {...}},
#  {'index': 9,  'distance': 1.3902, 'metadata': {...}}]
```

Note the index is **flat** — every query is an exact brute-force scan. Perfect for
hundreds or a few thousand chunks; swap in `IndexIVFFlat` once you're past that.

---

## 🩺 Troubleshooting

Real errors you may hit, and what they mean:

**`RuntimeError: ... could not open faiss_store\faiss.index for reading: No such file or directory`**
The index was never built. `app.py` now builds it automatically on first run, but if you
call `store.load()` directly on an empty `faiss_store/`, you'll see this. Build first, or
delete a half-written `faiss_store/` and let it rebuild.

**`UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position ...`**
`TextLoader` defaults to the platform encoding (cp1252 on Windows), which chokes on
UTF-8 files containing smart quotes. The loaders now pass `encoding="utf-8"` explicitly —
if you add a new loader type, do the same.

**`GroqError: The api_key client option must be set ...`**
No key reached `ChatGroq`. Set `GROQ_API_KEY` in `.env` and make sure `load_dotenv()`
runs before the client is constructed.

**`ModuleNotFoundError: No module named 'data_loader'`**
An import is missing its package prefix. Run from the project root and import as
`from src.data_loader import load_all_documents`.

**One file silently doesn't load**
The loaders catch per-file exceptions and log `[ERROR]`. Watch for that line — a single
malformed file is skipped while the rest of the run continues.

---

## 🗺️ Roadmap

- [ ] Load `GROQ_API_KEY` from the environment instead of a literal in `src/search.py`
- [ ] Store `source` and `page` in metadata so answers can cite their chunk
- [ ] Incremental re-indexing — only re-embed files whose modification time changed
- [ ] Move experimental notebook logic into `src/` and delete the dead ends
- [ ] Add a retrieval evaluation harness (hit rate @ k) so tuning is measurable
- [ ] Optional pluggable backends — the code already keeps embedding and store separate

---

## 📓 Notebooks

`notebook/` is the lab bench: `document.ipynb` pokes at ingestion, `rag.ipynb` at
retrieval and prompting. Use the `.venv` kernel (requires `ipykernel`) and promote
anything worth keeping into `src/`.

---

<sub>Built as a learning project — the goal is understanding every step, not shipping the fewest lines.</sub>
