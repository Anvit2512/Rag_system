import os, uuid, json, fitz
from dotenv import load_dotenv
from app.data.vector_store import VectorStore
from app.data.metadata_store import MetadataStore
from app.utils.chunking import split_by_tokens
from sentence_transformers import SentenceTransformer

load_dotenv()
EMBED_MODEL  = os.getenv("HF_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
VDB_DIR      = os.getenv("VDB_DIR", "./data/chroma")
FILES_DIR    = os.getenv("FILES_DIR", "./data/files")
SQLITE_URL   = os.getenv("SQLITE_URL", "sqlite:///data/metadata.db")

# Configuration
FILES_DIR = os.getenv("FILES_DIR", "./data/files")
VDB_DIR = os.getenv("VDB_DIR", "./data/chroma")
SQLITE_URL = os.getenv("SQLITE_URL", "sqlite:///./data/metadata.db")

# Initialize stores
vs = VectorStore(VDB_DIR, EMBED_MODEL, None)
meta = MetadataStore(SQLITE_URL)

# Initialize local embedding model
print(f"Loading embedding model: {EMBED_MODEL}")
embedding_model = SentenceTransformer(EMBED_MODEL)

# def hf_embed(texts: list[str]) -> list[list[float]]:
#     # Batch embed if the endpoint supports list inputs; otherwise loop.
#     r = requests.post(EMBED_URL, headers=HEADERS, json={"inputs": texts}, timeout=120)
#     r.raise_for_status()
#     out = r.json()
#     if isinstance(out, dict) and "embeddings" in out:
#         return out["embeddings"]
#     if isinstance(out, list) and isinstance(out[0], list):
#         return out
#     if isinstance(out, list) and isinstance(out[0], dict) and "embedding" in out[0]:
#         return [o["embedding"] for o in out]
#     raise ValueError("Unexpected embedding response")



def _sanitize_text(t: str, max_chars: int = 4000) -> str:
    # Trim super-long chunks & normalize whitespace; avoid empty strings
    if not t:
        return ""
    t = " ".join(t.split())
    return t[:max_chars]

def local_embed(texts: list[str]) -> list[list[float]]:
    # Remove empties after sanitization
    batch = [s for s in ( _sanitize_text(x) for x in texts ) if s]
    if not batch:
        # Return zero-vectors or skip; here we skip
        return []

    # Use local sentence transformer model
    embeddings = embedding_model.encode(batch, convert_to_tensor=False)
    return embeddings.tolist()


def ingest_pdf(path: str, title: str | None = None):
    doc = fitz.open(path)
    text_all = []
    for pno in range(len(doc)):
        page = doc[pno]
        text_all.append(page.get_text("text"))
        # Extract images to FILES_DIR
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            fname = f"{uuid.uuid4().hex}.png"
            fpath = os.path.join(FILES_DIR, fname)
            # if pix.n < 5:
            #     pix.save(fpath)
            # else:
            #     fitz.Pixmap(fitz.csRGB, pix).save(fpath)
            try:
                if pix.n < 5:  # RGB or grayscale
                    pix.save(fpath)
                else:  # CMYK or other
                    pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                    pix_converted.save(fpath)
                    pix_converted = None
            except Exception:
                print(f"Skipping unsupported image on page {pno} of {path}")
            finally:
                pix = None

    full_text = "\n\n".join(text_all)

    chunks = split_by_tokens(full_text, max_tokens=300)
    ids = []
    texts = []
    for i, ch in enumerate(chunks):
        doc_id = f"{os.path.basename(path)}::{i}"
        meta.upsert_document(doc_id, title or os.path.basename(path), ch, None)
        ids.append(doc_id)
        texts.append(ch)

    embs = []
    # Embed in safe mini-batches
    B = 8
    for i in range(0, len(texts), B):
        embs.extend(local_embed(texts[i:i+B]))

    vs.add(ids=ids, texts=texts, embeddings=embs)
    print(f"Ingested {path}: {len(ids)} chunks")


if __name__ == "__main__":
    import sys
    pdfs = [p for p in sys.argv[1:] if p.lower().endswith('.pdf')]
    if not pdfs:
        print("Usage: python scripts/ingest_pdfs.py <file1.pdf> <file2.pdf> ...")
        raise SystemExit(1)
    for p in pdfs:
        ingest_pdf(p)