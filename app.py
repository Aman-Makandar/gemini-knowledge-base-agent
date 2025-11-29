# app.py - Robust Gemini Knowledge Base Agent (upgraded, resilient)
from dotenv import load_dotenv
load_dotenv()

import os
import io
import hashlib
import numpy as np
import streamlit as st
from pypdf import PdfReader

# Try to import google.generativeai; if not available, the code still runs with fallback embeddings.
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ------------------ CONFIG ------------------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # Show in UI later, but raise helpful message now if running non-UI tests
    pass
else:
    if GENAI_AVAILABLE:
        try:
            genai.configure(api_key=API_KEY)
        except Exception:
            # some versions/configurations may require different init; ignore here and handle later
            pass

# Default model choices (we will auto-detect best available)
DEFAULT_EMBED_MODEL = "models/text-embedding-004"
TRY_CHAT_MODELS = [
    "gemini-1.5-flash",     # new naming
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "models/gemini-pro",    # older SDK model name
]

# ------------------ PDF TEXT EXTRACTION ------------------
def get_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for p in reader.pages:
        page_text = p.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ------------------ TEXT CHUNKING ------------------
def split_text(text, chunk_size=800, overlap=200):
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks


# ==========================
#  EMBEDDING (robust & safe)
# ==========================
def _local_fallback_embedding(text, dim=512):
    """Deterministic fallback embedding (fast, reproducible)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    nums = []
    i = 0
    while len(nums) < dim:
        nums.append(h[i % len(h)])
        i += 1
    arr = np.array(nums, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _try_module_embeddings(text, model_name):
    """Try top-level module functions from various SDK versions."""
    if not GENAI_AVAILABLE:
        raise AttributeError("genai module missing")

    # genai.Embedding.create(...)
    if hasattr(genai, "Embedding"):
        try:
            resp = genai.Embedding.create(model=model_name, input=text)
            return resp["data"][0]["embedding"]
        except Exception:
            pass

    # genai.embed_text(...)
    if hasattr(genai, "embed_text"):
        try:
            resp = genai.embed_text(model=model_name, text=text)
            if isinstance(resp, dict):
                if "embedding" in resp:
                    return resp["embedding"]
                if "data" in resp and resp["data"]:
                    return resp["data"][0].get("embedding")
        except Exception:
            pass

    # genai.embed_content(...)
    if hasattr(genai, "embed_content"):
        try:
            resp = genai.embed_content(model=model_name, content=text)
            if isinstance(resp, dict):
                if "embedding" in resp:
                    return resp["embedding"]
                if "data" in resp and resp["data"]:
                    return resp["data"][0].get("embedding")
        except Exception:
            pass

    raise AttributeError("no module-level embed method")


def _try_model_embeddings(text, model_name):
    """Try model-level embedding methods (GenerativeModel(...).embed_...)"""
    if not GENAI_AVAILABLE:
        raise AttributeError("genai module missing")
    try:
        model = genai.GenerativeModel(model_name)
    except Exception:
        raise AttributeError("GenerativeModel unavailable")

    for method_name in ("embed_content", "embed_text", "embed", "embedVector"):
        if hasattr(model, method_name):
            try:
                method = getattr(model, method_name)
                resp = method(text)
                # handle common return shapes
                if isinstance(resp, dict):
                    if "embedding" in resp:
                        return resp["embedding"]
                    if "data" in resp and resp["data"]:
                        d0 = resp["data"][0]
                        if isinstance(d0, dict):
                            return d0.get("embedding") or d0.get("vector")
                if hasattr(resp, "embedding"):
                    return resp.embedding
                if isinstance(resp, (list, np.ndarray)):
                    return list(resp)
            except Exception:
                continue
    raise AttributeError("No model embed method found")


def _get_embedding_for(text, model_name=DEFAULT_EMBED_MODEL):
    """Try available embedding APIs, else fallback to local embedding."""
    # 1) Try module-level
    try:
        emb = _try_module_embeddings(text, model_name)
        return np.array(emb, dtype=np.float32)
    except Exception:
        pass

    # 2) Try model-level
    try:
        emb = _try_model_embeddings(text, model_name)
        return np.array(emb, dtype=np.float32)
    except Exception:
        pass

    # 3) Local deterministic fallback
    return _local_fallback_embedding(text, dim=512)


def embed_texts(texts):
    """Compute embeddings for a list of texts; return np.array shape (n, d)."""
    if not texts:
        return np.empty((0, 512), dtype=np.float32)
    embs = [ _get_embedding_for(t) for t in texts ]
    maxlen = max(len(e) for e in embs)
    arrs = []
    for e in embs:
        if len(e) < maxlen:
            e = np.pad(e, (0, maxlen - len(e)))
        arrs.append(e)
    return np.vstack(arrs).astype(np.float32)


def embed_query(query):
    return _get_embedding_for(query)


# ------------------ COSINE SIMILARITY ------------------
def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)


# ------------------ RETRIEVAL ------------------
def retrieve_chunks(query, chunks, chunk_embeddings, top_k=5):
    if len(chunks) == 0 or chunk_embeddings is None:
        return []
    q_emb = embed_query(query)
    sims = cosine_similarity(chunk_embeddings, q_emb)
    idxs = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in idxs]


# ------------------ MODEL SELECTION & GENERATION ------------------
def detect_best_chat_model():
    """Try to detect a supported chat model. Returns a model name string."""
    if not GENAI_AVAILABLE:
        return None
    # If genai has a list_models function, try it
    try:
        if hasattr(genai, "list_models"):
            models = genai.list_models()
            # models may be a list of model dicts or objects; normalize
            available_names = []
            for m in models:
                try:
                    name = m["name"] if isinstance(m, dict) and "name" in m else getattr(m, "name", None)
                except Exception:
                    name = None
                if name:
                    available_names.append(name)
            # pick first match from TRY_CHAT_MODELS that is available
            for candidate in TRY_CHAT_MODELS:
                if candidate in available_names:
                    return candidate
    except Exception:
        # ignore any listing errors
        pass

    # fallback heuristics: try creating GenerativeModel for each candidate
    for candidate in TRY_CHAT_MODELS:
        try:
            model = genai.GenerativeModel(candidate)
            # if construction worked, return candidate
            return candidate
        except Exception:
            continue

    # no model found
    return None


def generate_answer_with_model(model_name, prompt):
    """Create/generate with the selected model. Returns text or raises."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("Google Generative AI SDK not installed.")
    # Try model-based generate_content
    try:
        model = genai.GenerativeModel(model_name)
        # many SDKs return an object; attempt to get text
        resp = model.generate_content(prompt)
        # resp.text is common in earlier snippets
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # sometimes resp has 'candidates' or 'output'
        if isinstance(resp, dict):
            # common pattern: resp['candidates'][0]['content'][0]['text'] etc. try a few
            if "candidates" in resp and resp["candidates"]:
                cand = resp["candidates"][0]
                if isinstance(cand, dict):
                    # try nested text fields
                    for key in ("content", "text", "output"):
                        if key in cand:
                            v = cand[key]
                            if isinstance(v, list) and len(v) and isinstance(v[0], dict) and "text" in v[0]:
                                return v[0]["text"]
                            if isinstance(v, str):
                                return v
            if "output" in resp and isinstance(resp["output"], str):
                return resp["output"]
        # last resort: str(resp)
        return str(resp)
    except Exception as e:
        # bubble up for caller to handle
        raise


# ------------------ STREAMLIT UI ------------------
def main():
    st.set_page_config(page_title="Gemini Knowledge Base Agent", layout="wide")
    st.title("📚 Gemini Knowledge Base Agent")

    # Show API key load status
    if not API_KEY:
        st.error("🚨 GEMINI_API_KEY is not set in your .env file. Add GEMINI_API_KEY=... and restart.")
        return

    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if "chunks" not in st.session_state:
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.chat_model = None

    # Allow manual model detection button
    if st.sidebar.button("Detect best chat model"):
        detected = detect_best_chat_model()
        st.session_state.chat_model = detected
        if detected:
            st.sidebar.success(f"Using model: {detected}")
        else:
            st.sidebar.warning("No supported remote model detected; generator will attempt fallbacks.")

    # Process uploaded PDFs
    if uploaded_files:
        full_text = ""
        for f in uploaded_files:
            t = get_pdf_text(io.BytesIO(f.read()))
            full_text += t
        chunks = split_text(full_text)
        if not chunks:
            st.warning("Uploaded PDFs contained no extractable text.")
            return
        embeddings = embed_texts(chunks)
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.sidebar.success(f"Indexed {len(chunks)} text chunks successfully.")

    st.subheader("Ask a question about your uploaded documents")
    if not st.session_state.chunks:
        st.info("Upload PDF documents to begin.")
        return

    question = st.text_input("Your question:")
    top_k = st.slider("Top K retrieved chunks", 1, 10, 5)

    if st.button("Get Answer") and question.strip():
        with st.spinner("Thinking..."):
            relevant = retrieve_chunks(question, st.session_state.chunks, st.session_state.embeddings, top_k=top_k)

            # Build prompt for the generator
            context_text = "\n\n".join(relevant)
            prompt = f"""You are a Knowledge Base Agent.
Answer the user's question using ONLY the context below. If the answer is not in the context,
reply: "I could not find this information in the provided documents."

Context:
{context_text}

Question:
{question}
"""

            # choose model: session override -> auto-detect -> fallback
            model_name = st.session_state.get("chat_model") or detect_best_chat_model() or "models/gemini-pro"

            # Try generating; if model fails, try fallback list
            answer = None
            tried = []
            candidates = [model_name] + [m for m in TRY_CHAT_MODELS if m != model_name]
            for m in candidates:
                if not m or m in tried:
                    continue
                tried.append(m)
                try:
                    answer = generate_answer_with_model(m, prompt)
                    # got an answer
                    st.success(f"Used model: {m}")
                    break
                except Exception as e:
                    # log briefly and continue to next
                    st.warning(f"Model {m} failed: {str(e)}")
                    continue

            if answer is None:
                # fallback: return a helpful message
                answer = "Could not generate answer using remote models. Showing retrieved context instead.\n\n" + context_text

        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Context"):
            for i, ch in enumerate(relevant, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(ch)
                st.markdown("---")


if __name__ == "__main__":
    main()
