import os
import time
import hashlib
from pathlib import Path

import streamlit as st
import chromadb
import tiktoken

from pypdf import PdfReader
from google import genai
from google.genai import types


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="College GPT",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "PDFs"
DB_DIR = BASE_DIR / "college_database"

PDF_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

CHAT_MODEL = "gemini-3.8-flash"
EMBED_MODEL = "gemini-embedding-001"

COLLECTION_NAME = "college_documents"

# Faster indexing settings
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
BATCH_SIZE = 5
BATCH_DELAY = 20

TOP_K = 5

# ==========================================
# 2. GEMINI API
# ==========================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "Gemini API key missing. "
        "Set GEMINI_API_KEY in PowerShell and restart."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)


# ==========================================
# 3. DATABASE
# ==========================================

db = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = db.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ==========================================
# 4. TOKENIZER
# ==========================================

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = None


def split_text(text):

    if tokenizer:
        tokens = tokenizer.encode(text)
        chunks = []

        step = CHUNK_SIZE - CHUNK_OVERLAP

        for start in range(0, len(tokens), step):
            part = tokens[start:start + CHUNK_SIZE]

            if part:
                chunks.append(tokenizer.decode(part))

        return chunks

    words = text.split()
    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + CHUNK_SIZE]))

    return chunks


# ==========================================
# 5. READ PDF FILES
# ==========================================

def read_all_pdfs():

    all_documents = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:

        try:
            reader = PdfReader(str(pdf_file))

            for page_number, page in enumerate(
                reader.pages, start=1
            ):

                text = page.extract_text() or ""

                if not text.strip():
                    continue

                chunks = split_text(text)

                for chunk_number, chunk in enumerate(chunks):

                    doc_id = hashlib.sha256(
                        (
                            pdf_file.name
                            + str(page_number)
                            + str(chunk_number)
                            + chunk
                        ).encode("utf-8")
                    ).hexdigest()

                    all_documents.append({
                        "id": doc_id,
                        "text": chunk,
                        "source": pdf_file.name,
                        "page": page_number,
                        "chunk": chunk_number
                    })

        except Exception as error:
            st.warning(
                f"Could not read {pdf_file.name}: {error}"
            )

    return all_documents, pdf_files


# ==========================================
# 6. GEMINI EMBEDDINGS
# ==========================================

def generate_embeddings(texts, task_type):

    for attempt in range(3):

        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type
                )
            )

            return [
                item.values
                for item in response.embeddings
            ]

        except Exception as error:

            error_text = str(error).upper()

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                if attempt == 2:
                    raise

                wait_time = 25 * (attempt + 1)

                time.sleep(wait_time)

            else:
                raise


# ==========================================
# 7. PDF INDEXING
# ==========================================

def index_pdfs():

    documents, pdf_files = read_all_pdfs()

    if not pdf_files:
        return 0, 0, "No PDF files found in PDFs folder."

    if not documents:
        return (
            len(pdf_files),
            0,
            "PDFs found, but no readable text was extracted."
        )

    existing_ids = set(collection.get()["ids"])

    new_documents = [
        item for item in documents
        if item["id"] not in existing_ids
    ]

    added = 0
    total_batches = (
        len(new_documents) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    progress = st.progress(0)
    status = st.empty()

    for start in range(
        0, len(new_documents), BATCH_SIZE
    ):

        batch = new_documents[
            start:start + BATCH_SIZE
        ]

        texts = [item["text"] for item in batch]

        try:
            embeddings = generate_embeddings(
                texts,
                "RETRIEVAL_DOCUMENT"
            )

            collection.add(
                ids=[item["id"] for item in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "source": item["source"],
                        "page": item["page"],
                        "chunk": item["chunk"]
                    }
                    for item in batch
                ]
            )

            added += len(batch)

            current_batch = start // BATCH_SIZE + 1

            progress.progress(
                current_batch / max(total_batches, 1)
            )

            status.write(
                f"Indexing: {added}/{len(new_documents)} chunks"
            )

        except Exception as error:

            progress.empty()
            status.empty()

            return (
                len(pdf_files),
                added,
                f"Indexing stopped. {added} chunks added. Error: {error}"
            )

        if start + BATCH_SIZE < len(new_documents):
            time.sleep(BATCH_DELAY)

    progress.empty()
    status.empty()

    return len(pdf_files), added, None


# ==========================================
# 8. RETRIEVE COLLEGE DOCUMENTS
# ==========================================

def retrieve_documents(question):

    if collection.count() == 0:
        return [], []

    query_embedding = generate_embeddings(
        [question],
        "RETRIEVAL_QUERY"
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count())
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []

    return documents, metadatas


# ==========================================
# 9. GEMINI AI CHAT
# ==========================================

def ask_college_gpt(question, history):

    documents = []
    metadatas = []

    try:
        if collection.count() > 0:
            documents, metadatas = retrieve_documents(
                question
            )

    except Exception:
        # General AI can still answer if PDF search fails
        pass

    context = ""
    references = []

    for document, metadata in zip(
        documents, metadatas
    ):

        context += (
            f"\nPDF: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Content: {document}\n"
        )

        references.append({
            "source": metadata["source"],
            "page": metadata["page"]
        })

    previous_chat = ""

    for message in history[-10:]:

        previous_chat += (
            f"\n{message['role']}: "
            f"{message['content']}"
        )

    prompt = f"""
You are College GPT, a helpful AI assistant like ChatGPT.

GENERAL QUESTIONS:
- Answer general questions using your knowledge.
- Help with maths, science, coding and education.
- Explain concepts with simple examples.
- Do not require PDFs for general questions.

COLLEGE QUESTIONS:
- Use provided college PDF context for college facts.
- Never invent fees, dates, rules or syllabus details.
- If college information is missing, say it was not found.
- Never invent PDF names or page numbers.

LANGUAGE:
- Reply in the student's language.
- Support Punjabi, Hindi and English.
- Give friendly, clear and useful answers.

CONVERSATION:
- Use previous chat for context.
- Answer follow-up questions properly.

PREVIOUS CHAT:
{previous_chat}

COLLEGE PDF CONTEXT:
{context if context else "No relevant college PDF information found."}

STUDENT QUESTION:
{question}

Give the best possible answer.
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    answer = response.text or "No response generated."

    return answer, references


# ==========================================
# 10. CHATGPT STYLE DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #212121;
    color: #ffffff;
}

section[data-testid="stSidebar"] {
    background-color: #171717;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.stChatMessage {
    border-radius: 14px;
}

div[data-testid="stChatInput"] {
    border-radius: 20px;
}

h1, h2, h3, p {
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 11. SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "pdf_count" not in st.session_state:
    st.session_state.pdf_count = 0

if "index_status" not in st.session_state:
    st.session_state.index_status = ""


# ==========================================
# 12. AUTOMATIC PDF LOADING
# ==========================================

if not st.session_state.pdf_loaded:

    with st.spinner("Loading your college PDFs..."):

        total, added, error = index_pdfs()

        st.session_state.pdf_count = total
        st.session_state.pdf_loaded = True

        if error:
            st.session_state.index_status = error
        else:
            st.session_state.index_status = (
                f"Successfully added {added} new chunks."
            )


# ==========================================
# 13. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎓 College GPT")

    st.caption("Your University AI Assistant")

    st.divider()

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.subheader("📚 Knowledge Base")

    st.write(
        f"PDF Files: {st.session_state.pdf_count}"
    )

    st.write(
        f"Stored Chunks: {collection.count()}"
    )

    st.caption("Powered by Gemini AI")

    if st.session_state.index_status:
        st.caption(st.session_state.index_status)

    st.divider()

    if st.button(
        "🔄 Reload PDF Documents",
        use_container_width=True
    ):

        with st.spinner("Updating PDF knowledge..."):

            total, added, error = index_pdfs()

            st.session_state.pdf_count = total

            if error:
                st.error(error)
            else:
                st.success(
                    f"Added {added} new chunks."
                )

            st.session_state.index_status = (
                error or f"Added {added} new chunks."
            )


# ==========================================
# 14. MAIN CHAT SCREEN
# ==========================================

if not st.session_state.chat_history:

    st.markdown(
        "<h1 style='text-align:center; "
        "margin-top:100px;'>🎓 College GPT</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; color:#aaaaaa;'>"
        "Ask anything. Learn anything."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;'>"
        "📚 Syllabus &nbsp; | &nbsp; "
        "💰 Fees &nbsp; | &nbsp; "
        "📅 Calendar &nbsp; | &nbsp; "
        "💡 General AI"
        "</p>",
        unsafe_allow_html=True
    )


# ==========================================
# 15. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("references"):

            with st.expander("📖 View PDF Sources"):

                unique_refs = {
                    (
                        ref["source"],
                        ref["page"]
                    )
                    for ref in message["references"]
                }

                for source, page in sorted(unique_refs):

                    st.write(
                        f"📄 {source} — Page {page}"
                    )


# ==========================================
# 16. CHAT INPUT
# ==========================================

question = st.chat_input(
    "Message College GPT..."
)

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                history = (
                    st.session_state.chat_history[:-1]
                )

                answer, references = ask_college_gpt(
                    question,
                    history
                )

                st.markdown(answer)

                if references:

                    with st.expander("📖 View PDF Sources"):

                        unique_refs = {
                            (
                                ref["source"],
                                ref["page"]
                            )
                            for ref in references
                        }

                        for source, page in sorted(unique_refs):

                            st.write(
                                f"📄 {source} — Page {page}"
                            )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "references": references
                })

            except Exception as error:

                error_message = f"Gemini error: {error}"

                st.error(error_message)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_message,
                    "references": []
                })

    st.rerun()