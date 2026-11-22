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
PDF_DIR.mkdir(exist_ok=True)

# On Render, set CHROMA_DB_PATH=/var/data/college_database
# Locally, the existing database folder is reused.
DB_DIR = Path(
    os.getenv(
        "CHROMA_DB_PATH",
        str(BASE_DIR / "college_database")
    )
)

DB_DIR.mkdir(parents=True, exist_ok=True)

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

COLLECTION_NAME = "college_documents"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

BATCH_SIZE = 5
BATCH_DELAY = 2

TOP_K = 5


# ==========================================
# 2. GEMINI API
# ==========================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "Gemini API key missing. "
        "Please set GEMINI_API_KEY in your environment."
    )
    st.stop()


@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


client = get_gemini_client(API_KEY)


# ==========================================
# 3. PERSISTENT CHROMADB
# ==========================================

@st.cache_resource
def get_database(database_path):

    db_client = chromadb.PersistentClient(
        path=database_path
    )

    db_collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return db_client, db_collection


db, collection = get_database(str(DB_DIR))


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

        chunks.append(
            " ".join(words[start:start + CHUNK_SIZE])
        )

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
# 6. GEMINI RETRY SYSTEM
# ==========================================

def is_temporary_error(error):

    error_text = str(error).upper()

    return any(code in error_text for code in [
        "429",
        "500",
        "502",
        "503",
        "504",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "INTERNAL"
    ])


def generate_embeddings(texts, task_type):

    for attempt in range(4):

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

            if attempt == 3 or not is_temporary_error(error):
                raise

            time.sleep(3 * (2 ** attempt))


def generate_ai_response(prompt):

    for attempt in range(4):

        try:

            return client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt
            )

        except Exception as error:

            if attempt == 3 or not is_temporary_error(error):
                raise

            time.sleep(3 * (2 ** attempt))


# ==========================================
# 7. PDF INDEXING
# ==========================================

def index_pdfs():

    # This function runs only during first indexing
    # or when the user manually requests an update.

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

    if not new_documents:

        return (
            len(pdf_files),
            0,
            "All PDF chunks are already stored. Nothing to download."
        )

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

        texts = [
            item["text"] for item in batch
        ]

        try:

            embeddings = generate_embeddings(
                texts,
                "RETRIEVAL_DOCUMENT"
            )

            collection.add(
                ids=[
                    item["id"] for item in batch
                ],
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
                f"Indexing stopped after {added} chunks. Error: {error}"
            )

        if start + BATCH_SIZE < len(new_documents):
            time.sleep(BATCH_DELAY)

    progress.empty()
    status.empty()

    return (
        len(pdf_files),
        added,
        None
    )


# ==========================================
# 8. LOAD EXISTING DATABASE
# ==========================================

def initialize_knowledge_base():

    # IMPORTANT:
    # If chunks already exist, DO NOT READ PDFs.
    # Reuse the existing ChromaDB database.

    stored_chunks = collection.count()

    if stored_chunks > 0:

        return (
            len(list(PDF_DIR.glob("*.pdf"))),
            f"Existing knowledge loaded: {stored_chunks} chunks."
        )

    # Only first-time indexing reaches this section.

    total, added, error = index_pdfs()

    if error:

        return total, error

    return (
        total,
        f"First indexing complete. Added {added} chunks."
    )


# ==========================================
# 9. RETRIEVE COLLEGE DOCUMENTS
# ==========================================

def retrieve_documents(question):

    total_chunks = collection.count()

    if total_chunks == 0:
        return [], []

    # Only the QUESTION is embedded here.
    # Existing document embeddings are reused.

    query_embedding = generate_embeddings(
        [question],
        "RETRIEVAL_QUERY"
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, total_chunks)
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []

    return documents, metadatas


# ==========================================
# 10. COLLEGE GPT AI CHAT
# ==========================================

def ask_college_gpt(question, history):

    documents = []
    metadatas = []

    try:

        documents, metadatas = retrieve_documents(
            question
        )

    except Exception as error:

        st.warning(
            f"Document search temporarily unavailable: {error}"
        )

    context = ""
    references = []

    for document, metadata in zip(
        documents, metadatas
    ):

        context += (
            f"\nPDF: {metadata.get('source', 'Unknown')}\n"
            f"Page: {metadata.get('page', 'Unknown')}\n"
            f"Content: {document}\n"
        )

        references.append({
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown")
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
- Use the provided college PDF context for college facts.
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

    response = generate_ai_response(prompt)

    answer = response.text or "No response generated."

    return answer, references


# ==========================================
# 11. CHATGPT STYLE DESIGN
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
# 12. SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "knowledge_initialized" not in st.session_state:
    st.session_state.knowledge_initialized = False

if "pdf_count" not in st.session_state:
    st.session_state.pdf_count = 0

if "index_status" not in st.session_state:
    st.session_state.index_status = ""


# ==========================================
# 13. SMART FIRST-TIME INITIALIZATION
# ==========================================

if not st.session_state.knowledge_initialized:

    if collection.count() > 0:

        # FAST OPEN:
        # Existing database is reused.
        # No PDF reading, chunking or embedding.

        st.session_state.pdf_count = len(
            list(PDF_DIR.glob("*.pdf"))
        )

        st.session_state.index_status = (
            f"Knowledge Base ready: {collection.count()} stored chunks."
        )

    else:

        # Only when the database is empty,
        # perform the first-time PDF indexing.

        with st.spinner(
            "First-time setup: preparing college documents..."
        ):

            total, status = initialize_knowledge_base()

            st.session_state.pdf_count = total
            st.session_state.index_status = status

    st.session_state.knowledge_initialized = True


# ==========================================
# 14. SIDEBAR
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

    st.caption(
        st.session_state.index_status
    )

    st.divider()

    if st.button(
        "🔄 Update PDF Documents",
        use_container_width=True
    ):

        with st.spinner(
            "Checking for new PDF documents..."
        ):

            total, added, error = index_pdfs()

            st.session_state.pdf_count = total

            if error:

                st.error(error)

            else:

                st.success(
                    f"Update complete! Added {added} new chunks."
                )

            st.session_state.index_status = (
                error or f"Added {added} new chunks."
            )


# ==========================================
# 15. MAIN CHAT SCREEN
# ==========================================

if not st.session_state.chat_history:

    st.markdown(
        """
        <h1 style='text-align:center; margin-top:100px;'>
            🎓 College GPT
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align:center; color:#aaaaaa;'>
            Ask anything. Learn anything.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align:center;'>
            📚 Syllabus &nbsp; | &nbsp;
            💰 Fees &nbsp; | &nbsp;
            📅 Calendar &nbsp; | &nbsp;
            💡 General AI
        </p>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 16. DISPLAY CHAT HISTORY
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
# 17. CHAT INPUT
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