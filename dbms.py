import streamlit as st
import os
import re
import io
import logging
from typing import TypedDict, List, Optional
import fitz  # PyMuPDF
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# ══════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════
load_dotenv()
PDF_FILE_PATH  = "dbms_notes.pdf"           # ← replace with your PDF path
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM            = ChatOpenAI(model_name="gpt-4o", temperature=0.0, api_key=OPENAI_API_KEY)

# ══════════════════════════════════════════════════════════════════
# Word-number → integer helpers
# ══════════════════════════════════════════════════════════════════
WORD_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}


def words_to_int(text: str) -> Optional[int]:
    text = text.strip().lower().replace("-", " ")
    if text.isdigit():
        return int(text)
    total = 0
    for part in text.split():
        val = WORD_TO_INT.get(part)
        if val is None:
            return None
        total += val
    return total if total > 0 else None


def force_int_page(value) -> Optional[int]:
    """Always returns a Python int or None — never a word-string."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    result = words_to_int(s)
    if result is not None:
        return result
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


# ══════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════
def format_for_display(text: str) -> str:
    """LaTeX-safe formatting + deduplicate diagram note lines."""
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"$\\frac{\1}{\2}$", text)
    lines = text.split("\n")
    seen: set = set()
    out: List[str] = []
    for line in lines:
        if "Relevant diagrams" in line or "relevant diagrams" in line:
            if line not in seen:
                seen.add(line)
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════
# PDF Processing
# ══════════════════════════════════════════════════════════════════
def extract_text_from_pdf(pdf_path: str):
    logger.info(f"Extracting text from PDF: {pdf_path}")
    try:
        pdf = PdfReader(pdf_path)
        output = []
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            text = re.sub(r"\n\s*\n", "\n\n", text)
            output.append((text, i))
        logger.info(f"Extracted text from {len(output)} pages")
        return output
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return []


def extract_images_from_pdf(pdf_path: str) -> dict:
    """Returns {page_number (int): [image_bytes, ...]}"""
    logger.info(f"Extracting images from PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        image_data: dict = {}
        for page_num in range(doc.page_count):
            page = doc[page_num]
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image.get("image")
                if img_bytes:
                    image_data.setdefault(page_num + 1, []).append(img_bytes)
        doc.close()
        logger.info(f"Images found on pages: {sorted(image_data.keys())}")
        return image_data
    except Exception as e:
        logger.error(f"Error extracting images: {e}")
        return {}


def text_to_docs(text_with_pages) -> List[Document]:
    logger.info("Converting text to LangChain Documents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = []
    for text, page_num in text_with_pages:
        for chunk in splitter.split_text(text):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": f"page-{page_num}", "page_num": page_num},
            ))
    logger.info(f"Created {len(docs)} document chunks")
    return docs


def create_vectordb(pdf_path: str):
    logger.info("Building FAISS vector database")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError("No text extracted from PDF.")
    docs = text_to_docs(pages)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    logger.info("FAISS vector database ready")
    return vectordb


# ══════════════════════════════════════════════════════════════════
# Translation
# ══════════════════════════════════════════════════════════════════
def multilingual_translate(text: str, source_lang: str, destination_language: str) -> str:
    if source_lang.strip().lower() == destination_language.strip().lower():
        return text
    user_prompt = (
        f"Translate the following text from {source_lang} to {destination_language}.\n"
        f"Rules:\n"
        f"- Translate all content accurately.\n"
        f"- CRITICAL: Keep ALL numeric digits exactly as Arabic numerals (0-9). "
        f"Do NOT spell out numbers as words and do NOT use other numeral systems.\n"
        f"- Do not omit, add, or interpret anything.\n\n"
        f"Original ({source_lang}):\n{text}\n\n"
        f"Translation ({destination_language}):"
    )
    response = LLM.invoke([
        SystemMessage(content="You are a highly accurate multilingual translator."),
        HumanMessage(content=user_prompt),
    ])
    return response.content.strip()


# ══════════════════════════════════════════════════════════════════
# Retrieval helpers
# ══════════════════════════════════════════════════════════════════
def retrieve_from_pdf(query: str, vectordb, image_data: dict) -> dict:
    """
    Vectordb similarity search.
    Returns content, integer page_num, images ONLY from that page.
    has_images = True only when diagrams actually exist on the retrieved page.
    """
    logger.info(f"Retrieving for query: {query}")
    docs = vectordb.similarity_search(query, k=3)
    if docs:
        doc = docs[0]
        page_num: int = int(doc.metadata["page_num"])
        content = f"Page {page_num}: {doc.page_content}"
        images = image_data.get(page_num, [])
        logger.info(f"Retrieved page {page_num} — images: {len(images)}")
        return {
            "content":    content,
            "page_num":   page_num,
            "image_data": images,
            "has_images": len(images) > 0,
        }
    logger.warning("No content retrieved")
    return {"content": "No content retrieved.", "page_num": None,
            "image_data": [], "has_images": False}


def augment_with_context(content: str, page_num: Optional[int]) -> str:
    if page_num:
        return f"{content}\n\nAdditional context: Sourced from page {page_num}."
    return f"{content}\n\nAdditional context: No specific page identified."


# ══════════════════════════════════════════════════════════════════
# Session history helpers
# ══════════════════════════════════════════════════════════════════
def get_last_assistant_meta() -> dict:
    """
    Walk backwards through st.session_state.messages to find the most recent
    assistant turn and return its page_num + image_indices.
    Returns {"page_num": int|None, "image_indices": list}
    """
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            return {
                "page_num":      msg.get("page_num"),
                "image_indices": msg.get("image_indices", []),
            }
    return {"page_num": None, "image_indices": []}


# ══════════════════════════════════════════════════════════════════
# Agent State
# ══════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    query:              str
    input_language:     str
    chat_history:       List[dict]
    # Intent classification result
    is_followup:        bool      # True  → conversational follow-up (summarize / clarify)
                                  # False → new topic question
    # Retrieval outputs
    retrieved_content:  Optional[str]
    page_num:           Optional[int]
    image_data:         List[bytes]
    has_images:         bool
    # Previous turn metadata (used when is_followup=True)
    prev_page_num:      Optional[int]
    prev_image_indices: List[int]
    # Pipeline outputs
    augmented_content:  Optional[str]
    response:           Optional[str]


# ══════════════════════════════════════════════════════════════════
# Prompts  —  all three agents + intent classifier
# ══════════════════════════════════════════════════════════════════

# ── Intent Classifier Prompt ─────────────────────────────────────
# Determines whether the current query is a follow-up/conversational
# question (summarize, clarify, explain again, what did you mean…)
# or a brand-new topic question that needs fresh retrieval.
INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are an intent classifier for a DBMS AI Tutor chat system.

Given the conversation history and the latest user query, decide whether the
query is a FOLLOW_UP or a NEW_QUESTION.

Definitions:
- FOLLOW_UP: The query does NOT introduce a new DBMS topic. It refers to or
  continues the previous answer. Examples:
    * "Summarize this"  / "এটা summarize করো"
    * "Explain again"   / "আবার বলো"
    * "What do you mean?" / "মানে কী?"
    * "Give an example" (when the previous answer was about a specific topic)
    * "Too long, make it shorter"
    * "Translate that to Hindi"
    * Any question that only makes sense in the context of the last answer
- NEW_QUESTION: The query asks about a specific DBMS/SQL concept or topic
  that is different from (or not directly continuing) the previous answer.
  Examples:
    * "What is normalization?"
    * "Explain B+ trees"
    * "How does indexing work?"

Reply with ONLY one of these two tokens — nothing else:
FOLLOW_UP
NEW_QUESTION
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Latest query: {query}"),
])

# ── 1. Retrieve Agent Prompt ─────────────────────────────────────
RETRIEVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are the Retrieve Agent for a DBMS AI Tutor system.

Your sole responsibility is to identify the most relevant content from the
DBMS textbook for the user's query.

Rules:
- The query you receive is always in English.
- Analyse the query and determine what DBMS topic is being asked about.
- Format your output as: 'Page X: [brief content summary]' when relevant.
- If no relevant content exists, output exactly: 'No content retrieved.'
- Use chat history to understand follow-up or contextual questions.
- Do NOT answer the question yourself — only retrieve and format.
- Do NOT add any explanation, greeting, or extra text.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Retrieve content for this query: {query}"),
])

# ── 2. Augment Agent Prompt ──────────────────────────────────────
AUGMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are the Augment Agent for a DBMS AI Tutor system.

Your sole responsibility is to enrich retrieved content with clear source
attribution so the Generate Agent can cite it properly.

Rules:
- If content is available and a page number is given, return:
  [original content]
  Sourced from page [X].
  where [X] is the EXACT INTEGER provided — NEVER write it as a word.
- If content is 'No content retrieved.' or page number is unknown, return:
  No augmented content.
- Do NOT summarise, rewrite, or answer the question.
- Do NOT add any extra explanation or formatting.
- Use chat history only to stay consistent with prior turns.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Retrieved content: {retrieved_content}\nPage number: {page_num}"),
])

# ── 3. Generate Agent Prompt ─────────────────────────────────────
GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert AI Tutor specialising in Database Management Systems (DBMS).

Your responsibility is to provide clear, accurate, and detailed answers
strictly based on the book content provided to you.

Critical Rules:
- Answer ONLY from the provided book content — do not use outside knowledge.
- NEVER mention page numbers, sources, diagrams, images, or figures inside
  your answer body.
- NEVER write 'Source:', 'Page', 'diagram available', 'figure', or similar.
- Give a clean, well-structured educational answer only.
- For follow-up questions (summarize, clarify, explain again), use the
  chat history to generate an appropriate response based on prior content.
- If the question is completely unrelated to DBMS/SQL/databases/data topics,
  reply with ONLY this single token (nothing else): NOT_APPLICABLE
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "User Query: {query}\n\nContent from book:\n{augmented_content}"),
])


# ══════════════════════════════════════════════════════════════════
# Shared utility
# ══════════════════════════════════════════════════════════════════
def _build_lc_history(chat_history: List[dict]) -> List:
    """Convert raw dict list to LangChain HumanMessage / AIMessage objects."""
    messages = []
    for msg in chat_history:
        if msg.get("type") == "human":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ══════════════════════════════════════════════════════════════════
# Agent Nodes
# ══════════════════════════════════════════════════════════════════

# ── Node 0: Intent Classifier ─────────────────────────────────────
def intent_classifier_agent(state: AgentState) -> AgentState:
    """
    Classifies the query as FOLLOW_UP or NEW_QUESTION.
    If no prior conversation exists, always NEW_QUESTION.
    """
    logger.info("── Intent Classifier: start ──")

    # No history → always treat as new question
    if not state["chat_history"]:
        logger.info("No history — classified as NEW_QUESTION")
        return {
            "is_followup":        False,
            "prev_page_num":      None,
            "prev_image_indices": [],
        }

    english_query = multilingual_translate(
        state["query"], state["input_language"], "English"
    )

    lc_history = _build_lc_history(state["chat_history"])
    classifier_chain = INTENT_CLASSIFIER_PROMPT | LLM | StrOutputParser()
    result = classifier_chain.invoke({
        "query":        english_query,
        "chat_history": lc_history,
    }).strip().upper()

    is_followup = result.startswith("FOLLOW_UP")
    logger.info(f"Intent classifier result: '{result}' → is_followup={is_followup}")

    # Fetch previous turn metadata to reuse if follow-up
    prev_meta = get_last_assistant_meta() if is_followup else {"page_num": None, "image_indices": []}

    return {
        "is_followup":        is_followup,
        "prev_page_num":      force_int_page(prev_meta["page_num"]),
        "prev_image_indices": prev_meta["image_indices"],
    }


# ── Node 1: Retrieve Agent ────────────────────────────────────────
def retrieve_agent(state: AgentState) -> AgentState:
    """
    If is_followup=True  → skip vectordb search, reuse previous page metadata.
    If is_followup=False → do fresh vectordb search.
    """
    logger.info("── Retrieve Agent: start ──")

    if state["is_followup"]:
        # ── Follow-up: reuse previous page and images ─────────────
        prev_page = state["prev_page_num"]
        logger.info(f"Follow-up detected — reusing prev page_num={prev_page}")

        if prev_page is not None:
            # Re-fetch content from the same page so generate_agent has context
            page_images = st.session_state.image_data.get(prev_page, [])
            # Pull content for that page from vectordb using page metadata filter
            all_docs = st.session_state.vectordb.similarity_search(
                state["query"], k=10
            )
            same_page_docs = [
                d for d in all_docs if int(d.metadata["page_num"]) == prev_page
            ]
            if same_page_docs:
                content = f"Page {prev_page}: {same_page_docs[0].page_content}"
            else:
                content = f"Page {prev_page}: (Previously retrieved content for this topic.)"

            return {
                "retrieved_content": content,
                "page_num":          prev_page,
                "image_data":        page_images,
                "has_images":        len(page_images) > 0,
            }
        else:
            # No previous page at all — treat as new question
            logger.info("Follow-up but no prev page — falling back to fresh retrieval")

    # ── New question: translate query and do fresh retrieval ──────
    english_query = multilingual_translate(
        state["query"], state["input_language"], "English"
    )
    logger.info(f"English query: {english_query}")

    # Retrieve intent context via LLM (logging + chain invocation)
    lc_history = _build_lc_history(state["chat_history"])
    retrieve_chain = RETRIEVE_PROMPT | LLM | StrOutputParser()
    llm_output = retrieve_chain.invoke({
        "query":        english_query,
        "chat_history": lc_history,
    })
    logger.info(f"Retrieve LLM output: {llm_output[:100]}...")

    # Actual ground-truth retrieval from vectordb
    retrieved = retrieve_from_pdf(
        english_query,
        st.session_state.vectordb,
        st.session_state.image_data,
    )

    logger.info(
        f"── Retrieve Agent: done — page={retrieved['page_num']}, "
        f"has_images={retrieved['has_images']} ──"
    )
    return {
        "retrieved_content": retrieved["content"],
        "page_num":          retrieved["page_num"],
        "image_data":        retrieved["image_data"],
        "has_images":        retrieved["has_images"],
    }


# ── Node 2: Augment Agent ─────────────────────────────────────────
def augment_agent(state: AgentState) -> AgentState:
    logger.info("── Augment Agent: start ──")

    page_num   = force_int_page(state.get("page_num"))
    lc_history = _build_lc_history(state["chat_history"])

    # Run AUGMENT_PROMPT for structured formatting (LLM call)
    augment_chain = AUGMENT_PROMPT | LLM | StrOutputParser()
    augment_chain.invoke({
        "retrieved_content": state.get("retrieved_content", "No content retrieved."),
        "page_num":          str(page_num) if page_num else "unknown",
        "chat_history":      lc_history,
    })

    # Build augmented content ourselves to guarantee integer page_num
    content = state.get("retrieved_content") or ""
    if content and content != "No content retrieved.":
        augmented = augment_with_context(content, page_num)
    else:
        augmented = "No augmented content."

    logger.info(f"── Augment Agent: done — page_num={page_num} ──")
    return {"augmented_content": augmented, "page_num": page_num}


# ── Node 3: Generate Agent ────────────────────────────────────────
def generate_agent(state: AgentState) -> AgentState:
    logger.info("── Generate Agent: start ──")
    try:
        english_query = multilingual_translate(
            state["query"], state["input_language"], "English"
        )
        lc_history = _build_lc_history(state["chat_history"])

        generate_chain = GENERATE_PROMPT | LLM | StrOutputParser()
        raw_answer = generate_chain.invoke({
            "query":             english_query,
            "augmented_content": state.get("augmented_content") or "No relevant content.",
            "chat_history":      lc_history,
        }).strip()

        # ── Off-topic check ──────────────────────────────────────
        if "NOT_APPLICABLE" in raw_answer.upper():
            final_answer = multilingual_translate(
                "This topic is not covered in the DBMS book.",
                "English", state["input_language"],
            )
            logger.info("── Generate Agent: off-topic ──")
            return {"response": final_answer, "page_num": None,
                    "image_data": [], "has_images": False}

        # ── Translate answer to user's language ─────────────────
        final_answer = multilingual_translate(raw_answer, "English", state["input_language"])

        # ── Resolve page_num — always force to int ───────────────
        page_num: Optional[int] = force_int_page(state.get("page_num"))

        # For follow-ups, page_num is already set to prev_page_num by retrieve_agent.
        # If still None (edge case), fall back to last known page from history.
        if page_num is None:
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and isinstance(msg.get("page_num"), int):
                    page_num = msg["page_num"]
                    logger.info(f"Fell back to previous page_num: {page_num}")
                    break

        # ── Build source line with guaranteed digit ──────────────
        if isinstance(page_num, int) and page_num > 0:
            source_en = f"Source: Page {page_num}"
        else:
            source_en = "Source: Page not specified"
            page_num  = None

        translated_source = multilingual_translate(
            source_en, "English", state["input_language"]
        )

        # Safety net: if digit was lost during translation, re-insert it
        if page_num is not None and not re.search(r"\d+", translated_source):
            translated_source = re.sub(
                r"(Page|पृष्ठ|পৃষ্ঠা|पान|பக்கம்|పేజీ|ಪುಟ|താൾ)\s*\S*",
                lambda m: f"{m.group(1)} {page_num}",
                translated_source,
            )
            if not re.search(r"\d+", translated_source):
                translated_source = translated_source.rstrip() + f" {page_num}"

        final_answer += f"\n\n{translated_source}"

        # ── Images: only show when page actually has diagrams ────
        has_images: bool    = state.get("has_images", False)
        images: List[bytes] = state.get("image_data", []) if has_images else []

        if has_images and images:
            img_note = multilingual_translate(
                "Relevant diagrams are available for this topic.",
                "English", state["input_language"],
            )
            final_answer += f"\n\n{img_note}"
            logger.info(f"Image note added — {len(images)} image(s) from page {page_num}")
        else:
            logger.info(
                f"No image note — has_images={has_images}, "
                f"image count={len(state.get('image_data', []))}"
            )

        logger.info(f"── Generate Agent: done — page={page_num}, images={len(images)} ──")
        return {
            "response":   final_answer,
            "page_num":   page_num,
            "image_data": images,
            "has_images": has_images and len(images) > 0,
        }

    except Exception as e:
        logger.error(f"Generate Agent error: {e}", exc_info=True)
        err = multilingual_translate(
            "Sorry, something went wrong. Please try again.",
            "English", state["input_language"],
        )
        return {"response": err, "page_num": None, "image_data": [], "has_images": False}


# ══════════════════════════════════════════════════════════════════
# Conditional edges
# ══════════════════════════════════════════════════════════════════
def decide_augmentation(state: AgentState) -> str:
    """After retrieve_agent: go to augment if content found, else directly to generate."""
    content = state.get("retrieved_content") or ""
    if content and content != "No content retrieved.":
        logger.info("Edge → augmentation")
        return "augmentation"
    logger.info("Edge → generation (no content)")
    return "generation"


# ══════════════════════════════════════════════════════════════════
# Build LangGraph workflow
# ══════════════════════════════════════════════════════════════════
workflow = StateGraph(AgentState)
workflow.add_node("intent_classifier_agent", intent_classifier_agent)
workflow.add_node("retrieve_agent",          retrieve_agent)
workflow.add_node("augment_agent",           augment_agent)
workflow.add_node("generate_agent",          generate_agent)

workflow.set_entry_point("intent_classifier_agent")
workflow.add_edge("intent_classifier_agent", "retrieve_agent")
workflow.add_conditional_edges(
    "retrieve_agent",
    decide_augmentation,
    {"augmentation": "augment_agent", "generation": "generate_agent"},
)
workflow.add_edge("augment_agent",  "generate_agent")
workflow.add_edge("generate_agent", END)

agent = workflow.compile()


# ══════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════
def display_chat_history():
    """Re-render full conversation with images persisted per turn."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                page_num      = message.get("page_num")           # int or None
                image_indices = message.get("image_indices", [])  # [] when no images

                # Show images only when this turn actually had images
                if page_num and image_indices:
                    page_images = st.session_state.image_data.get(page_num, [])
                    for idx in image_indices:
                        if idx < len(page_images):
                            try:
                                st.image(
                                    io.BytesIO(page_images[idx]),
                                    caption=f"Diagram — Page {page_num}",
                                    use_container_width=True,
                                )
                            except Exception as e:
                                logger.error(f"Image re-render error: {e}")
                                st.warning("Could not re-display image.")


def main():
    logger.info("Starting AI Professor Assistant")
    st.set_page_config(page_title="📚 AI Professor Assistant", layout="wide")
    st.title("📚 AI Professor Assistant")
    st.markdown(
        "Ask any question from your **DBMS book** in any language. "
    )

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("🗣️ Language Settings")
        input_language = st.selectbox(
            "Select your language",
            [
                "English", "Hindi", "Bengali", "Tamil", "Telugu",
                "Marathi", "Kannada", "Malayalam", "Gujarati",
            ],
        )

    # ── Session initialisation ────────────────────────────────────
    if "vectordb" not in st.session_state:
        with st.spinner("⏳ Loading PDF and building search index… (first run only)"):
            try:
                st.session_state.vectordb   = create_vectordb(PDF_FILE_PATH)
                st.session_state.image_data = extract_images_from_pdf(PDF_FILE_PATH)
                logger.info("Session state initialised successfully")
            except Exception as e:
                logger.error(f"Initialisation failed: {e}")
                st.error(f"Failed to load PDF: {e}")
                st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render existing conversation ──────────────────────────────
    display_chat_history()

    # ── Chat input ────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything from the DBMS book in any language…")
    if not user_input:
        return

    logger.info(f"User query: '{user_input}' | Language: {input_language}")

    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Build multi-turn history (exclude current user turn) ──────
    chat_history: List[dict] = []
    for msg in st.session_state.messages[:-1]:
        chat_history.append({
            "type":    "human" if msg["role"] == "user" else "ai",
            "content": msg["content"],
        })

    initial_state: AgentState = {
        "query":              user_input,
        "input_language":     input_language,
        "chat_history":       chat_history,
        "is_followup":        False,
        "retrieved_content":  None,
        "page_num":           None,
        "image_data":         [],
        "has_images":         False,
        "prev_page_num":      None,
        "prev_image_indices": [],
        "augmented_content":  None,
        "response":           None,
    }

    # ── Run agent and render response ─────────────────────────────
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            try:
                final_state      = agent.invoke(initial_state)
                answer: str      = final_state["response"]
                formatted_answer = format_for_display(answer)
                placeholder.markdown(formatted_answer)

                page_num: Optional[int]      = final_state.get("page_num")
                image_data_turn: List[bytes] = final_state.get("image_data", [])
                has_images: bool             = final_state.get("has_images", False)
                is_followup: bool            = final_state.get("is_followup", False)
                image_indices: List[int]     = []

                # ── Image display logic ───────────────────────────────
                # For FOLLOW-UP turns: show images from the SAME page as the
                #   previous answer (already loaded into image_data by retrieve_agent).
                # For NEW QUESTION turns: show images only if the retrieved page
                #   actually has diagrams in the PDF.
                # In BOTH cases: suppress if answer is off-topic or no images exist.

                is_off_topic = (
                    "NOT_APPLICABLE" in answer.upper()
                    or "not covered" in answer.lower()
                )

                show_images = (
                    not is_off_topic
                    and has_images
                    and image_data_turn
                    and page_num is not None
                )

                if show_images:
                    for idx, img_bytes in enumerate(image_data_turn):
                        try:
                            st.image(
                                io.BytesIO(img_bytes),
                                caption=f"Diagram — Page {page_num}",
                                use_container_width=True,
                            )
                            image_indices.append(idx)
                            logger.info(
                                f"Displayed image {idx} from page {page_num} "
                                f"(followup={is_followup})"
                            )
                        except Exception as img_err:
                            logger.error(f"Image display error: {img_err}")
                            st.warning("Could not display one of the diagrams.")
                else:
                    if is_off_topic:
                        logger.info("Images suppressed — off-topic")
                    elif not has_images:
                        logger.info(f"Images suppressed — page {page_num} has no diagrams")
                    elif not image_data_turn:
                        logger.info("Images suppressed — image_data is empty")

                # ── Persist assistant message with metadata ───────────
                st.session_state.messages.append({
                    "role":          "assistant",
                    "content":       formatted_answer,
                    "page_num":      page_num,        # int or None
                    "image_indices": image_indices,   # [] when no images shown
                })
                logger.info(
                    f"Saved — page={page_num}, images={len(image_indices)}, "
                    f"followup={is_followup}"
                )

            except Exception as e:
                logger.error(f"Query processing error: {e}", exc_info=True)
                err_msg = multilingual_translate(
                    f"An error occurred while processing your query: {e}",
                    "English", input_language,
                )
                st.error(err_msg)


if __name__ == "__main__":
    main()
