import streamlit as st
import os
import re
import io
import logging
from datetime import datetime
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

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to terminal
        # Optionally add FileHandler for persistent logs
        # logging.FileHandler('ai_tutor.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
PDF_FILE_PATH = "dbms_notes.pdf"  # Replace with your PDF path
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Load from environment variable
LLM = ChatOpenAI(model_name="gpt-4o", temperature=0.0, api_key=OPENAI_API_KEY)

# Formatting for Streamlit display (LaTeX support and deduplication)
def format_for_display(text):
    logger.info("Formatting text for display")
    def replace_latex(match):
        latex_expr = match.group(1)
        return f"$${latex_expr}$$"  # Use $$ for Streamlit Markdown to render LaTeX
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'$\\frac{\1}{\2}$', text)
    
    lines = text.split("\n")
    seen = set()
    deduplicated_lines = []
    for line in lines:
        if "Relevant diagrams are available for this topic" in line:
            if line not in seen:
                seen.add(line)
                deduplicated_lines.append(line)
        else:
            deduplicated_lines.append(line)
    
    return "\n".join(deduplicated_lines)

# Extract text from PDF with page numbers
def extract_text_from_pdf(pdf_path):
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
        logger.info(f"Successfully extracted text from {len(output)} pages")
        return output
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return []

# Extract images from PDF and store in memory
def extract_images_from_pdf(pdf_path):
    logger.info(f"Extracting images from PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        image_data = {}
        for page_num in range(doc.page_count):
            page = doc[page_num]
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if not image_bytes:
                    continue
                if page_num + 1 not in image_data:
                    image_data[page_num + 1] = []
                image_data[page_num + 1].append(image_bytes)
        doc.close()
        logger.info(f"Extracted images from {len(image_data)} pages")
        return image_data
    except Exception as e:
        logger.error(f"Error extracting images: {e}")
        return {}
    
# Convert text to LangChain Documents
def text_to_docs(text_with_pages):
    logger.info("Converting text to LangChain Documents")
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    for text, page_num in text_with_pages:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source": f"page-{page_num}", "page_num": page_num}
            )
            docs.append(doc)
    logger.info(f"Created {len(docs)} documents")
    return docs

# Create FAISS vector database
def create_vectordb(pdf_path):
    logger.info("Creating FAISS vector database")
    text_with_pages = extract_text_from_pdf(pdf_path)
    if not text_with_pages:
        logger.error("No text extracted from PDF")
        raise ValueError("No text extracted from PDF.")
    docs = text_to_docs(text_with_pages)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    logger.info("FAISS vector database created successfully")
    return vectordb

# Multilingual Translation
def multilingual_translate(text: str, source_lang: str, destination_language: str) -> str:
    """
    Translate text from source_language to destination_language using ChatOpenAI.
    Also converts numeric values to the numeral system of the destination language.
    """
    system_msg = "You are a highly accurate multilingual translator."
    user_prompt = (
        f"You are a precise multilingual translator.\n"
        f"Translate the following text from {source_lang} to {destination_language}.\n"
        f"Translate all content accurately, including numerical values into the numeral system of the destination language.\n"
        f"Do not omit, add, or interpret anything.\n\n"
        f"Original Text ({source_lang}):\n{text}\n\n"
        f"Translated Text ({destination_language}):"
    )

    response = LLM.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_prompt)
    ])

    return response.content.strip()

# Define Tools
def retrieve_from_pdf(query: str, vectordb, image_data) -> dict:
    logger.info(f"Retrieving content for query: {query}")
    docs = vectordb.similarity_search(query, k=3)
    if docs:
        doc = docs[0]
        page_num = doc.metadata["page_num"]
        content = f"Page {page_num}: {doc.page_content}"
        images = image_data.get(page_num, [])
        logger.info(f"Retrieved content from page {page_num}, found {len(images)} images")
        return {"content": content, "page_num": page_num, "image_data": images}
    logger.warning("No content retrieved for query")
    return {"content": "No content retrieved.", "page_num": None, "image_data": []}

def augment_with_context(content: str, page_num: Optional[int]) -> str:
    logger.info(f"Augmenting content for page {page_num}")
    if content != "No content retrieved." and page_num:
        augmented_content = f"{content}\n\nAdditional context: Sourced from page {page_num}."
        logger.info("Content augmented successfully")
        return augmented_content
    logger.warning("No content to augment")
    return f"{content}\n\nAdditional context: No specific page identified."

# Define the Agent State
class AgentState(TypedDict):
    query: str
    input_language: str
    chat_history: List[dict]
    retrieved_content: Optional[str]
    page_num: Optional[int]
    image_data: List[bytes]
    augmented_content: Optional[str]
    response: Optional[str]

# Define Agent Prompts
RETRIEVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
        You are the Retrieve Agent. Your task is to assist in fetching the most relevant text and images from a PDF based on the user's query.
        - The query is provided in English after translation if needed.
        - The retrieval function will provide content and a single page number.
        - Format the output as 'Page X: [content]' if content is found, or 'No content retrieved.' if none is found.
        - Use the chat history to maintain context.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

AUGMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
        You are the Augment Agent. Your task is to enhance the retrieved content by adding a clear reference to the page number.
        - If content is available, return the content followed by 'Sourced from page X.' where X is the page number.
        - If no content is retrieved or the page number is missing, return 'No augmented content.'
        - Use the chat history to ensure consistency.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Retrieved content: {retrieved_content}\nPage number: {page_num}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
        You are an expert AI Tutor for Database Management Systems (DBMS).
        Provide clear, detailed, and accurate answers strictly based on the given content.
        
        Important Rules:
        - Never mention page numbers, sources, diagrams, or images.
        - Do not write anything like "Source:", "Page", "diagram available", etc.
        - If the question is not related to DBMS/SQL, reply only: "Not applicable"
        - Just give the clean educational answer.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "User Query: {query}\n\nContent from book:\n{augmented_content}"),
])

# Define Multi-Agent Nodes
def retrieve_agent(state: AgentState) -> AgentState:
    logger.info(f"Retrieve Agent processing query: {state['query']}")
    query_in_english = multilingual_translate(state["query"], source_lang=state["input_language"], destination_language="English")
    retrieved = retrieve_from_pdf(query_in_english, st.session_state.vectordb, st.session_state.image_data)
    logger.info(f"Retrieve Agent completed for page {retrieved['page_num']}")
    return {
        "retrieved_content": retrieved["content"],
        "page_num": retrieved["page_num"],
        "image_data": retrieved["image_data"]
    }

def augment_agent(state: AgentState) -> AgentState:
    logger.info("Augment Agent processing content")
    if state["retrieved_content"] and state["retrieved_content"] != "No content retrieved.":
        augmented_content = augment_with_context(state["retrieved_content"], state["page_num"])
    else:
        augmented_content = "No augmented content."
    logger.info("Augment Agent completed")
    return {"augmented_content": augmented_content}

def generate_agent(state: AgentState) -> AgentState:
    logger.info("Generate Agent: Generating response")
    chain = GENERATE_PROMPT | LLM
    
    try:
        # Translate query to English for accurate understanding
        english_query = multilingual_translate(
            state["query"], 
            state["input_language"], 
            "English"
        )
        
        response = chain.invoke({
            "query": english_query,
            "augmented_content": state["augmented_content"] or "No relevant content found.",
            "chat_history": state["chat_history"]
        })
        
        raw_answer = response.content.strip()
        
        # Check if off-topic
        if "not applicable" in raw_answer.lower():
            translated_answer = multilingual_translate(
                "This question is not covered in the DBMS material.",
                "English",
                state["input_language"]
            )
            final_response = translated_answer
            final_page_num = None
            image_data_to_use = []
        else:
            # Translate the main answer
            translated_answer = multilingual_translate(
                raw_answer, 
                "English", 
                state["input_language"]
            )
            
            # === GUARANTEED: Always add numeric source ===
            page_num = state.get("page_num")
            if page_num is not None and page_num > 0:
                source_line = f"Source: Page {page_num}"
            else:
                source_line = "Source: Not specified"
                
            translated_source = multilingual_translate(
                source_line, "English", state["input_language"]
            )
            
            final_response = translated_answer + f"\n\n{translated_source}"
            
            # Add image message if images exist
            if state.get("image_data"):
                image_note = "Relevant diagrams are available for this topic."
                translated_image_note = multilingual_translate(
                    image_note, "English", state["input_language"]
                )
                final_response += f"\n\n{translated_image_note}"
                image_data_to_use = state["image_data"]
            else:
                image_data_to_use = []
            
            final_page_num = page_num
            
        logger.info(f"Final response generated with page: {final_page_num}")
        
        return {
            "response": final_response,
            "page_num": final_page_num,
            "image_data": image_data_to_use
        }
        
    except Exception as e:
        logger.error(f"Error in generate_agent: {e}")
        error_msg = multilingual_translate(
            "Sorry, an error occurred while processing your question.",
            "English",
            state["input_language"]
        )
        return {
            "response": error_msg,
            "page_num": None,
            "image_data": []
        }

# Define Conditional Edge Logic
def decide_augmentation(state: AgentState) -> str:
    logger.info("Deciding augmentation path")
    if state["retrieved_content"] and state["retrieved_content"] != "No content retrieved.":
        logger.info("Proceeding to augmentation")
        return "augmentation"
    logger.info("Proceeding to generation")
    return "generation"

# Build and Compile the LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_agent", retrieve_agent)
workflow.add_node("augment_agent", augment_agent)
workflow.add_node("generate_agent", generate_agent)
workflow.set_entry_point("retrieve_agent")
workflow.add_conditional_edges(
    "retrieve_agent",
    decide_augmentation,
    {
        "augmentation": "augment_agent",
        "generation": "generate_agent"
    }
)
workflow.add_edge("augment_agent", "generate_agent")
workflow.add_edge("generate_agent", END)
agent = workflow.compile()

# Streamlit UI Configuration
def main():
    logger.info("Starting Streamlit application")
    # Page configuration
    st.set_page_config(page_title="📚 AI Tutor Agent", layout="wide")
    st.title("📚 AI Tutor Agent")
    st.markdown(
        "Ask any question from your book in any language, and get detailed answers "
        "with a single source page and relevant images!"
    )

    # Sidebar - Language Selection
    with st.sidebar:
        st.header("🗣️ Language Settings")
        input_language = st.selectbox(
            "Select Language",
            ["English", "Tamil", "Marathi", "Malayalam", "Telugu", "Hindi", "Kannada", "Gujarati", "Bengali"]
        )

    # Initialize session state
    if "vectordb" not in st.session_state:
        logger.info("Initializing vector database and image data")
        with st.spinner("Loading PDF content and images... This may take a minute."):
            try:
                st.session_state.vectordb = create_vectordb(PDF_FILE_PATH)
                st.session_state.image_data = extract_images_from_pdf(PDF_FILE_PATH)
                logger.info("Session state initialized successfully")
            except Exception as e:
                logger.error(f"Failed to load PDF or images: {e}")
                st.error(f"Failed to load PDF or images: {e}")
                st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized empty message history")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "image_indices" in message:
                page_num = message.get("page_num")
                if page_num in st.session_state.image_data:
                    for idx in message["image_indices"]:
                        if idx < len(st.session_state.image_data[page_num]):
                            image_bytes = st.session_state.image_data[page_num][idx]
                            try:
                                st.image(io.BytesIO(image_bytes), caption="Relevant image")
                                logger.info(f"Displayed image {idx} for page {page_num}")
                            except Exception as e:
                                logger.error(f"Failed to display image: {e}")
                                st.warning(f"Failed to display image: {e}")

    # User input
    user_input = st.chat_input("Ask anything from the PDF in any language")
    if user_input:
        logger.info(f"User query received: {user_input} (Language: {input_language})")
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            chat_history = [
    {"type": "human" if msg["role"] == "user" else "ai", "content": msg["content"]}
    for msg in st.session_state.messages  # Now includes the latest user message too
]
            initial_state = {
                "query": user_input,
                "input_language": input_language,
                "chat_history": chat_history,
                "retrieved_content": None,
                "page_num": None,
                "image_data": [],
                "augmented_content": None,
                "response": None
            }

            with st.spinner("Processing..."):
                try:
                    logger.info("Invoking agent workflow")
                    final_state = agent.invoke(initial_state)
                    answer = final_state["response"]
                    formatted_answer = format_for_display(answer)
                    message_placeholder.markdown(formatted_answer)
                    logger.info(f"Response displayed: {formatted_answer[:50]}...")

                    # Show images only if answer is relevant
                    image_indices = []
                    if "Not applicable" not in answer and final_state["image_data"]:
                        page_num = final_state["page_num"]
                        if page_num in st.session_state.image_data:
                            for idx, image_bytes in enumerate(final_state["image_data"]):
                                try:
                                    st.image(io.BytesIO(image_bytes), caption="Relevant image")
                                    image_indices.append(idx)
                                    logger.info(f"Displayed image {idx} for page {page_num}")
                                except Exception as e:
                                    logger.error(f"Failed to display image: {e}")
                                    st.warning(f"Failed to display image: {e}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_answer,
                        "page_num": final_state["page_num"],
                        "image_indices": image_indices
                    })
                    logger.info(f"Message history updated with response for page {final_state['page_num']}")

                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    error_msg = multilingual_translate(
                        f"Error processing query: {e}",
                        source_lang="English",
                        destination_language=input_language
                    )
                    st.error(error_msg)

# Run the app
if __name__ == "__main__":
    main()

    








