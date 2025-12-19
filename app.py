# app.py
import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# --- Caching the Embedding Model ---
# This decorator ensures the model is loaded only once, improving performance.
@st.cache_resource
def load_embeddings():
    """Load the HuggingFace embeddings model."""
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# --- Core RAG Chain Creation Function ---
def create_rag_chain(pdf_file, embeddings):
    """
    Takes an uploaded PDF file, processes it, and returns a RAG chain.
    """
    # PyPDFLoader needs a file path, so we save the uploaded file to a temporary location.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    # 1. Load the document
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    # Clean up the temporary file
    os.remove(tmp_file_path)

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # 3. Create an in-memory FAISS vector store from the chunks
    vector_store = FAISS.from_documents(docs, embeddings)

    # 4. Create the retriever
    retriever = vector_store.as_retriever()

    # 5. Define the prompt template for the LLM
    template = """
    Use the following context to answer the question.
    If you don't know the answer, just say you don't know.
    Keep the answer concise.

    Context: {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # 6. Initialize the LLM (using a local Ollama model like Llama 3)
    # Make sure Ollama is running: ollama serve
    llm = Ollama(model="llama3")

    # 7. Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    return qa_chain


# --- Streamlit Application UI ---
st.set_page_config(page_title="Chat with Your PDF")
st.title("Chat With Your PDF (Locally!)")

# Load embeddings once at the start
embeddings = load_embeddings()

# Use session state to store the RAG chain across reruns
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar for file uploading and processing
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing your document... This may take a moment."):
                # Create the RAG chain and store it in the session state
                st.session_state.rag_chain = create_rag_chain(uploaded_file, embeddings)
            st.success("Document processed! You can now ask questions.")
        else:
            st.warning("Please upload a PDF file first.")

# Main chat interface
st.header("Ask Questions About Your Document")

if st.session_state.rag_chain is None:
    st.info("Please upload and process a document in the sidebar to start chatting.")
else:
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Thinking..."):
            result = st.session_state.rag_chain.invoke({"query": question})
            
            # Display the answer
            st.subheader("Answer:")
            st.write(result["result"])
            
            # (Optional) Display source documents for verification
            with st.expander("Show Sources"):
                for doc in result["source_documents"]:
                    st.write("---")
                    st.write(f"**Source (Page {doc.metadata.get('page', 'N/A')}):**")
                    st.write(doc.page_content)
