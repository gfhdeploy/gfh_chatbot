import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import time
import hashlib

# Streamlit configuration
st.set_page_config(page_title="Gluten Free Harmonie AI Chat", page_icon="🫒")

# Add this near the top of your script, after the st.set_page_config() call
st.markdown("""
<script>
    document.addEventListener('DOMContentLoaded', (event) => {
        setTimeout(() => {
            document.activeElement.blur();
        }, 0);
    });
</script>
""", unsafe_allow_html=True)

# Configure Groq API
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Custom hash function for caching
def hash_documents(docs):
    return hashlib.md5(str(docs).encode()).hexdigest()

# Document ingestion and vectorization
@st.cache_data(hash_funcs={list: hash_documents})
def load_and_process_documents():
    documents = []
    doc_dir = "data/"
    
    # Load all documents at once
    for file in os.listdir(doc_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(doc_dir, file), encoding='utf-8')
            documents.extend(loader.load())
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    
    # Use a smaller, faster model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and return the vector store
    return FAISS.from_documents(texts, embeddings)

# Initialize session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Create a placeholder for the loading message
loading_placeholder = st.empty()

# Load vectorstore if not already loaded
if st.session_state.vectorstore is None:
    with loading_placeholder.container():
        with st.spinner("Initializing the chatbot..."):
            st.session_state.vectorstore = load_and_process_documents()
            time.sleep(1)  # Add a small delay to ensure the spinner is visible

# Remove the loading message
loading_placeholder.empty()

# initialize the chat history as streamlit session state of not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# streamlit page title
#st.title("SH's Virtual Assistant :sunglasses:")
st.subheader("Gluten Free Harmonie AI Chat 🫒", divider="gray")
st.subheader("", divider=False)

# display chat history
for message in st.session_state.chat_history:
    if message["role"]=='user':
        av = "💬"
    else:
        av = "🫒"
    with st.chat_message(message["role"],avatar=av):
        st.markdown(message["content"])

# input field for user's message:
user_prompt = st.chat_input("Ask a question..")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            div.embeddedAppMetaInfoBar_container__DxxL1 {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

if user_prompt:
    st.chat_message("user", avatar="💬").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Retrieve relevant documents
    relevant_docs = st.session_state.vectorstore.similarity_search(user_prompt, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Prepare messages for Groq API
    system_message = {
        "role": "system",
        "content": '''
        You are an AI agent assisting the visitors of Gluten Free Harmonie, a website specialized in Gluten Free recipes with a Moroccan-Mediterranean inspiration. You live in the Gluten Free realm. Use the following context to answer the user's question:
        '''
    }
    
    # Prepare the message list
    messages = [
        system_message,
        {"role": "user", "content": f"Context: {context}"},
    ]

    # Add chat history
    messages.extend(st.session_state.chat_history[:-1])

    # Add the current user message
    messages.append({"role": "user", "content": user_prompt})

    # Get response from Groq API
    response = client.chat.completions.create(
        messages=messages,
        model="Llama-3.1-70b-Versatile",
        temperature=0.2,
        max_tokens=1024,
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display the LLM's response
    with st.chat_message("assistant", avatar="🫒"):
        st.markdown(assistant_response)

