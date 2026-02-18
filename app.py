import streamlit as st
import time
import logging
import warnings
from rag_chromadb import RAGEngine

# Aggressive silence for terminal
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Page Config
st.set_page_config(page_title="ArgusHR Policy Bot", page_icon="🛡️")

st.title("🛡️ ArgusHR Policy Assistant")
st.markdown("Ask questions about company policies, benefits, and regulations.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    prompt_version = st.selectbox(
        "Prompt Version",
        ("v2", "v1"),
        index=0,
        help="v2: Improved Structured Prompt\nv1: Baseline Prompt"
    )
    st.info(f"Using: **{prompt_version.upper()}** Prompt")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG Engine (Cached)
@st.cache_resource
def get_engine():
    return RAGEngine()

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to initialize RAG Engine: {e}")
    st.stop()

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What is the policy on..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            try:
                # Query RAG Engine
                result = engine.query(prompt, prompt_version=prompt_version)
                answer = result['answer']
                sources = result['sources']
                
                # Format Sources
                source_text = "\n\n**Sources:**\n"
                for s in sources:
                    meta = s['metadata']
                    source_name = meta.get('source_name', 'Unknown')
                    # Add extra info like page/sheet if available
                    extra = ""
                    if 'page' in meta: extra = f" (Page {meta['page']})"
                    elif 'sheet' in meta: extra = f" (Sheet {meta['sheet']})"
                    
                    source_text += f"- {source_name}{extra}\n"
                
                # Combine answer and sources if not already in answer (V2 puts sources in answer)
                if prompt_version == "v1":
                    full_response = answer + source_text
                else:
                    full_response = answer # V2 includes sources in its structured output
                
                # Simulate typing effect
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Error: {e}")
                full_response = "I encountered an error processing your request."
                message_placeholder.markdown(full_response)
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
