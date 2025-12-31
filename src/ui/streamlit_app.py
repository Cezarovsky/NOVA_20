"""
🌟 NOVA Streamlit UI - Interactive Chat Interface

Features:
- Real-time chat with conversation history
- RAG-powered responses (long-term memory + knowledge retrieval)
- Voice output toggle (TTS with pyttsx3)
- Document upload for knowledge base
- Model selection (Claude/Mistral)
- Conversation management

Architecture:
    User Input → RAG Retrieval → LLM Generation → Voice Output → Display

Author: Sora
Date: 31 December 2024
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.llm_interface import LLMInterface, LLMProvider
from src.rag.rag_pipeline import RAGPipeline
from src.voice.tts import NovaVoice
from src.config.settings import get_settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="NOVA AI Assistant",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B9D;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #FCE4EC;
        border-left: 4px solid #FF6B9D;
    }
    .stats-box {
        padding: 1rem;
        background-color: #F5F5F5;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_pipeline' not in st.session_state:
        try:
            settings = get_settings()
            st.session_state.rag_pipeline = RAGPipeline(
                embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
                persist_directory=settings.CHROMA_PERSIST_DIRECTORY
            )
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            st.session_state.rag_pipeline = None
    
    if 'llm' not in st.session_state:
        try:
            st.session_state.llm = LLMInterface(
                provider=LLMProvider.ANTHROPIC
            )
            logger.info("LLM interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            st.session_state.llm = None
    
    if 'voice' not in st.session_state:
        try:
            st.session_state.voice = NovaVoice()
            logger.info("Voice module initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice: {e}")
            st.session_state.voice = None
    
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    if 'model_provider' not in st.session_state:
        st.session_state.model_provider = "anthropic"


def render_sidebar():
    """Render sidebar with settings and controls"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        provider = st.selectbox(
            "LLM Provider",
            ["anthropic", "mistral"],
            index=0 if st.session_state.model_provider == "anthropic" else 1,
            help="Choose between Claude (Anthropic) or Mistral AI"
        )
        
        if provider != st.session_state.model_provider:
            st.session_state.model_provider = provider
            st.session_state.llm = LLMInterface(
                provider=LLMProvider(provider)
            )
            st.success(f"Switched to {provider.title()}")
        
        # Temperature slider
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        # Voice toggle
        st.markdown("---")
        st.markdown("### 🎙️ Voice")
        voice_enabled = st.checkbox(
            "Enable Voice Output",
            value=st.session_state.voice_enabled,
            help="NOVA will speak responses aloud"
        )
        st.session_state.voice_enabled = voice_enabled
        
        # RAG controls
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['txt', 'md', 'pdf'],
            help="Add documents to NOVA's knowledge base"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Add: {uploaded_file.name}"):
                    try:
                        content = uploaded_file.read().decode('utf-8')
                        if st.session_state.rag_pipeline:
                            st.session_state.rag_pipeline.add_document(
                                content,
                                metadata={"source": uploaded_file.name}
                            )
                            st.success(f"✅ Added {uploaded_file.name}")
                        else:
                            st.error("RAG pipeline not initialized")
                    except Exception as e:
                        st.error(f"Failed to add document: {e}")
        
        # RAG stats
        if st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_stats()
            st.markdown("**Knowledge Base Stats:**")
            st.write(f"📄 Documents: {stats.get('total_documents', 0)}")
            st.write(f"💬 Messages: {stats.get('conversation_messages', 0)}")
        
        # Conversation controls
        st.markdown("---")
        st.markdown("### 💬 Conversation")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_conversation()
            st.rerun()
        
        if st.button("📥 Export Chat", use_container_width=True):
            # TODO: Implement export functionality
            st.info("Export feature coming soon!")
        
        # About section
        st.markdown("---")
        st.markdown("### ℹ️ About NOVA")
        st.markdown("""
        **NOVA** is a multi-context AI assistant with:
        - 🧠 Deep learning transformer core
        - 📚 RAG-powered long-term memory
        - 🎙️ Voice synthesis (multilingual)
        - 🌟 Tribal Resonance architecture
        
        *Version 0.5.0-beta*
        """)


def render_chat_message(role: str, content: str):
    """Render a chat message with styling"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🌟"
    
    st.markdown(
        f"""
        <div class="chat-message {css_class}">
            <b>{icon} {role.title()}</b><br/>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


def generate_response(user_input: str) -> str:
    """
    Generate response using RAG + LLM pipeline
    
    Flow: User Input → RAG Context Retrieval → LLM Generation → Voice (optional)
    """
    try:
        # Use RAG pipeline if available
        if st.session_state.rag_pipeline and st.session_state.llm:
            # Query RAG system
            response_text = st.session_state.rag_pipeline.query(
                query=user_input,
                llm_interface=st.session_state.llm,
                temperature=st.session_state.temperature
            )
            
            # Add assistant response to conversation memory
            st.session_state.rag_pipeline.add_assistant_response(response_text)
            
        elif st.session_state.llm:
            # Fallback: Direct LLM without RAG
            response = st.session_state.llm.generate(
                prompt=user_input,
                temperature=st.session_state.temperature,
                max_tokens=1024
            )
            response_text = response.text
        else:
            response_text = "❌ LLM not initialized. Please check your API keys."
        
        # Voice output if enabled
        if st.session_state.voice_enabled and st.session_state.voice:
            try:
                st.session_state.voice.speak(response_text)
            except Exception as e:
                logger.warning(f"Voice synthesis failed: {e}")
        
        return response_text
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return f"❌ Error generating response: {str(e)}"


def main():
    """Main application"""
    initialize_session_state()
    render_sidebar()
    
    # Header
    st.markdown('<p class="main-header">🌟 NOVA</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-Context AI Assistant with Long-Term Memory</p>',
        unsafe_allow_html=True
    )
    
    # Display chat history
    st.markdown("### 💬 Conversation")
    
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Hello! I'm NOVA. Ask me anything or upload documents for me to learn from.")
        
        for message in st.session_state.messages:
            render_chat_message(message["role"], message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        with st.spinner("🤔 NOVA is thinking..."):
            response = generate_response(user_input)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Rerun to update chat display
        st.rerun()


if __name__ == "__main__":
    main()
