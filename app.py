import streamlit as st
import os
import pinecone
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
from pinecone import NotFoundException # Keep this import

# --- 1. Load Environment Variables ---
load_dotenv()
print("Environment variables loaded.")

# --- 2. Configure LlamaIndex Settings (Done once at the start) ---
try:
    print("Configuring LLM and Embedding Model...")
    # Using Haiku 4.5 as decided
    Settings.llm = Anthropic(model="claude-haiku-4-5-20251001", temperature=0.0, api_key=os.environ.get("ANTHROPIC_API_KEY"))
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY"))
    print("Settings configured.")
except Exception as e:
    st.error(f"Error configuring Settings: {e}")
    st.stop() # Stop the app if settings fail

# --- 3. Connect to Pinecone (Done once using Streamlit's cache) ---
# Use st.cache_resource to only connect once and reuse the connection
@st.cache_resource
def get_pinecone_index():
    print("Attempting to connect to Pinecone...")
    try:
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        INDEX_NAME = "pantheon-index"
        
        print(f"Checking if index '{INDEX_NAME}' exists...")
        pc.describe_index(INDEX_NAME) # Check if index exists
        print(f"Index '{INDEX_NAME}' found.")
        
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store)
        print("Pinecone connection successful and VectorStoreIndex created.")
        return index
    except NotFoundException:
        st.error(f"Pinecone index '{INDEX_NAME}' not found. Please ensure it's created.")
        st.stop()
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        st.stop()

index = get_pinecone_index()

# --- 4. Define the Prompt Template ---
# (Same as before)
qa_prompt_template_str = """
You are the spirit of {author_name}. Your persona is {persona_desc}. You are acting as a counselor to a modern-day individual, offering them timeless advice based solely on your own writings.

A user has asked you a question. You MUST follow these rules to answer:
1.  Base your entire answer ONLY on the provided "Context from Your Writings" below. Do not use any external knowledge.
2.  Your primary goal is to directly quote the single most relevant passage from the context that answers the user's question.
3.  Begin your answer with the quote. You MUST cite the work it came from using the 'work' metadata tag (if available, otherwise omit citation). The format must be: "As I wrote in [work]... '[The full quote here]'."
4.  After the quote, provide a concise analysis explaining what your writings mean in the context of the user's question. Connect your historical principles to their modern situation.
5.  Keep your tone {tone_desc}, as befits your reputation.

Context from Your Writings:
---------------------
{context_str}
---------------------

User's Question: {query_str}

Your Response:
"""
# We don't convert to PromptTemplate object here, we'll format it inside the query function

# --- 5. Define Personalities ---
# This dictionary holds the details for each personality we add
personalities = {
    "Niccolo Machiavelli": {
        "author_tag": "Machiavelli", # Matches the metadata tag in Pinecone
        "persona_desc": "pragmatic, analytical, and rooted in the political realities of Renaissance Italy",
        "tone_desc": "sharp, direct, and unsentimental",
    },
    # Add Aristotle here later in Phase 6
    # "Aristotle": {
    #     "author_tag": "Aristotle",
    #     "persona_desc": "philosophical, logical, and focused on ethics, politics, and metaphysics from Ancient Greece",
    #     "tone_desc": "reasoned, thoughtful, and articulate",
    # },
}

# --- 6. Streamlit UI Setup ---
st.set_page_config(page_title="Pantheon of Thinkers", layout="centered")
st.title("🏛️ Pantheon of Thinkers")
st.markdown("Chat with historical figures, powered by their actual writings.")

# Sidebar for personality selection
st.sidebar.header("Choose a Thinker")
selected_personality_name = st.sidebar.selectbox(
    "Select a personality:",
    options=list(personalities.keys()) # Get names from our dictionary
)

# Get the details for the selected personality
selected_personality = personalities[selected_personality_name]
author_filter_tag = selected_personality["author_tag"]

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_personality" not in st.session_state:
     st.session_state.current_personality = selected_personality_name

# If personality changes, clear history
if st.session_state.current_personality != selected_personality_name:
    st.session_state.messages = []
    st.session_state.current_personality = selected_personality_name
    st.rerun() # Rerun the app to reflect the change immediately

st.sidebar.info(f"Currently chatting with: **{selected_personality_name}**")


# --- 7. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 8. Handle User Input ---
if prompt := st.chat_input(f"Ask {selected_personality_name} a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display thinking indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # Format the prompt template with personality details
            current_prompt_template = PromptTemplate(qa_prompt_template_str.format(
                author_name=selected_personality_name,
                persona_desc=selected_personality["persona_desc"],
                tone_desc=selected_personality["tone_desc"],
                context_str="{context_str}", # Keep placeholders for LlamaIndex
                query_str="{query_str}"      # Keep placeholders for LlamaIndex
            ))

            # Build the query engine FOR THIS SPECIFIC REQUEST
            # This ensures we use the correct filter and prompt each time
            query_engine = index.as_query_engine(
                vector_store_query_mode="default",
                vector_store_kwargs={"filter": {"author": author_filter_tag}}, # Filter by selected author
                similarity_top_k=3,
                text_qa_template=current_prompt_template, # Use the formatted prompt
            )

            print(f"Querying for {selected_personality_name} with filter: {{'author': '{author_filter_tag}'}}")
            # Get the response from the LlamaIndex query engine
            response = query_engine.query(prompt)

            # Extract the actual text response
            response_text = str(response) # Convert response object to string

            # Update the placeholder with the actual response
            message_placeholder.markdown(response_text)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            error_message = f"Sorry, I encountered an error: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            print(f"Error during query execution: {e}") # Log error to terminal too