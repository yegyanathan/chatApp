import uuid
import requests
import configparser
import streamlit as st

# Load configuration
config = configparser.ConfigParser()
config.read("../config/config.ini")

# FastAPI settings
FASTAPI_HOST = config["fastapi"]["host"]
FASTAPI_PORT = config["fastapi"]["port"]
API_VERSION = config["fastapi"]["api_version"]

# API Endpoints
API_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/{API_VERSION}"

############# Initialize Session State #############

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "threads" not in st.session_state:
    st.session_state.threads = []
if "files" not in st.session_state:
    st.session_state.files = []
if "selected_file" not in st.session_state:
    st.session_state.selected_file = ""
if "enable_web_search" not in st.session_state:
    st.session_state.enable_web_search = False

############# Functions #############
    
def upload_file(uploaded_file):
    """Upload the selected file"""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response

def get_uploaded_files():
    """Fetch list of uploaded files from API."""
    response = requests.get(f"{API_URL}/files")
    return response.json().get("files", []) if response.status_code == 200 else []

def get_existing_threads():
    """Fetch existing chat threads from API."""
    response = requests.get(f"{API_URL}/conversations")
    return response.json().get("threads", []) if response.status_code == 200 else []

def get_thread_messages(thread_id):
    """Fetch chat history for a selected thread."""
    response = requests.get(f"{API_URL}/conversations/{thread_id}")
    return response.json().get("messages", []) if response.status_code == 200 else []

def delete_uploaded_file(file_name):
    """Delete uploaded file based on file name"""
    response = requests.delete(f"{API_URL}/delete/{file_name}")
    return response

############# Sidebar: File Upload Section #############

st.sidebar.header("📁 File Upload")

uploaded_file = st.sidebar.file_uploader("Upload a file")

if uploaded_file and st.sidebar.button("Upload"):
    response = upload_file(uploaded_file)
    
    if response.status_code == 200:
        st.session_state.files = get_uploaded_files()
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        st.rerun()
    else:
        st.error("❌ Failed to upload file.")

if "files" not in st.session_state or not st.session_state.files:
    st.session_state.files = get_uploaded_files()

st.session_state.selected_file = st.sidebar.radio("Select a file", ["None"] + st.session_state.files)

if st.sidebar.button("🗑️ Delete Selected File") and st.session_state.selected_file != None:
    response = delete_uploaded_file(st.session_state.selected_file) 
    if response.status_code == 200:
        st.success(f"✅ File '{st.session_state.selected_file}' deleted successfully!")
        st.session_state.files = get_uploaded_files()
        st.rerun()
    else:
        st.error("❌ Failed to delete file.")

############# Sidebar: Chat Controls #############

st.sidebar.header("💬 Chat Controls")

st.session_state.enable_web_search = st.sidebar.checkbox("Enable Web Search")

# Load existing chat threads
if not st.session_state.threads:
    st.session_state.threads = get_existing_threads()

selected_thread = st.sidebar.selectbox(
    "Select a conversation",
    ["New Conversation"] + st.session_state.threads,
    index=0
)

if selected_thread == "New Conversation":
    if st.sidebar.button("➕ Start New Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
else:
    if st.session_state.thread_id != selected_thread:
        st.session_state.thread_id = selected_thread
        st.session_state.messages = get_thread_messages(selected_thread)
        st.rerun()

############# Chat Interface #############

st.title("💬 AI Chat with LangGraph")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Type a message..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare API request
    payload = {
        "query": prompt,
        "file": "" if st.session_state.selected_file == "" else st.session_state.selected_file,
        "enable_web_search": st.session_state.enable_web_search,
    }
    response = requests.post(f"{API_URL}/conversations/{st.session_state.thread_id}/chat", json=payload)

    if response.status_code == 200:
        assistant_response = response.json().get("response", "")
        
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Save response in chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    else:
        st.error("❌ Failed to fetch response from LangGraph API.")
