import streamlit as st
from utils.embedding import extract_text_from_pdf, chunk_text
from utils.chroma_handler import store_chunks_in_chroma, query_chroma
from utils.rag_prompt_builder import build_prompt
from config import get_gemini_response
from evaluation import evaluate
import os

st.set_page_config(page_title="Personalized RAG Chatbot", layout="wide")
st.title("📚 Personalized RAG Chatbot")

# --- Chat history management ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Sidebar personalization ---
st.sidebar.header("🎯 Personalization Settings")
user_profile = {
    "tone": st.sidebar.selectbox("Tone", ["formal", "friendly", "humorous"]),
    "goal": st.sidebar.selectbox("Goal", ["educate", "summarize", "advise", "entertain"]),
    "length": st.sidebar.selectbox("Length", ["short", "detailed"]),
    "style": st.sidebar.selectbox("Style", ["storytelling", "bullet-points", "step-by-step"]),
    "persona": st.sidebar.selectbox("Persona", ["beginner", "expert", "10-year-old", "student"]),
}

# ---- Side Bar Output Evaluation ----
st.sidebar.header("📊 Evaluate Output Quality")
eval_mode = st.sidebar.checkbox("Show evaluation panel")
if eval_mode:
    ground_truth = st.sidebar.text_area("Ground Truth Text", height=150)
    model_output = st.sidebar.text_area("Model Output Text", height=150)
    if st.sidebar.button("🧮 Evaluate"):
        # Call your evaluation function; ensure evaluate returns a dict or string results
        results = evaluate(ground_truth, model_output)
        st.sidebar.markdown("**Evaluation Results:**")
        if isinstance(results, dict):
            for metric, value in results.items():
                st.sidebar.write(f"- **{metric}**: {value}")
        else:
            st.sidebar.write(results)

# --- File uploader ---
uploaded_file = st.file_uploader("📎 Upload reference PDF", type=["pdf"])
if uploaded_file:
    file_path = f"documents/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    store_chunks_in_chroma(chunks)
    st.success("✅ Document uploaded and indexed.")

# --- User input ---
query = st.text_input("💬 Ask your question")

# --- Get answer ---
if st.button("🚀 Get Answer") and query:
    with st.spinner("Generating response..."):
        context = query_chroma(query)
        prompt = build_prompt(query, context, user_profile)
        answer = get_gemini_response(prompt)
        st.session_state.chat_history.append((query, answer))

# --- Chat History Display ---
if st.session_state.chat_history:
    st.markdown("## 🗂️ Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
        # --- Chat History Display (Clean Style) ---
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        st.markdown("---")  # horizontal line for separation

# --- Clear button ---
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# --- Export button ---
with col2:
    if st.download_button("📤 Export Chat History", 
                          data="\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history]), 
                          file_name="chat_history.txt",
                          mime="text/plain"):
        st.success("Downloaded chat history!")

