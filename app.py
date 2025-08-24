import streamlit as st
from mediguru.qa import LocalRAG

st.set_page_config(page_title="MediGuru (Local RAG)", layout="wide")
st.title("MediGuru — Local Medical Q&A (Gemma via Ollama)")
st.markdown(
    "> **Disclaimer:** Research prototype. Not medical advice. Always consult a clinician."
)


@st.cache_resource
def load_rag():
    return LocalRAG(index_dir="index")


rag = load_rag()

query = st.text_input("Ask a medical question", "What are the latest COPD treatments?")
top_k = st.slider("Top-K passages", 3, 8, 5)
model = st.text_input("Ollama model", "gemma3:1b")

if st.button("Search") and query.strip():
    with st.spinner("Retrieving and synthesizing answer..."):
        try:
            answer, sources = rag.answer(query, model=model, k=top_k)
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback

            st.text(traceback.format_exc())
            answer, sources = "Error occurred", []

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for s in sources:
        st.markdown(
            f"**PMID:** {s.get('pmid')} | **Journal:** {s.get('journal')} | **Score:** {s.get('score'):0.3f}"
        )
        st.write(
            (s.get("text", ""))[:600] + ("..." if len(s.get("text", "")) > 600 else "")
        )
