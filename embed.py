import os, streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Tự lấy Key từ môi trường hoặc secrets để bảo mật
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

def run():
    for f in ["apple.pdf", "microsoft.pdf"]:
        if os.path.exists(f):
            idx = VectorStoreIndex.from_documents(SimpleDirectoryReader(input_files=[f]).load_data())
            idx.storage_context.persist(persist_dir=f"{f.split('.')[0]}_docs.DB")
            print(f"✅ Xong: {f}")

if __name__ == "__main__": run()