import os, streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent # Bản Agent chuẩn

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

def chat():
    tools = []
    for name, db in [("apple", "apple_docs.DB"), ("ms", "ms_docs.DB")]:
        if os.path.exists(db):
            idx = load_index_from_storage(StorageContext.from_defaults(persist_dir=db))
            tools.append(QueryEngineTool(query_engine=idx.as_query_engine(), 
                                         metadata=ToolMetadata(name=name, description=name)))

    agent = OpenAIAgent.from_tools(tools=tools, verbose=True)
    print("--- 🏦 READY ---")
    while (q := input("\nUSER: ")).upper() != 'QUIT':
        print(f"\nAGENT: {agent.chat(q)}")

if __name__ == "__main__": chat()