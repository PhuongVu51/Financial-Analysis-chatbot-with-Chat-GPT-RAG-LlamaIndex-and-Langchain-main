import streamlit as st, os, platform, pytesseract, shutil
from pdf2image import convert_from_path
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

st.set_page_config(page_title="Financial AI Agent", layout="wide")

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Lấy Key từ Secrets
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

st.title("📊 Universal Financial AI Agent")

# Sidebar
with st.sidebar:
    model_choice = st.selectbox("Mô hình AI:", ["gpt-4o", "gpt-3.5-turbo"])
    if st.button("🗑️ Làm mới bộ nhớ"):
        if os.path.exists("temp_dir"): shutil.rmtree("temp_dir")
        st.rerun()

# 1. Xử lý File
up_files = st.file_uploader("Tải lên PDF:", type="pdf", accept_multiple_files=True)

if up_files:
    if not os.path.exists("temp_dir"): os.makedirs("temp_dir")
    docs = []
    with st.spinner("🤖 Đang quét OCR..."):
        for f in up_files:
            path = os.path.join("temp_dir", f.name)
            with open(path, "wb") as tmp: tmp.write(f.getbuffer())
            # Quét trang 1-15 (DPI 400 để chính xác)
            pages = convert_from_path(path, dpi=400, first_page=1, last_page=15)
            content = "\n".join([pytesseract.image_to_string(p, lang='vie', config='--psm 1') for p in pages])
            docs.append(Document(text=content))

    # 2. Khởi tạo Agent (Bản sửa lỗi triệt để)
    engine = VectorStoreIndex.from_documents(docs).as_query_engine(similarity_top_k=10)
    tool = QueryEngineTool(query_engine=engine, metadata=ToolMetadata(name="scanner", description="Data"))
    
    # Khởi tạo LLM trước để không bị NameError
    llm = OpenAI(model=model_choice, temperature=0)
    
    # Dùng OpenAIAgent và truyền system_prompt trực tiếp
    agent = OpenAIAgent.from_tools(
        tools=[tool], 
        llm=llm, 
        system_prompt="Bạn là chuyên gia tài chính. Hãy dùng 'scanner' để tra cứu số liệu.",
        verbose=True
    )

    # 3. Giao diện Chat
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Nhập câu hỏi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            res = agent.chat(prompt)
            st.write(str(res))
            st.session_state.messages.append({"role": "assistant", "content": str(res)})