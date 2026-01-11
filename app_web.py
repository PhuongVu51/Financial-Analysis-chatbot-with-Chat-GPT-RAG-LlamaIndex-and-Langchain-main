import streamlit as st, os, platform, pytesseract, shutil
from pdf2image import convert_from_path
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

# 1. CẤU HÌNH HỆ THỐNG
st.set_page_config(page_title="Universal Financial AI", layout="wide")

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Lấy Key bảo mật từ Streamlit Secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ Thiếu OPENAI_API_KEY trong cấu hình Secrets!")
    st.stop()

st.title("📊 Universal Financial AI Agent")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_choice = st.selectbox("Mô hình:", ["gpt-4o", "gpt-3.5-turbo"])
    if st.button("🗑️ Làm mới"):
        if os.path.exists("temp_dir"): shutil.rmtree("temp_dir")
        st.rerun()

# 2. XỬ LÝ DỮ LIỆU (OCR 400 DPI)
up_files = st.file_uploader("Tải báo cáo PDF:", type="pdf", accept_multiple_files=True)

if up_files:
    if not os.path.exists("temp_dir"): os.makedirs("temp_dir")
    all_docs = []
    
    with st.spinner("🤖 Đang quét dữ liệu..."):
        for f in up_files:
            path = os.path.join("temp_dir", f.name)
            with open(path, "wb") as tmp: tmp.write(f.getbuffer())
            
            # Quét trang 1-20 để bao quát các bảng quan trọng
            pages = convert_from_path(path, dpi=400, first_page=1, last_page=20)
            text_content = ""
            for i, page in enumerate(pages):
                text_content += f"\n--- TRANG {i+1} ---\n" + pytesseract.image_to_string(page, lang='vie', config='--psm 1')
            
            all_docs.append(Document(text=text_content, metadata={"file": f.name}))

    # 3. KHỞI TẠO CÔNG CỤ TRUY VẤN
    index = VectorStoreIndex.from_documents(all_docs)
    query_engine = index.as_query_engine(similarity_top_k=10)
    
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(name="financial_scanner", description="Truy xuất dữ liệu tài chính.")
    )

    # 4. KHỞI TẠO AGENT (SỬA LỖI ATTRIBUTERROR & NAMEERROR)
    # Khởi tạo LLM trước để tránh NameError
    llm = OpenAI(model=model_choice, temperature=0)
    
    # Dùng OpenAIAgent để ổn định nhất trên Cloud
    system_prompt = (
        "Bạn là chuyên gia tài chính. Hãy dùng 'financial_scanner' để tra cứu. "
        "Luôn phân biệt cột Hợp nhất/Riêng lẻ và thời điểm báo cáo."
    )
    
    agent = OpenAIAgent.from_tools(
        tools=[tool], 
        llm=llm, 
        system_prompt=system_prompt, 
        verbose=True
    )

    # 5. GIAO DIỆN CHAT
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Hỏi AI..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            try:
                res = agent.chat(prompt)
                st.write(str(res))
                st.session_state.messages.append({"role": "assistant", "content": str(res)})
            except Exception as e:
                st.error(f"Lỗi: {e}")