import streamlit as st, os, platform, pytesseract
from pdf2image import convert_from_path
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Universal Financial AI", layout="wide")
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

st.title("📊 Universal Financial AI Agent")

up_files = st.file_uploader("Tải báo cáo PDF:", type="pdf", accept_multiple_files=True)

if up_files:
    all_docs = []
    with st.spinner("🤖 Đang quét OCR chất lượng cao (400 DPI)..."):
        for f in up_files:
            path = f"temp_{f.name}"
            with open(path, "wb") as tmp: tmp.write(f.getbuffer())
            imgs = convert_from_path(path, dpi=400, first_page=1, last_page=15)
            text = "\n".join([pytesseract.image_to_string(img, lang='vie', config='--psm 1') for img in imgs])
            all_docs.append(Document(text=text, metadata={"file": f.name}))
    
    engine = VectorStoreIndex.from_documents(all_docs).as_query_engine(similarity_top_k=10)
    tool = QueryEngineTool(query_engine=engine, metadata=ToolMetadata(name="scanner", description="Financial Data"))
    
    # Khởi tạo Agent và sửa lỗi update_prompts
    agent = OpenAIAgent.from_tools(tools=[tool], verbose=True)
    
    system_msg = (
        "Bạn là chuyên gia tài chính. Hãy dùng công cụ scanner để tìm số liệu. "
        "Luôn phân biệt rõ cột 'Hợp nhất' và 'Riêng lẻ'. Trả lời chính xác số liệu tìm được."
    )
    # Cấu hình Prompt chuẩn để tránh lỗi AttributeError
    agent.update_prompts({"agent_worker:system_prompt": ChatPromptTemplate.from_str(system_msg)})

    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Hỏi về số liệu..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            res = agent.chat(prompt)
            st.write(str(res))
            st.session_state.messages.append({"role": "assistant", "content": str(res)})