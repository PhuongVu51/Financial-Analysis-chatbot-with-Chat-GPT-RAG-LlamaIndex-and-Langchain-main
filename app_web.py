import streamlit as st
import os
import shutil
import platform
import pytesseract
import pandas as pd # Bổ sung để xử lý dữ liệu biểu đồ
import json
import re
from pdf2image import convert_from_path
import pypdf
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & MÔI TRƯỜNG
# ==========================================
st.set_page_config(page_title="Agentic Financial AI - Group 08", layout="wide", page_icon="📈")

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ Thiếu OPENAI_API_KEY trong Secrets!")
    st.stop()

st.title("📈 Agentic Financial AI & Visualization")
st.markdown("---")

# ==========================================
# 2. THANH SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.write("**Leader:** Phương Vũ")
    model_choice = st.selectbox("LLM Model:", ["gpt-4o", "gpt-3.5-turbo"])
    
    if st.button("🗑️ Clear Cache & Temp Files"):
        if os.path.exists("temp_dir"):
            shutil.rmtree("temp_dir")
            st.success("Cache cleared!")
            st.rerun()

# ==========================================
# 3. AGENTIC PIPELINE (OCR & MULTI-DOC)
# ==========================================
uploaded_files = st.file_uploader("Upload Financial Reports (PDF):", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")
    
    individual_tools = []
    
    with st.spinner("🤖 Agent is processing documents with High-DPI OCR..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_dir", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Sử dụng DPI 400 để đảm bảo Numerical Accuracy cho biểu đồ
            pages = convert_from_path(file_path, dpi=400, first_page=8, last_page=10)
            content = ""
            for i, page in enumerate(pages):
                page_text = pytesseract.image_to_string(page, lang='vie', config='--psm 1')
                content += f"\n--- Source: {uploaded_file.name} - Page {i+8} ---\n{page_text}"

            doc = Document(text=content, metadata={"file_name": uploaded_file.name})
            index = VectorStoreIndex.from_documents([doc])
            
            engine = index.as_query_engine(similarity_top_k=5)
            tool = QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name=f"tool_{uploaded_file.name.replace('.', '_').replace(' ', '_')}",
                    description=f"Provides financial data from {uploaded_file.name}"
                )
            )
            individual_tools.append(tool)

    llm = OpenAI(model=model_choice, temperature=0)
    
    sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_tools,
        llm=llm,
        verbose=True
    )

    master_tool = QueryEngineTool(
        query_engine=sub_query_engine,
        metadata=ToolMetadata(
            name="multi_financial_analyzer",
            description="Use for data extraction and multi-doc comparisons."
        )
    )

    # Cập nhật đoạn khởi tạo agent
    agent = OpenAIAgent.from_tools(
        tools=[master_tool], 
        llm=llm, 
        verbose=True,
        system_prompt=(
            "Bạn là một chuyên gia phân tích tài chính có khả năng lập luận Agentic. "
            "KHI SO SÁNH: Bạn phải gọi công cụ 'multi_financial_analyzer' để lấy số liệu từ TẤT CẢ các file liên quan. "
            "Nếu không thấy số liệu ngay lập tức, hãy thử tìm các từ khóa tương đương (ví dụ: 'Total Assets' và 'Tổng cộng tài sản'). "
            "BẮT BUỘC: Khi được yêu cầu vẽ biểu đồ, hãy trả về dữ liệu số dưới dạng JSON ngay cả khi bạn không chắc chắn 100%, kèm theo nguồn trang."
        )
)

    # ==========================================
    # 4. CHAT INTERFACE & VISUALIZATION LOGIC
    # ==========================================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ví dụ: So sánh tổng tài sản qua các năm và vẽ biểu đồ cột"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Agentic reasoning..."):
                try:
                    # 1. Trả lời văn bản
                    response = agent.chat(prompt)
                    st.markdown(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})

                    # 2. Logic tự động vẽ biểu đồ
                    if any(word in prompt.lower() for word in ["vẽ", "biểu đồ", "chart", "graph", "plot"]):
                        st.divider()
                        # Agent trích xuất JSON dữ liệu
                        data_msg = agent.chat(
                            "Extract the metrics and values from your previous answer into a raw JSON format: "
                            "{'Metric': ['A', 'B'], 'Value': [100, 200]}. Use the same language as the user query for labels. "
                            "Return ONLY the JSON block."
                        )
                        
                        # Làm sạch và chuyển đổi sang DataFrame
                        json_match = re.search(r'\{.*\}', data_msg.response, re.DOTALL)
                        if json_match:
                            data_dict = json.loads(json_match.group())
                            df = pd.DataFrame(data_dict)
                            
                            st.subheader("📊 Data Visualization")
                            # Hiển thị biểu đồ cột
                            st.bar_chart(df.set_index(df.columns[0]))
                            # Hiển thị bảng dữ liệu đi kèm để đối chiếu
                            st.table(df)
                            
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("👋 Hello Phương Vũ! Upload reports to activate the Agentic Visualization System.")