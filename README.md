
![LangChain](https://img.shields.io/badge/Built%20with-LangChain-121212?style=flat-square&logo=langchain)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-00ff7f?style=flat-square&logo=database&logoColor=black)
![uv](https://img.shields.io/badge/Managed%20by-uv-261230?style=flat-square&logo=uv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

# Cairo-RAG: Chatbot Tư vấn Trường Điện - Điện tử

**Cairo-RAG** là một hệ thống chatbot thông minh sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)**. Hệ thống đóng vai trò là chuyên gia tư vấn cho chương trình kỹ sư chuyên sâu tại **Trường Điện - Điện tử**, giúp giải đáp các thắc mắc dựa trên tài liệu 
nội bộ một cách chính xác.

## Tổng quan
<p align="center">
<img src="demo\images\block-diagram.png" width="824" height=""339>
</p>

- **Truy xuất kiến thức chính xác:** Sử dụng ChromaDB để truy xuất thông tin từ các tài liệu PDF đã được vector hóa, đảm bảo câu trả lời thực tế và tin cậy.
- **Xử lý ngữ cảnh thông minh:** Tích hợp `History-aware retriever` giúp AI hiểu được các câu hỏi liên quan đến lịch sử trò chuyện (ví dụ: "Ngành này học mấy năm?" sau khi đã hỏi về một ngành cụ thể).
- **Phản hồi thời gian thực (Streaming):** Hỗ trợ hiển thị câu trả lời ngay khi đang khởi tạo, tạo cảm giác tương tác mượt mà cho người dùng.
- **Embedding Model:** Sử dụng mô hình Embedding đa ngôn ngữ `paraphrase-multilingual-MiniLM-L12-v2` từ HuggingFace.
- **Quản lý bộ nhớ:** Tự động duy trì và giới hạn 10 tin nhắn gần nhất để tối ưu hóa token và hiệu suất xử lý.

## Công nghệ sử dụng

- **Framework:** [LangChain](https://python.langchain.com/)
- **LLM:** [DeepSeek AI](https://www.deepseek.com/) (`deepseek-chat`)
- **Embeddings:** HuggingFace Transformers
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Ngôn ngữ:** Python 3.13

## Cài đặt

1. **Clone repository:**
   ```bash
   git clone [https://github.com/FWD-LeTung/Cairo-RAG.git](https://github.com/FWD-LeTung/Cairo-RAG.git)
   cd Cairo-RAG
   ```
2. **Cài đặt môi trường và thư viện (sử dụng uv):**
    ```bash
    uv init
    uv venv
    source .venv/bin/activate
    uv sync
    ```
3. **Thiết lập biến môi trường:**
    ```bash
    mkdir .env
    echo "DEEPSEEK_API_KEY" > .env
    ```
4. **Demo:**
    ```bash
    uv run src/chat.py
    ```
