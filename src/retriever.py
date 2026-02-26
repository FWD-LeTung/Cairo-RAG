import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def test_retrieval(query):
    # Initialize Embedding Model
    print("Đang nạp Embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    #connect vectorDB
    if not os.path.exists("vector_db/"):
        print("Không tìm thấy thư mục vector_db. Hãy chạy ingestion.py trước!")
        return

    vector_db = Chroma(
        persist_directory="vector_db/",
        embedding_function=embeddings
    )

    #Querry
    print(f"Đang tìm kiếm thông tin cho: '{query}'...")
    results = vector_db.similarity_search_with_score(query, k=3)

    #Result
    print("\n" + "="*50)
    print("KẾT QUẢ TRUY XUẤT:")
    print("="*50)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n[{i+1}] Độ tương đồng: {score:.4f}")
        print(f"Nguồn: {doc.metadata.get('source', 'N/A')} - Trang: {doc.metadata.get('page', 'N/A')}")
        print(f"Nội dung: {doc.page_content[:200]}...")
        print("-" * 30)

if __name__ == "__main__":
    user_query = "Thời gian đào tạo hệ kỹ sư chuyên sâu là bao lâu?"
    test_retrieval(user_query)