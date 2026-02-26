import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load API Key
load_dotenv()

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_db = Chroma(
        persist_directory="vector_db/",
        embedding_function=embeddings
    )

    #LLM DeepSeek
    llm = ChatDeepSeek(
        model='deepseek-chat', 
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.1 
    )

    #Prompt Template
    template = """Bạn là một trợ lý ảo tư vấn chương trình kỹ sư chuyên sâu. 
    Hãy sử dụng các đoạn ngữ cảnh sau đây để trả lời câu hỏi. 
    Nếu không biết câu trả lời, hãy nói là bạn không biết, đừng cố tự tạo ra câu trả lời.
    Trả lời bằng tiếng Việt một cách chuyên nghiệp.

    Ngữ cảnh: {context}

    Câu hỏi: {question}

    Trả lời:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    #RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Nhồi context vào prompt
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Lấy top 3 đoạn liên quan nhất
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain

if __name__ == "__main__":
    chain = get_rag_chain()
    query = "Chương trình đào tạo gồm bao nhiêu tín chỉ?"
    response = chain.invoke(query)
    print(f"\nChatbot trả lời:\n{response['result']}")