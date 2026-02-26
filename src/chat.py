import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_db = Chroma(persist_directory="vector_db/", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = ChatDeepSeek(
    model='deepseek-chat', 
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.1
)

# context prompt
contextualize_q_system_prompt = (
    "Sử dụng lịch sử trò chuyện và câu hỏi mới nhất của người dùng "
    "để tạo ra một câu hỏi độc lập có thể hiểu được mà không cần lịch sử."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

#Prompt
system_prompt = (
    "Bạn là chuyên gia tư vấn chương trình kỹ sư chuyên sâu Trường Điện - Điện tử. "
    "Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi. "
    "Nếu không biết, hãy nói không biết. Trả lời chuyên nghiệp bằng tiếng Việt."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- BƯỚC 4: Vòng lặp Chat liên tục ---
def start_chat():
    chat_history = [] # Nơi lưu trữ lịch sử tạm thời trong phiên làm việc
    print("\n🤖 BMO: Chào bạn! Tôi đã sẵn sàng tư vấn về chương trình kỹ sư. (Gõ 'exit' để thoát)")
    
    while True:
        user_input = input("\n👤 Bạn: ")
        if user_input.lower() in ["exit", "quit", "thoát"]:
            print("🤖 BMO: Tạm biệt! Hẹn gặp lại bạn.")
            break
            
        if not user_input.strip():
            continue

        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        answer = response["answer"]
        print(f"🤖 BMO: {answer}")

        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer)
        ])
        
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

if __name__ == "__main__":
    start_chat()