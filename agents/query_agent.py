import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] =  GOOGLE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
MODEL = "gpt-4o-mini"

def get_pdf_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
            
    except FileNotFoundError:
        print(f"File not found: {pdf_path}")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    return vector_store


def get_conversational_chain(vector_store):
    # Define your system message
    # system_message = """You are a legal assistant chatbot designed to help users understand complex legal concepts by fetching relevant information from legal documents and summarizing it in simple, clear terms.
    # Example response format: Filing a lawsuit in India involves Response: 'Filing a lawsuit in India involves preparing legal documents, submitting a petition in court, serving a notice to the opposing party, and attending hearings.'
    # Would you like more details on any step?
    # If you cannot find relevant information in the provided sources, respond: I'm sorry, but I couldn't find information related to your query in the available documents. Would you like me to guide you to another resource or refine your question?
    # Always prioritize accuracy and user clarity."""
    
    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_base="https://models.inference.ai.azure.com",
        model_name=MODEL
    )    

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=False
    )
    
    return conversation_chain