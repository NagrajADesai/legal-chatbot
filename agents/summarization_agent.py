from agents.query_agent import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
import os

pdf1 = os.path.abspath("data/Legal_compliance_corporate_laws.pdf")
pdf2 = os.path.abspath("data/Guide-to-Litigation-in-India.pdf")

# use global to avoid initializing converation chain each time
_conversation_chain = None

def initialize_conversation_chain():
    global _conversation_chain
    
    if _conversation_chain is None:
        print("Initializing conversation chain...")
        # Process PDFs
        text1 = get_pdf_text(pdf1)
        text2 = get_pdf_text(pdf2)
        raw_text = text1 + text2
        
        # Create text chunks and vector store
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        
        # Create conversation chain
        _conversation_chain = get_conversational_chain(vector_store)
        print("Conversation chain initialized.")
    
    return _conversation_chain

def conversation_chain():
    return initialize_conversation_chain()