from agents.summarization_agent import conversation_chain
from langchain.schema import SystemMessage, HumanMessage
import gradio as gr


def chat(message, history):
    conversation = conversation_chain()
    result = conversation.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)



