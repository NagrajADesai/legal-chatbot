# Legal Chatbot

A multi-agent chatbot designed to fetch legal information from PDF documents, summarize complex legal concepts, and provide clear, concise answers to users. Built with Python and Gradio, this project streamlines legal research and makes legal knowledge more accessible.

## Project Structure

- **`query_agent.py`**: Contains all the necessary functions to query and retrieve relevant information from legal documents.
- **`summarization_agent.py`**: Combines the querying functions and handles summarization to simplify complex legal texts.
- **`app.py`**: The main application file, built using Gradio, to create a user-friendly interface for interacting with the chatbot.

## Technology Stack

- **Retrieval-Augmented Generation (RAG)**: Used for extracting relevant text from PDF documents.
- **FAISS**: Utilized to store and efficiently search through the document embeddings.

## How to Run the Project

Make sure you have Python installed, along with the required libraries. Then, simply run the following command:

```bash
python app.py
```

This will start the Gradio interface, allowing you to ask questions!

## Requirements

Install the necessary libraries with:

```bash
pip install -r requirements.txt
```

## Usage

1. Launch the app by running `python app.py`.
2. Enter your query in the chatbot interface.
3. Receive summarized, easy-to-understand answers to your legal questions.

## Future Improvements

- Support for multiple document types (e.g., Word, HTML).
- Enhanced natural language understanding with fine-tuned models.
- Caching and indexing for faster query responses.
