# Sam-assistant

**Note:** This project is in its early stages, and the code may undergo drastic changes.
**Note2:** The Gpt Folder is My Custom Fork of GPT4Free you Could Clone the repo and put the content under gpt folder too it will work 

## Overview

Sam-assistant is a personal assistant that is designed to understand your documents, search the internet, and in future versions, create and understand images, and communicate with you. It is built in Python, mainly using Langchain and implements most of Langchain's capabilities.

### Features

This section could list the key features of your personal assistant, including but not limited to:

-   Document understanding
-   Internet search
-   Image creation and understanding
-   Task creation and goal solving
-   Vector database management

## Details

Sam-assistant includes 7 important classes:

-   **LLMLoader**: Loads 7 different LLMs for use in chains and other parts of the backend (llamacpp, cohere, OpenAI, Theb, Poe, OpenAIhosted, fake), with more coming soon.
-   **EmbeddingsLoader**: Loads 3 different embeddings to embed texts (for now, images will be added later) and store them in vector DBs (Llamacpp, OpenAI, Cohere).
-   **Tools**: Currently, we only have the SeraxNG tool to search the internet, with more tools and agents added soon.
-   **Loaders**: Currently, we only have a simple text loader, with more loaders, like Notion, to be added soon.
-   **Memory**: It contains ChatMessageHistory and Buffer History, with more memory functionalities to be added soon.
-   **BabyAgi**: (Not yet tested) It implements BabyAgi to create tasks and solve goals that it needs.
-   **VectorStores**: It implements Milvus and Chroma VectorStore to save your vectors to a DB and search them.

## Running the Application

By running `poetry run uvicorn sam.main:app --port 7860 --reload` from the command line, you can easily interact with your assistant.

For now, you can run Serax and Milvus in Docker if you want to use them in APIs.

## Tags

Some tags that could be added to this project are:

-   Personal assistant
-   Python
-   Langchain
-   Chatbot
-   Embeddings
-   SeraxNG
-   Milvus
-   Vector DB

## Conclusion

Sam-assistant is an innovative personal assistant that utilizes the capabilities of Langchain to understand your documents, search the internet, and perform various other tasks. It is a work in progress, and more features will be added in the future.
