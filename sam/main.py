import copy
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain import Cohere
from langchain.chains import RetrievalQA
from pydantic import BaseModel

from sam.core.chains.chains import create_chain
from sam.core.llms.llms import EmbeddingLoader, LLMLoader
from sam.core.loaders.loaders import Loaders
from sam.core.prompts.prompts import create_prompt_icl_qa
from sam.core.vectorstore.vectorstores import VectoreStores

load_dotenv()

# create the app
app = FastAPI()

# connection_args = {"host": "127.0.0.1", "port": "19530"}
# sysctl -w vm.max_map_count=262144


@app.get("/")
async def hello():
    return {"hello": "wooooooorld"}


class MessageRequest(BaseModel):
    message: str


@app.post("/lang")
async def lang(request: Request, message_request: MessageRequest):
    # model = LLMLoader("poe", token="KaEMfvDPEXoS115jzAFRRg%3D%3D", model_name="gpt-3.5-turbo" | "sage")
    # model = LLMLoader("cohere", top_k=40, top_p=0.95, model="command-xlarge-nightly" | "command-nightly" | "command-light-nightly")
    # model = LLMLoader("llamacpp")
    # model = LLMLoader("theb").load_model()
    model = LLMLoader("cohere", top_k=40, top_p=0.95, model="command-xlarge-nightly")
    prompt = create_prompt_icl_qa()
    cohere_api = os.environ.get("COHERE_API_KEY")
    result = create_chain(
        model,
        prompt,
        message_request.message,
    )
    return {"answer": result}


@app.get("/embed")
async def embed():
    # embedding = EmbeddingLoader(
    #     "openai", model_name="text-embedding-ada-002", n_ctx=8191).load_embeddings()
    # embedding = EmbeddingLoader("cohere", model_name="large").load_embeddings()
    embeddings = EmbeddingLoader("cohere").load_embeddings()
    text = "Cohere allows you to implement language AI into your product. Get started and explore Cohere's capabilities with the Playground or Quickstart tutorials."
    query_result = embeddings.embed_query(text)
    return {"answer": query_result}


@app.get("/vector/load")
async def vector_load():
    path = rf"D:\AI\chat-your-data\state_of_union.txt"
    embeddings = EmbeddingLoader("cohere").load_embeddings()
    # vectordb = VectoreStores(embeddings, db_type="milvus", connection_args=connection_args)
    vectordb = VectoreStores(embeddings, db_type="chroma")
    file_text = Loaders.load_file(path)
    docs = Loaders.split_docs(file_text)
    vectordb.add_documents(docs)
    return {"status": 200}


@app.get("/vector/advanced-search")
async def vector_ad_search():
    embeddings = EmbeddingLoader("cohere").load_embeddings()
    vectordb = VectoreStores(embeddings, db_type="chroma")
    retriever = vectordb.as_retriever()
    qa = vectordb.embedding_filter(retriever)
    text = "What did the president say about Ketanji Brown Jackson ?"
    docs = qa.get_relevant_documents(text)
    vectordb.pretty_print_docs(docs)
    return {"answer": docs}


@app.get("/vector/search")
async def search():
    embeddings = EmbeddingLoader("cohere").load_embeddings()
    vector_db = VectoreStores(embeddings, db_type="chroma")

    query = "What did the president say about Ketanji Brown Jackson"
    docs = vector_db.search_text(query)

    retriever = vector_db.as_retriever()
    model = LLMLoader("theb").load_model()
    compressed = vector_db.compress_retriever(model, retriever)
    comp = compressed.get_relevant_documents(query)
    return {"all": docs, "compressed": comp}


@app.get("/vector/qa")
async def qa():
    model = LLMLoader("theb").load_model()
    embeddings = EmbeddingLoader("cohere").load_embeddings()
    vector_db = VectoreStores(embeddings, db_type="chroma")
    retriever = vector_db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)
    query = "What did the president say about Ketanji Brown Jackson"

    response = qa.run(query)

    return {"response": response}


@app.get("/agents/1")
async def agents_1():
    model = LLMLoader("theb").load_model()
    tools = [Tool()]
    return {"response": response}


# @app.get("/llama")
# async def llama(request: Request):
#     stream = llm(
#         "name 10 footballer players in alphabatic order :",
#     )

#     async def async_generator():
#         for item in stream:
#             print(copy.deepcopy(item)["choices"][0]["text"])
#             yield item

#     async def server_sent_events():
#         async for item in async_generator():
#             if await request.is_disconnected():
#                 break

#             result = copy.deepcopy(item)
#             text = result["choices"][0]["text"]

#             yield {"data": text}

#     return EventSourceResponse(server_sent_events())


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=7860)
