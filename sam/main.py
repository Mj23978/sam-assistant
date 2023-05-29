from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain.agents import AgentType, initialize_agent
from langchain.chains import RetrievalQA
from pydantic import BaseModel

from sam.core.agents.babyagi.babyagi import BabyAGI
from sam.core.chains.chains import Chains
from sam.core.llms.llms import EmbeddingLoader, LLMLoader
from sam.core.loaders.loaders import CustomOutputParser, Loaders
from sam.core.prompts.prompts import create_prompt_icl_qa
from sam.core.tools.tools import Tools
from sam.core.utils import logger
from sam.core.vectorstore.vectorstores import VectoreStores

load_dotenv()

# create the app
app = FastAPI()

connection_args = {"host": "127.0.0.1", "port": "19530"}
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
    model = LLMLoader().load_you()
    prompt = create_prompt_icl_qa()
    chain = Chains.llm_chain(
        model,
        prompt,
    )
    result = chain.run(message_request.message)
    logger.debug(result)
    return {"answer": result}


@app.post("/weather")
async def weather(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["openweathermap-api"], llm=llm)
    agent_chain = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    result = agent_chain.run(message_request.message)
    return {"answer": result}


@app.post("/serax")
async def serax(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_cohere()
    tools = Tools.createTools(["seraxng"], llm=llm)
    # output_parser = CustomOutputParser()
    agent_chain = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, 
    )
    result = agent_chain.run(message_request.message)
    return {"answer": result}


@app.post("/wikipedia")
async def wikipedia(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_cohere()
    tools = Tools.createTools(["wikipedia"], llm=llm)
    agent_chain = initialize_agent(
        tools=tools, llm=llm, verbose=True, 
    )
    result = agent_chain.run(message_request.message)
    return {"answer": result}


@app.post("/requests_all")
async def req_all(request: Request, message_request: MessageRequest):
    logger.info("Search Net ...")
    res = Tools.requests_all(message_request.message)
    return {"answer": res}


@app.post("/dall_e")
async def dall_e(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["requests_all", "dall_e"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}

@app.post("/music_plugin")
async def music_plugin(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["music_plugin"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}

@app.post("/app_builder")
async def app_builder(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["app_builder", "requests_all"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}


@app.post("/url_reader")
async def url_reader(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["url_reader"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}

@app.post("/medium")
async def medium(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["medium"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}

@app.post("/transvid")
async def transvid(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["transvid"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}

@app.post("/quickchart")
async def quickchart(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["quickchart", "requests_all"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}

@app.post("/speak")
async def speak(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["speak", "requests_all"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}

@app.post("/ai_tool_hunt")
async def ai_tool_hunt(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()
    tools = Tools.createTools(["ai_tool_hunt", "requests_all"])
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    res = agent_chain.run(message_request.message)
    return {"answer": res}

@app.post("/stable_diffusion")
async def stable_diffusion(request: Request, message_request: MessageRequest):
    # llm = LLMLoader().load_useless()
    # tools = Tools.createTools(["stable_diffusion"])
    # agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # res = agent_chain.run(message_request.message)
    res = Tools.stable_diffusion(search=message_request.message)
    return {"answer": res}


@app.post("/search")
async def search_net(request: Request, message_request: MessageRequest):
    logger.info("Search Net ...")
    res = Tools.seraxng_lyric(message_request.message)
    return {"answer": res}


@app.post("/search_pac")
async def search_pacakge(request: Request, message_request: MessageRequest):
    logger.info("Search Net ...")
    res = Tools.seraxng_repos(message_request.message)
    return {"answer": res}

@app.post("/search_app")
async def search_app(request: Request, message_request: MessageRequest):
    logger.info("Search Net ...")
    res = Tools.seraxng_apps(message_request.message)
    return {"answer": res}

@app.post("/search_social")
async def search_social(request: Request, message_request: MessageRequest):
    logger.info("Search Net ...")
    res = Tools.seraxng_social(message_request.message)
    return {"answer": res}

@app.post("/search_map")
async def search_map(request: Request, message_request: MessageRequest):
    logger.info("Search Net ...")
    res = Tools.seraxng_map(message_request.message)
    return {"answer": res}

@app.post("/search_youtube")
async def search_youtube(request: Request, message_request: MessageRequest):
    logger.info("Search Net ...")
    res = Tools.youtube(message_request.message)
    return {"answer": res}


@app.post("/babyagi")
async def babyagi(request: Request, message_request: MessageRequest):
    llm = LLMLoader().load_useless()

    embeddings = EmbeddingLoader("cohere").load_embeddings()
    vector_db = VectoreStores(embeddings, connection_args=connection_args).load_milvus()

    verbose = True
    max_iterations: int = 3
    baby_agi = BabyAGI.from_llm(
        llm=llm,
        vectorstore=vector_db,
        verbose=verbose,
        max_iterations=max_iterations,
    )

    res = baby_agi._call({"objective": message_request.message})
    return {"answer": res}


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
    model = LLMLoader().load_useless()
    compressed = vector_db.compress_retriever(model, retriever)
    comp = compressed.get_relevant_documents(query)
    return {"all": docs, "compressed": comp}


@app.get("/vector/qa")
async def qa():
    model = LLMLoader().load_useless()
    embeddings = EmbeddingLoader("cohere").load_embeddings()
    vector_db = VectoreStores(embeddings, db_type="chroma")
    retriever = vector_db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)
    query = "What did the president say about Ketanji Brown Jackson"

    response = qa.run(query)

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
