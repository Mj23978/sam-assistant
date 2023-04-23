import asyncio
import copy

import requests
from fastapi import FastAPI, Request
from llama_cpp import Llama
from sse_starlette import EventSourceResponse

# from langchain import PromptTemplate, LLMChain

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

print("Loading the model ...")

# model_name = r"D:\AI\llama.cpp\models\13B\koala-13B-4bit-128g.GGML.bin"
model_name = r"D:\AI\llama.cpp\models\13B\ggml-vicuna-13b-4bit.bin"
llm = Llama(model_path=model_name, n_ctx=2048, n_threads=6,
            f16_kv=False, seed=0, verbose=True,)

print("Done")

# create the app
app = FastAPI()


@app.get("/")
async def hello():
    return {"hello": "wooooooorld"}


@app.get("/jokes")
async def jokes(request: Request):
    def get_messages():
        url = "https://official-joke-api.appspot.com/random_ten"
        response = requests.get(url)
        if response.status_code == 200:
            jokes = response.json()
            messages = []
            for joke in jokes:
                setup = joke['setup']
                punchline = joke['punchline']
                message = f"{setup} {punchline}"
                messages.append(message)
            return messages
        else:
            return None

    async def sse_event():
        while True:
            if await request.is_disconnected():
                break

            for message in get_messages():
                yield {"data": message}

            await asyncio.sleep(1)
    
    print("starting")
    return EventSourceResponse(sse_event())


@app.get("/llama")
async def llama(request: Request):
    stream = llm(
        "Question: Who is Ada Lovelace? Answer: ",
        max_tokens=200,
        stop=["\n", " Q:"],
        stream=True,
    )

    async def async_generator():
        for item in stream:
            print(copy.deepcopy(item)["choices"][0]["text"])
            yield item

    async def server_sent_events():
        async for item in async_generator():
            if await request.is_disconnected():
                break

            result = copy.deepcopy(item)
            text = result["choices"][0]["text"]

            yield {"data": text}

    return EventSourceResponse(server_sent_events())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=7860)
