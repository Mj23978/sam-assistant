import copy

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from sse_starlette import EventSourceResponse

from sam.chains import create_chain
from sam.llms import load_model
from sam.prompts import create_prompt_icl_qa

load_dotenv()

# create the app
app = FastAPI()


@app.get("/")
async def hello():
    return {"hello": "wooooooorld"}


@app.get("/lang")
async def hello():

    model = load_model("cohere")
    prompt = create_prompt_icl_qa()
    result = create_chain(model, prompt, "Name 10 Footbal Players")
    return {"answer": result}


@app.get("/llama")
async def llama(request: Request):
    stream = llm(
        "name 10 footballer players in alphabatic order :",
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
