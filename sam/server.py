# from concurrent import futures

# import grpc
import asyncio
from typing import AsyncIterator

from dotenv import load_dotenv
from grpclib.server import Server

from sam.core.chains.chains import create_chain
from sam.core.llms.llms import EmbeddingLoader, LLMLoader
from sam.core.loaders.loaders import Loaders
from sam.core.prompts.prompts import create_prompt_vega
from sam.core.vectorstore.vectorstores import VectoreStores
from sam.generated.chatbot import (GenerateTextRequest, GenerateTextResponse,
                                     LanguageModelServiceBase, LanguageModelServiceStub)
from sam.core.utils import logger

load_dotenv()


class SamChatServicer(LanguageModelServiceBase):
    async def complete(self, request: GenerateTextRequest) -> GenerateTextResponse:
        model = LLMLoader("cohere", top_k=40, top_p=0.95, model="command-xlarge-nightly").load_model()
        prompt = create_prompt_vega()
        result = create_chain(
            model, prompt, request.user_input)
        response = GenerateTextResponse(
            generated_text=result,
        )
        logger.info(F"Response: {response.generated_text}")
        return response


async def main():
    server = Server([SamChatServicer()])
    await server.start("127.0.0.1", 50051)
    print("🚀 Server Starting 50051 ...")
    await server.wait_closed()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

# server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
# add_SamChatServicer_to_server(SamChatServicer(), server)
# server.add_insecure_port("[::]:50051")
# server.start()
# print("Starting server on port 50051....")
# server.wait_for_termination()
