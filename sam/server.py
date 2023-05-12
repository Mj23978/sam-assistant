# from concurrent import futures

# import grpc
import asyncio
from typing import AsyncIterator

from dotenv import load_dotenv
from grpclib.server import Server

from sam.core.chains.chains import create_chain
from sam.core.llms.llms import EmbeddingLoader, LLMLoader
from sam.core.loaders.loaders import Loaders
from sam.core.prompts.prompts import create_prompt_icl_qa
from sam.core.vectorstore.vectorstores import VectoreStores
from sam.generated.myservice import (CompleteRequest, CompleteResponse,
                                     SamChatBase, SamChatStub)


class SamChatServicer(SamChatBase):
    async def complete(self, request: CompleteRequest) -> CompleteResponse:
        model = LLMLoader("theb").load_model()
        prompt = create_prompt_icl_qa()
        result = create_chain(
        model, prompt, request.message)
        response = CompleteResponse(
            message=result,
            more="More ... "
        )
        print(response.message)
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
