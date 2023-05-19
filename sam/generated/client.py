import asyncio

from grpclib.client import Channel
from chatbot import (CompleteRequest, CompleteResponse, SamChatBase,
                       SamChatStub)

async def main():
    channel = Channel(host="127.0.0.1", port=50051)
    service = SamChatStub(channel)
    response = await service.complete(CompleteRequest(message="hello"))
    print(response)

    channel.close()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())