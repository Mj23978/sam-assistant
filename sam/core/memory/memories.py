from typing import List, Optional, Type

from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
    RedisChatMessageHistory,
    RedisEntityStore,
    VectorStoreRetrieverMemory,
)


class Memory:
    @staticmethod
    def messageHistory(path: str):
        history = ChatMessageHistory()
        return history

    @staticmethod
    def bufferMemory(path: str):
        memory = ConversationBufferMemory()
        return memory

    @staticmethod
    def chatSummary(path: str):
        memory = ConversationSummaryMemory()
        return memory
