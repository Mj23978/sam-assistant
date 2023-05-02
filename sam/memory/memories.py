from typing import List, Optional, Type

from langchain.memory import ChatMessageHistory, ConversationBufferMemory


class Loaders:

    @staticmethod
    def messageHistory(path: str):
        history = ChatMessageHistory()
        return history

    @staticmethod
    def bufferMemory(path: str):
      memory = ConversationBufferMemory() 
      return memory

