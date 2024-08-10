from langchain_pinecone import vectorstores
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.agents import create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from abc import ABC

load_dotenv()


session_id = "abc123"
openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = vectorstores.PineconeVectorStore(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    embedding=openai_embeddings,
    index_name="dataindex",
)


def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
    if session_id not in self.store:
        self.store[session_id] = ChatMessageHistory()
    return self.store[session_id]


def create_agent():

    system_prompt = ""

    with open("system_prompt.txt", "r") as file:
        system_prompt = file.read().replace("\n", "")

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(input_variables=["input"], template="{input}")
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    retriever_agent = create_retriever_tool(
        pc.as_retriever(),
        "pinecone_data_retriever",
        "Necessary for retrieving data from a pinecone database regarding Oracle Fusion questions. Returns a retrieval chain",
    )

    print(qa_prompt)

    tools = [retriever_agent]

    agent = create_tool_calling_agent(llm, tools, qa_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


def main(query="What is the primary key of PJT_PLAN_VERSIONS"):

    session_id = "abc123"
    openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = vectorstores.PineconeVectorStore(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        embedding=openai_embeddings,
        index_name="dataindex",
    )

    agent_executor = create_agent()

    message_history = ChatMessageHistory()

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    agent_with_history.invoke(
        {"input": query},
        config={"configurable": {"session_id": "abc123"}},
    )


main()
