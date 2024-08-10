from langchain_pinecone import vectorstores
from abc import ABC
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import tool

load_dotenv()


class LangchainChat(ABC):

    def __init__(self) -> None:
        super().__init__()
        openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

        # hugging_face = HuggingFaceHubEmbeddings(huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"))
        self.pc = vectorstores.PineconeVectorStore(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            embedding=openai_embeddings,
            index_name="dataindex",
        )

        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=1)

        with open("system_prompt.txt", "r") as file:
            system_prompt = file.read().replace("\n", "")

        self.prompt = system_prompt
        self.conversational_rag_chain = None
        self.history_aware_retriever = None
        self.rag_chain = None
        self.establish_conversational_chain()
        self.store = {}

        self.memory = []

    def establish_conversational_chain(self):

        with open("system_prompt.txt", "r") as file:
            system_prompt = file.read().replace("\n", "")

        retriever = self.pc.as_retriever()

        ### Contextualize question ###
        contextualize_q_system_prompt = system_prompt

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        ### Answer question ###
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, question_answer_chain
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def invoke_conversation(self, query):
        """
        Use this to create a conversation that returns an
        SQL query to find specific data in an Oracle Fusion
        Database based on Pinecone Data.
        """

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = self.conversational_rag_chain.invoke(
            {"input": query, "context": self.prompt},
            config={"configurable": {"session_id": "abc123"}},
        )

        return response["answer"]


def main():

    chatbot = LangchainChat()
    while True:
        query = input("Ask a question, type 'exit' to finish: \n")
        if query == "exit":
            print("Goodbye!!!")
            break
        else:
            answer = chatbot.invoke_conversation(query)
            print(answer + "\n")


# main()
