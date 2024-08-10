import streamlit as st
from langchain_pinecone import vectorstores
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import dotenv
import os
from langchain_groq import ChatGroq
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
from langchain.agents import tool
from soap_requests_extract_data import main, delete_objects_in_session
import pandas as pd
from langchain_community.callbacks import StreamlitCallbackHandler


dotenv.load_dotenv()

import streamlit as st

try:
    delete_objects_in_session()
except Exception as e:
    pass

try:
    os.remove("AI_Output.xlsx")
except FileNotFoundError as e:
    pass

session_id = "abc123"
openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
# llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="mixtral-8x7b-32768")
pc = vectorstores.PineconeVectorStore(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    embedding=openai_embeddings,
    index_name="dataindex",
)


def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
    if session_id not in self.store:
        self.store[session_id] = ChatMessageHistory()
    return self.store[session_id]


@tool
def report_download():
    """
    Name: download_report

    Description: YOU MUST USE THIS AFTER YOU HAVE GENERATED THE REPORT,
    SO THAT THE USER CAN ACCESS THE DATA. This tool generates an Excel file from the
    downloaded csv file and allows the user to download the Excel
    file to see their data

    Arguments:
        None

    Returns:
        String explaining that the report has been downloaded to
        AI_Output.xlsx
    """
    try:
        os.remove("AI_Output.xlsx")
    except FileNotFoundError:
        pass

    df = pd.read_csv("AI_Output/AI_Output_0.csv")
    df.to_excel("AI_Output.xlsx")

    return "Report has been downloaded"


@tool
def generate_report(query: str):
    """
    Name: Report_Generator

    Description:  Generates a report based on a parsed SQL query and returns
    it to the user for download using a SOAP API. PLEASE MAKE SURE THAT YOU
    HAVE PARSED THE SQL QUERY FIRST BEFORE YOU GENERATE THE REPORT.

    Arguments:
        query: SQL query to generate report

    Returns:
        documents: returns documents as reports
    """

    userID = "lisa.jones"
    password = "Aw#4GG%9"
    base_folder_path = ""
    query = query

    # res = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    res = "AI_Output"

    try:
        main(res, userID, password, base_folder_path, query)
        return "The SQL query has been added to the user's path"
    except Exception as e:
        return str(e)


@tool("parse_sql")
def parse_sql(output: str):
    """
    Name: SQL_Query_Parser

    Description:
    YOU MUST USE THIS BEFORE GENERATING A REPORT
    Parses SQL queries from the natural language text given and
    returns a formatted and parsed SQL query that can be used.

    Arguments:
        output: string value that represents a natural language text

    Returns:
        sql_query: returns a parsed and formatted sql query

    """

    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    template = """
    Answer the question based on the context below.

    Context: {context}

    Input:
    {input}
    """

    system_prompt = (
        "You have been given a natural language text. Please "
        + "parse out only the SQL query and return that query "
        + "as a string text. Please remove the semicolon at the"
        + "end of the query."
    )

    prompt = ChatPromptTemplate.from_template(template)

    # prompt = ChatPromptTemplate.from_template(template=template)
    # prompt.format(context=system_prompt, text=output)
    # prompt = ChatPromptTemplate.from_template(template)

    print(prompt)

    model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo")

    parser = StrOutputParser()

    inputs = {"context": system_prompt, "input": output}

    chain = prompt | model | parser

    sql_query = chain.invoke(inputs)

    sql_query = sql_query.format()

    return sql_query


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
        "SQL_Query_Generator",
        "Necessary for generating SQL queries by retrieving data from a pinecone database with Oracle Fusion documentation. USE THIS TOOL FIRST. Please use parse_sql tool after this",
    )

    tools = [retriever_agent, parse_sql, generate_report, report_download]

    agent = create_tool_calling_agent(llm, tools, qa_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


def agent_history():

    session_id = "abc123"

    agent_executor = create_agent()

    message_history = ChatMessageHistory()

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_history, session_id


st.title("Rite Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_agent" not in st.session_state:
    st.session_state.history_agent, st.session_state.session_id = agent_history()
    # print(st.session_state.history_agent)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st_callback = StreamlitCallbackHandler(st.container())

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer = st.session_state.history_agent.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": st.session_state.session_id},
            "callbacks": [st_callback],
        },
    )

    answer_1 = answer["output"]

    response = f"Rite Bot: {answer_1}."

    print(answer)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        st.markdown(response)

        xlxbyte = None

        try:
            with open("AI_Output.xlsx", "rb") as xlxfile:
                xlxbyte = xlxfile.read()
            st.download_button(
                label="Download Your Report Here",
                data=xlxbyte,
                file_name="AI_Output.xlsx",
            )
        except FileNotFoundError:
            pass

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
