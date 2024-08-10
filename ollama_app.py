import streamlit as st
import numpy as np
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
import random
import string
import traceback
from langchain.agents import tool
from sqlparse import format
from soap_requests_extract_data import main, authenticate
import pandas as pd
from langchain_community.callbacks import StreamlitCallbackHandler
from time import sleep
from langchain_community.llms.ollama import Ollama

dotenv.load_dotenv()

import streamlit as st


st.set_page_config(layout="wide")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

login_placeholder = st.empty()

if not st.session_state.logged_in:

    with login_placeholder.form(key="my_form"):

        user_id = ""
        pw = ""

        st.markdown("## Rite Agent Login")
        st.markdown("###### Enter to sign in into the Oracle Applications Cloud")
        user_id = st.text_input(label="Enter the username")
        pw = st.text_input(label="Enter the password", type="password")

        submit_button = st.form_submit_button(label="Submit")

    if "clicked" not in st.session_state:
        st.session_state.clicked = False

    if submit_button:
        st.session_state.clicked = True

    if st.session_state.clicked:

        authentication = False

        token = authenticate(userID=user_id, password=pw)

        if isinstance(token, Exception):
            authentication = False
            st.session_state.clicked = False
            st.warning("Username/Password in Incorrect.")
            sleep(5)
            st.rerun()
        else:
            authentication = True
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.session_state.pw = pw
            login_placeholder.empty()


if st.session_state.logged_in:
    try:
        os.remove("AI_Output.xlsx")
    except FileNotFoundError as e:
        pass

    current_model_name = "llama3.1:latest"

    session_id = "abc123"
    openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(
        model=current_model_name,
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        temperature=0,
    )
    # llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-405b-reasoning")
    pc = vectorstores.PineconeVectorStore(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        embedding=openai_embeddings,
        index_name="newdataindex",
    )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    @tool
    def report_download(path):
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
            the file.xlsx
        """
        try:
            os.remove("AI_Output.xlsx")
        except FileNotFoundError:
            pass

        df = pd.read_csv(f"{path}/{path}.csv")
        df.to_excel("Your_Report.xlsx")

        os.remove(path=f"{path}/{path}.csv")
        os.remove(path=f"{path}/{path}.txt")

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
            string: returns path to access download
        """
        global username
        global password

        userID = st.session_state.user_id
        passkey = st.session_state.pw
        base_folder_path = ""
        query = query

        res = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        # res = "AI_Output"

        try:
            result = main(res, userID, passkey, base_folder_path, query)

            if result:
                return (
                    "The SQL query has been added to the user's path",
                    res,
                )

            else:
                return "SQL query has failed. Run again."

        except Exception as e:
            return str(e), "Please run again"

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

        model = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo"
        )

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

        tools = [
            retriever_agent,
            parse_sql,
            generate_report,
            report_download,
        ]

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

    # st.title("Rite Agent Bot")
    st.markdown(
        "<h1 style='text-align: center; margin-top: -5%; color: black;'>Rite Agent</h1>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "history_agent" not in st.session_state:
        st.session_state.history_agent, st.session_state.session_id = agent_history()
        # print(st.session_state.history_agent)

    col2, col3 = st.columns([3, 2])

    st.sidebar.header("Chat History")
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.sidebar:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with col2:
        st.header("Chat")
        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

    with col3:
        st.header("Thinking process")
        st_callback = StreamlitCallbackHandler(st.container())

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

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            xlxbyte = None
            try:
                with open("Your_Report.xlsx", "rb") as xlxfile:
                    xlxbyte = xlxfile.read()
                st.download_button(
                    label="Download Your Report Here",
                    data=xlxbyte,
                    file_name="Your_Report.xlsx",
                )
                os.remove("Your_Report.xlsx")
            except FileNotFoundError:
                pass
