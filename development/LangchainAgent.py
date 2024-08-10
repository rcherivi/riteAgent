from abc import ABC
import mysql.connector
from langchain_community.utilities import SQLDatabase
import ollama
import json
import pandas as pd


class LangchainAgent(ABC):

    def __init__(self, user, host, password, db):

        # initialize the database connection
        self._user = user
        self._host = host
        self._password = password
        self._db = db

        self.response = None

        # initialize LLM

    def establish_connection(self):
        self._conn = self.conn = mysql.connector.connect(
            host=self._host,
            user=self._user,
            password=self._password,
            database=self._db,
        )
        self._cursor = self._conn.cursor()

    def send_user_query(self):

        print("What data would you like to generate?")
        user_query = input()

        return user_query

    def run_generate_query(self, user_query):

        with open("system_prompt.txt", "r") as file:
            system_prompt = file.read()

        # generates a response through Ollama to generate query
        self.response = ollama.generate(
            model="llama3:latest", system=system_prompt, prompt=user_query
        )

        sql_query = self.response["response"]

        try:
            self._cursor.execute(sql_query)
        except Exception as e:
            print("An error occurred:", e)
            self.run_generate_query(user_query)

        return sql_query

    def report_prompt(self, user_query, sql_query, table_data) -> str:

        prompt = (
            f"Generate a 250 word report based on the user query (USER QUERY: {user_query}), \nthe generated SQL"
            + f"query (SQL QUERY: {sql_query}) \n and the following data stored in tuples (TUPLE DATA: {table_data})"
        )

        return prompt

    def generate_report(self, user_query, sql_query, table_data):

        system_prompt = self.report_prompt(user_query, sql_query, table_data)
        print("hi!")

        response = ollama.generate(model="gemma:2b", prompt=system_prompt)
        print(response["response"])

    def query_to_report(self):

        # get the user query
        self.establish_connection()
        user_query = self.send_user_query()

        sql_query = self.run_generate_query(user_query)
        result = self._cursor.fetchall()

        self.generate_report(user_query, sql_query, result)

        self._conn.close()


def main():
    host = "host-ip-address-here"
    user = "username-here"
    password = "password-here"
    database = "database-here"

    TextSQLBot = LangchainAgent(user=user, host=host, password=password, db=database)
    TextSQLBot.query_to_report()


main()
