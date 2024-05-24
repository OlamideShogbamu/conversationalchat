import getpass
import os
import ast
import re
from flask import Flask, request, jsonify
from langchain_community.utilities import SQLDatabase
from geoalchemy2 import Geometry
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

app = Flask(__name__)

db = SQLDatabase.from_uri('postgresql://pgAdmin:Geost4r%40123@pggeost4r.postgres.database.azure.com:5432/rmnchn')

examples = [
    {"input": "List all artists. Enumerate every artist. Provide a comprehensive list of artists. Compile a roster of all artists", 
     "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'. Retrieve all albums by the artist 'AC/DC'. Locate every album belonging to 'AC/DC'. Find all albums associated with the artist 'AC/DC'",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre. Enumerate tracks categorized under 'Rock' genre. Provide a list of all 'Rock' genre tracks. Compile a roster of tracks belonging to the 'Rock' genre",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks. Calculate the cumulative duration of all tracks. Determine the combined length of all tracks. Compute the total duration of all tracks",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada. Enumerate all Canadian customers. Provide a list of customers hailing from Canada. Compile a roster of Canadian customers",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5. Determine the number of tracks in the album with ID 5. Find the count of tracks within the album having ID 5. Calculate the total number of tracks present in the album identified by ID 5",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
     {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]

system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question from a user, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
 \nHere is the relevant table info: {table_info}\n\nHere is a non-exhaustive \
list of possible feature values. 
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.

Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If an error occurs during query execution, rewrite the query and attempt again. Avoid making any DML statements (INSERT, UPDATE, DELETE, DROP, etc.).

Also, when filtering based on a feature value, ensure to validate its spelling against a provided list of "proper_nouns" and correct it in your query, generate a response based on the correction, but let the user know in the final output that you made some corrections, stating the exact corrections you made.
If a user query appears unrelated to the database, prompt them to reconstruct the question and ask again. 

Here are some examples of user inputs and their corresponding SQL queries:"""

queries = [
        "SELECT DISTINCT dmg_lga FROM microplan_2023_2024",
        "SELECT DISTINCT dmg_ward_di FROM microplan_2023_2024",
        "SELECT DISTINCT dmg_health_facility FROM microplan_2023_2024"
    ]
    
results = []
for query in queries:
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    results.extend(res)
    
@app.route('/scorecard/key/<key>/<question>')
def chatbot(key, question):
    os.environ["OPENAI_API_KEY"] = key
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    vector_db = FAISS.from_texts(results, OpenAIEmbeddings())
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
    valid proper nouns. Use the noun most similar to the search."""
    retriever_tool = create_retriever_tool(
    retriever,
    name="proper_nouns",
    description=description,)

    example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],)

    from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,)

    few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect","table_info", "top_k", "proper_nouns"],
    prefix=system_prefix,
    suffix="",)

    full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_sql_agent(
    llm=llm,
    db=db,
    extra_tools=[retriever_tool],
    prompt=full_prompt,
    agent_type="openai-tools",
    verbose=True,
    agent_executor_kwargs={"return_intermediate_steps": True})
    
    # return agent
    
    res = agent.invoke({"input": question})
    for action, r in res["intermediate_steps"]:
        for message in action.message_log:
            if message.content.strip():
                print(message.content)
    
    print(res['output'])
    
    return res['output']
# question = input('ask your question')
# chatbot(examples, system_prefix, question)


if __name__ == '__main__':
    app.run(debug=True)
