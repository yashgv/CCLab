# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3

import os
import re
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
load_dotenv()

# Convert CSV file into a SQL DB

csv_file_path = 'Healthcare_dataset_with_summary.csv'


df = pd.read_csv(csv_file_path)
df.head()

database_path = 'data.db'


connection = sqlite3.connect(database_path)

df.to_sql('patients', connection, if_exists='replace', index=False)

query_result = pd.read_sql('SELECT * FROM patients', connection)
print(query_result.head())

# connection.close()

from langchain_community.utilities import SQLDatabase

db_path = 'data.db'
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM patients LIMIT 10;")


os.environ["GOOGLE_API_KEY"] = 'AIzaSyDpkr0kKnwlOBkB__ypof-2ETIFTTx7rBk'


# llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key='', convert_system_message_to_human=True, temperature=0.0)

llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True, temperature=0.0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "What is the count occurence of all blood groups in the data?"})


regex_pattern = r'^```sql\n|\n```$'


modified_response = re.sub(regex_pattern, '', response)
print(modified_response)

db.run(modified_response)




df_query_result = pd.read_sql_query(modified_response, connection)
print(df_query_result)
type(df_query_result)

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent as experimental_pandas_agent


# pandas_agent = create_pandas_dataframe_agent(llm, 
#                                              df_query_result,
#                                              allow_dangerous_code = True)


agent = create_pandas_dataframe_agent(llm, df_query_result, verbose=True,
    return_intermediate_steps=True,
    prefix=f"""You are a matplotlib expert. Generate a syntactically matplotlib code, for the given query.
                             All the variables used in the matplotlib code must be present in the dataframe provided. 
                             NOTE: The matplotlib code must provide Plots with color and legends, for all the query.
                             Only generate matplotlib code for this.
                             
            The description of each column in the dataframe {df} is provided below:
            Based on the dataset imported, you must select the appropriate table schema from the below schemas of the tables defined below.
            End of column Description.
            
            The code must contain extraction of data of {df} from the above mentioned column names only. 
            Donot prepare dataset of you own and work on the data given.
            Make sure to add proper code for whatever query is being asked.
            - You are stritly prohibited to use the column of your choice and only work with the columns mentioned in the {df}
            - If you wish to create additional columns, make sure to define them before using them in your code e.g. month_names = [].
            - Review the query produced and make sure to convert date columns in a proper error free format.
            - Make sure you execute the code and check if its error free and can be executed without any errors. Avoid using .to_period() for date columns.
            - All the variables must be based on the dataframe provided only.
            - Usage of plt.subplots() is strictly prohibited.
                - Make sure you use your intelligence to map only the available months and quarters in the code. for e.g if the data is only available 
                  for 3 quarters then use 3 instead of 4.
            - You are strictly subjected to produce plots that are aesthetically pleasing with proper labels, legends, axes, etc.
            - You are strictly subjected to preduce plots that are clearly visible, no legends should overlap with the actual plot
            Final answer should be a matplotlib code in the format without the plt.show() command in the code: python <code>""",
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            include_df_in_prompt=None,
            # handle_parsing_errors=True,
            agent_executor_kwargs=dict(handle_parsing_errors=True),
            max_iterations=3,
            suffix = """
                        This is the result of `print(df.head())`:
                        {df_head}
        
                        Conversation history:
                        {history}

                        Begin!
                        Question: {input}
                        {agent_scratchpad}""" 
                )

pandas_query = "plot the data"
agent_result = agent.invoke({
    "input": pandas_query,
    "df":df_query_result,
    "agent_scratchpad": "",
    "history": ""})
print(agent_result)


