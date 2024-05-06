import streamlit as st
import sqlite3
import pandas as pd
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain

# Connect to the database
conn = sqlite3.connect('pincode.db')

# Read CSV file
@st.cache
def load_data():
    df = pd.read_csv('Pincode_30052019.csv', encoding='ISO-8859-1')
    df.columns = ['CircleName', 'RegionName', 'DivisionName', 'OfficeName', 'Pincode',
                  'OfficeType', 'Delivery', 'District', 'StateName']
    return df

df = load_data()

# Create table if not exists
cursor = conn.cursor()
query = '''
CREATE TABLE IF NOT EXISTS Postal_Offices (
    CircleName VARCHAR(255),
    RegionName VARCHAR(255),
    DivisionName VARCHAR(255),
    OfficeName VARCHAR(255),
    Pincode INTEGER,
    OfficeType VARCHAR(255),
    Delivery VARCHAR(255),
    District VARCHAR(255),
    StateName VARCHAR(255)
);
'''
cursor.execute(query)

import os 
os.environ["GOOGLE_API_KEY"] = "AIzaSyAXOjM6OYg0bDS-NckePUAK_IAb_15G_z0"

# Import the CSV into the database
df.to_sql('Postal_Offices', conn, if_exists='append', index=False)

# Create SQLDatabaseChain
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, verbose=True)
db = SQLDatabase.from_uri("sqlite:///pincode.db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
chain = create_sql_query_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, verbose=True), db)

# Streamlit UI
st.title('SQL Query Generator')
query_question = st.text_input('Write the prompt here')
if st.button('Run'):
    response = chain.invoke({"question": query_question})
    st.write(response)

# Close the cursor and connection
cursor.close()
conn.close()
