from tempfile import NamedTemporaryFile
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os


def upload_csv():
    load_dotenv()

    # st.set_page_config(page_title="Ask your CSV")
    st.header("Extract insights from your financial data 💸")

    file = st.file_uploader("upload file", type="csv")

    if file is not None:
        temp_dir = 'C:/Temp'  # Change this to the desired directory path
        df = pd.read_csv(file)
        st.dataframe(df)
        os.makedirs(temp_dir, exist_ok=True)  # Create the directory if it 
        with NamedTemporaryFile(mode='w+b', suffix=".csv",dir=temp_dir, delete=False) as f:
            f.write(file.getvalue())
            f.flush()
            llm = OpenAI(temperature=0)
            user_input = st.text_input("Question here:")
            agent = create_csv_agent(llm, f.name, verbose=True)
            if user_input:
                response = agent.run(user_input)
                st.write(response)


