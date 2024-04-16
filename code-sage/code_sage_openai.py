from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("API_KEY")

client = OpenAI(api_key = OPENAI_API_KEY)

st.title('Code Sage 🧙🏻‍♂️')
st.subheader('Analyse your code with help of AI')

prompt = st.text_area('Enter your code to analyse')

if st.button('Generate') == True:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
        {"role": "system", "content": """You are a polite, expert, experienced, helpful assistant. 
                                        Your task is to analyse this code return and errors or shortcomings, in a tabular format.
                                        The columns of table should be,serial number, line of code with error highlighted, error description, possible solution.
                                        Also, only return the table and not any extra message like 'Certainly! Let's analyze the code provided:' or anything else.
                                        Make sure that the table is in a format such that it is directly rendered into the streamlit's st.write().
                                        If the input is not relevant, you can politely ask the user to enter valid code.
                                        """},
        {"role": "user", "content": prompt}
        ]
                      
    )

    st.write(response.choices[0].message.content)



