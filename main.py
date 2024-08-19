import os
import json

import streamlit as st
from groq import Groq


# streamlit page configuration
st.set_page_config(
    page_title="Gluten Free Harmonie AI Chatbot",
    page_icon="🫒",
    layout="centered"
)

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
f = open('knowledge_base.txt', "r")
knowledge = f.read()

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# save the api key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# initialize the chat history as streamlit session state of not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# streamlit page title
#st.title("SH's Virtual Assistant :sunglasses:")
st.subheader("Gluten Free Harmonie AI Chatbot 🫒", divider="gray")
st.subheader("", divider=False)

# display chat history
for message in st.session_state.chat_history:
    if message["role"]=='user':
        av = "💬"
    else:
        av = "🫒"
    with st.chat_message(message["role"],avatar=av):
        st.markdown(message["content"])


# input field for user's message:
user_prompt = st.chat_input("Ask a question..")


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            div.embeddedAppMetaInfoBar_container__DxxL1 {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


if user_prompt:

    st.chat_message("user", avatar="💬").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # sens user's message to the LLM and get a response

    instruction= '''
    You are a AI agent assisting the visitors of Gluten Free Harmonie, a website specialized in Gluten Free recipes with a Moroccan-Mediterranean inspiration. You live in the Gluten Free realm. You have access to the following info :\n
    '''
    messages = [
        {"role": "system", "content": instruction+knowledge},
        *st.session_state.chat_history
    ]

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # display the LLM's response
    with st.chat_message("assistant", avatar="🫒"):
        st.markdown(assistant_response)

