import logging
from dotenv import load_dotenv
import streamlit as st
from htmlTemplates import css, bot_template, user_template
from generate import QuestionAnsweringModel
import pandas as pd


class ChatBot:
    def __init__(self):
        self.conversation = None
        self.history = None
        
        df = pd.read_csv("data/final.csv", sep=";")
        if "model" not in st.session_state:
          st.session_state.model = QuestionAnsweringModel(df)

    def main(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting app...")

        load_dotenv()

        st.set_page_config(page_title='Amazon Chatbot',
                           page_icon=':robot:', initial_sidebar_state='auto')

        if "history" not in st.session_state:
            st.session_state.history = [
                {
                    "role": "bot",
                    "content": "Hello, I am an Amazon Sagework Expert. How can I help you?"
                }
            ]

        self.load_ui()

    def generate_response(self, user_question):
        response = st.session_state.model.generate_response(user_question)
        return response

    def load_css_styles(self):
        st.write(css, unsafe_allow_html=True)

    def display_sidebar(self):
        st.sidebar.title("Amazon Chatbot")
        st.sidebar.markdown(
            """
            This is a chatbot that answers questions about Amazon Sagework.
            """
        )

    def display_messages(self, history):
        for message in history:
            if message['role'] == 'user':
                st.markdown(user_template.replace(
                    '{{MSG}}', message['content']), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace(
                    '{{MSG}}', message['content']), unsafe_allow_html=True)

    def handle_user_input(self, user_question):
        if user_question:
            st.session_state.history.append({
                "role": "user",
                "content": user_question
            })

            response_content = self.generate_response(user_question)
            st.session_state.history.append({
                "role": "bot",
                "content": response_content
            })

    def load_ui(self):
        self.load_css_styles()

        self.display_sidebar()

        st.title('Hello, I am an Amazon SageMaker Expert 👩🏻‍🦰')
        user_question = st.text_input('Ask a question about Amazon SageMaker:')

        self.handle_user_input(user_question)

        self.display_messages(st.session_state.history)

    def run(self):
        self.main()


if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run()
