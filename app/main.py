import logging
import os
from dotenv import load_dotenv
import streamlit as st
from htmlTemplates import css, bot_template, user_template


class ChatBot:
    def __init__(self):
        self.conversation = None
        self.history = None

    def main(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting app...")

        load_dotenv()

        st.set_page_config(page_title='Amazon Chatbot', page_icon=':robot:')

        # if "conversation" not in st.session_state:
        #     st.session_state.conversation = self.get_conversation()

        # if "history" not in st.session_state:
        #     st.session_state.history = []

        self.load_ui()

    def get_conversation(self):
        model = "blabla"
        return model

    def load_css_styles(self):
        st.write(css, unsafe_allow_html=True)

    def load_ui(self):
        self.load_css_styles()

        # self.display_sidebar()

        st.title('Hello, I am an Amazon Sagework Expert 👩🏻‍🦰')
        # user_question = st.text_input('Ask a question about Amazon Sagework:')

        # self.handle_user_input(user_question)

        # self.display_messages(st.session_state.history)

    def run(self):
        self.main()


if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run()