import streamlit as st
import helper as hp
import questions as qn

# Predefined questions
question1 = qn.qn1
question2 = qn.qn2
question3 = qn.qn3
question4 = qn.qn4

st.set_page_config(page_title="RAGTAG ChatBot", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    .reportview-container {
        background: #3A435E
    }
    .reportview-container .main .block-container {
        padding: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a container for the header
header = st.container()
header.title("RAGTAG ChatBot")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

# Create a sticky header for the title
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 0;
        background-color: white;
        padding: 10px;
        z-index: 999;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def handle_user_input(prompt):
    # display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # save user input
    st.session_state.messages.append({"role": "user", "content": prompt})

    # show loading icon
    with st.spinner("Generating... 🤖"):
        # get response from model
        response = hp.run_agent(prompt, hp.build_workflow(), st.session_state.messages)

        # display the response from agent in chat message container
        with st.chat_message("Bot"):
            st.markdown(response)

    # save model response
    st.session_state.messages.append({"role": "bot", "content": response})

def button_event_handler(button_qn):
    # add qn from button to chat history
    st.session_state.messages.append({"role": "user", "content": button_qn})

    # show loading icon
    with st.spinner("Generating... 🤖"):
        # get response from model
        response = hp.run_agent(button_qn, hp.build_workflow(), st.session_state.messages)

        # save model response
    st.session_state.messages.append({"role": "bot", "content": response})

# initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def display_welcome_message():
    st.markdown("Hello! I'm the RAGTAG ChatBot. Ask me anything! 🤖")
    # Predefined question buttons
    if st.button(question1):
        button_event_handler(question1)
        return

    if st.button(question2):
        button_event_handler(question2)
        return

    if st.button(question3):
        button_event_handler(question3)
        return

    if st.button(question4):
        button_event_handler(question4)
        return

# if no chat messages, display welcome message
if not st.session_state.messages:
    display_welcome_message()

# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input in field
if prompt := st.chat_input("Ask me anything!"):
    handle_user_input(prompt)
