import streamlit as st
import helper as hp



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

st.title("RAGTAG ChatBot")

# initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input in field
if prompt := st.chat_input("Ask me anything!"):
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
        with st.chat_message("PAI"):
            st.markdown(response)

    # save model response
    st.session_state.messages.append({"role": "bot", "content": response})



# # Create a session state variable to store chat history
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = ""

# def display_chat_history():
#     st.write(st.session_state["chat_history"])

# # App title and layout
# user_query = st.text_input("Enter your message:")

# if st.button("Send"):
#     with st.spinner("Loading..."):
#         local_agent = hp.build_workflow()
#         response = hp.run_agent(user_query, local_agent, st.session_state["chat_history"])

#         # Update chat history (append user query and response)
#         st.session_state["chat_history"] += f"\nYou: {user_query}\nBot: {response}\n"

#         # Display updated chat history
#         display_chat_history()