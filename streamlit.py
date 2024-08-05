import streamlit as st 

st.title("Echo Bot")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]

#Display chat message from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
#react the user input
prompt =st.chat_input("What is your query?")

if prompt:
    #Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role":"user",
                                      "content":prompt})
    reponse=f"Echo:{prompt}"
    # Display the assistant reponse in chat message container 
    with st.chat_message("assistant"):
        st.markdown(reponse)
    # Add assitant reponse to chat history
    st.session_state.messages.append({"role":"assistant",
                                      "content":reponse}) 
