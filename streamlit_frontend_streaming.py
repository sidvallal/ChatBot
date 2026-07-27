import streamlit as st
from langGraph_backend import chatbot
from langchain_core.messages import SystemMessage,HumanMessage, BaseMessage

CONFIG = {'configurable':{'thread_id': 'thread-1'}}

if 'messages_history' not in st.session_state:
    st.session_state['messages_history'] = []


for message in st.session_state['messages_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
        
        
user_input = st.chat_input('Type here')

if user_input:
    
    st.session_state['messages_history'].append({'role':'user', 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config = CONFIG)
    # ai_message = response['messages'][-1].content
    

    with st.chat_message('assistant'):
        # st.text(ai_message)
        ai_message = st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                     {'messages': [HumanMessage(content=user_input)]},
                        config = {'configurable':{'thread_id': 'thread-1'}},
                        stream_mode = 'messages'
                )
            )
    st.session_state['messages_history'].append({'role':'assistant', 'content':ai_message})    
        