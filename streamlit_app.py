import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def initialize_chatbot():
    """Initialize the chatbot components"""
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.7)
    
    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""You are a helpful AI assistant. Please respond to the following question:
        
        Question: {question}
        
        Answer:"""
    )
    
    # Create an LLM chain
    return LLMChain(llm=llm, prompt=prompt_template)

def get_response(chain, question):
    """Get response from the chatbot"""
    response = chain.invoke({"question": question})
    return response["text"]

def main():
    # Set page configuration
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="centered"
    )
    
    # Add header
    st.title("🤖 AI Chatbot")
    st.markdown("Ask me anything and I'll try to help!")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(st.session_state.chatbot, prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add sidebar with information
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This is a simple AI chatbot built with:
        - Streamlit
        - LangChain
        - OpenAI GPT-3.5
        
        You can ask questions and get AI-powered responses!
        """)
        
        # Add clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 