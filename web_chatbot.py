import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def query_llm(question):
    """Query the LLM with a question and return the response"""
    response = llm_chain.invoke({"question": question})
    return response["text"]

def main():
    print("Welcome to the AI Assistant! Type 'quit' to exit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            break
        response = query_llm(user_input)
        print("\nAssistant:", response)

if __name__ == "__main__":
    main()
