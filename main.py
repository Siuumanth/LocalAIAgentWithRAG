from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a 
pizza restaurant, here are some relevant reviews: 
{reviews}

Here is the question: {question}
"""

prompt = PromptTemplate.from_template(template=template)

chain = prompt | model #Invoking the chain 

while True:
    print('\n\n\n--------------------------------------\n')
    question = input("What is your question? (q to quit)")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result =chain.invoke({"reviews":reviews, "question": question})
    print(result)