##### Tutorial - [Tech with Tim](https://www.youtube.com/watch?v=E4l91XKQSgw)

### **What is Ollama?**

✅ **Ollama is a tool that lets you easily run AI models (LLMs) _locally_ on your own computer.**  
Like — no cloud, no API calls to OpenAI — just **your laptop or PC** runs the AI model directly.

You can load models like:

- **Llama 2** (by Meta)
- **Mistral 7B**
- **Gemma (Google's open models)**
- **TinyLlama** (small models for fast work)


We will be installing `llama3.2` and `mxbai-embed-large` for embedding

##### **Llama 3.2 (in Ollama)**:  
Meta's latest open-source large language model (LLM), good at reasoning, coding, and conversations. You are downloading it through Ollama to use as your main AI "brain."


##### **mxbai (for Embeddings)**:  
MXBAI (short for "MixBai") is an embedding model.  
Embeddings are how text gets converted into numerical vectors, allowing the AI to **search**, **retrieve**, and **compare** chunks of information quickly.  
You are using MXBAI to handle this conversion part, important for your RAG (Retrieval-Augmented Generation) system.

---
### What is **Embedding** in more detail?

- **Text is complicated** for a computer. It can't "read" words like humans.
- So we **convert** words, sentences, or even documents into **vectors** (a list of numbers) that represent the **meaning** of the text.
- These vectors are usually **hundreds or thousands** of numbers long.

Once the text is converted into vectors, we can:
- **Search** for similar texts (like Google search but smarter).
- **Cluster** related information together.
- **Feed** it into machine learning models.
    

---

### Why do we use Embeddings in **RAG**?

In RAG (Retrieval-Augmented Generation), the steps look like:
1. You ask a question → `"What is the capital of France?"`
2. The system **embeds** your question into a vector.
3. It **searches** a vector database (of pre-embedded documents) for the closest matches.
4. It finds `"Paris is the capital of France."`
5. It sends that information to the LLM (Llama 3.2) to **help it answer you correctly**.

Without embeddings, the AI would have **no fast way** to find relevant documents.

---

### A simple real-world analogy:

- **Text** = messy raw data (like a library full of unorganized books).
- **Embedding** = like giving each book a smart barcode that encodes what it's about.
- **Search** = now you can scan barcodes to find the right books very fast.

---

### Types of Embedding Models:
- **mxbai**: A fast, lightweight embedding model you are using.
- **OpenAI Embeddings**: Expensive but powerful (need API key).
- **Hugging Face models**: Tons of free ones (some small, some big).

You chose **mxbai** because it runs **locally**, is **fast**, and works well for normal text tasks.

---
### Quick Visual Flow:


`Text ---------------> Embedding (Vector) ------------> Search/Retrieve "dog"                 [0.2, -0.4, 0.8, ...]             Finds related info`


---

# Actual working and code:

# 🍕 Pizza Restaurant Reviews RAG (Retrieval-Augmented Generation) Project

---

## main.py

```python
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from vector import retriever

# Load the LLM model (Llama3.2) via the Ollama interface
model = OllamaLLM(model="llama3.2")

# Define the prompt template to inject retrieved reviews and the user question
template = """
You are an expert in answering questions about a
pizza restaurant, here are some relevant reviews:
{reviews}

Here is the question: {question}
"""

# Create a PromptTemplate object from the string template
prompt = PromptTemplate.from_template(template=template)

# Chain the prompt and model so that input can be piped directly to the model
chain = prompt | model

# Start an interactive loop to continuously take user input and generate answers
while True:
    print('\n\n\n--------------------------------------\n')
    question = input("What is your question? (q to quit)")
    if question == "q":
        break

    # Retrieve relevant reviews based on the user question
    reviews = retriever.invoke(question)
    
    # Fill the prompt with reviews and the user question, then invoke the model
    result = chain.invoke({"reviews": reviews, "question": question})

    # Output the model's response
    print(result)
```

---

## vector.py

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the CSV file containing realistic pizza restaurant reviews
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize the embedding model for converting text into vector representations
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define the path for the local Chroma vector database
db_location = "./chroma_langchain_db"

# Check if the database already exists; if not, documents need to be added
add_documents = not os.path.exists(db_location)

# If documents need to be added, convert each CSV row into a Document object
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],  # Combine title and review text
            metadata={
                "rating": row["Rating"],  # Attach metadata: rating
                "date": row["Date"]       # Attach metadata: review date
            }
        )
        ids.append(str(i))  # Use the row index as the unique ID
        documents.append(document)

# Initialize the Chroma vector store with embeddings and persistence settings
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# If database was not previously created, add the documents and embed them
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Convert the vector store into a retriever object for similarity search during queries
retriever = vector_store.as_retriever(
    search_type="similarity",           # Use similarity search
    search_kwargs={"k": 5}               # Retrieve top 5 most relevant reviews
)
```

---

# ✨ How This Retrieval-Augmented Generation (RAG) System Works

This project is a Retrieval-Augmented Generation (RAG) based question-answering system focused on pizza restaurant reviews. Here's the complete flow:

1. **Data Preparation**:
    
    - The project starts by reading a CSV file (`realistic_restaurant_reviews.csv`) containing review data.
        
    - Each review (Title + Review) is combined and wrapped inside a `Document` object.
        
    - Metadata such as `rating` and `date` are attached to each document for potential future use.
        
    - Using the `mxbai-embed-large` model, each document is converted into an embedding (numerical vector).
        
    - These vectors are stored persistently in a local Chroma vector database.
        
2. **Retrieval**:
    - When a user asks a question, the retriever searches the vector database for the top 5 documents that are semantically closest to the user's query.
        
    - This is based on vector similarity, not just keyword matching.
        
3. **Prompt Construction**:
    - The retrieved relevant reviews are inserted into a pre-defined prompt template.
        
    - This template clearly provides context to the language model about what information is available.
        
4. **LLM Invocation**:
    - The prompt (filled with context and question) is then passed to the Llama3.2 model through Ollama.
        
    - The model generates a detailed, factual, and context-aware answer based on the retrieved information.
        
5. **Interactive Loop**:
    - The system runs in a continuous loop, allowing the user to ask multiple questions interactively.
        
    - The loop only stops when the user explicitly quits.
        

Thus, the LLM is no longer hallucinating or guessing blindly. It grounds its answers based on real, retrieved restaurant review data, significantly improving factual correctness, relevance, and user trust.

---