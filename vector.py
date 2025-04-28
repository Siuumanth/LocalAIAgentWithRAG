from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


df = pd.read_csv("realistic_restaurant_reviews.csv")

#defining the embedding model from ollama
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")


db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)  # to ensure it runs only once

# runs if the database doesn't exist in the location
if add_documents:
    documents = []
    ids = []

    for i,row in df.iterrows():
        document = Document (
            page_content = row ["Title"] + " " + row ["Review"], #search column 
            metadata = {
                "rating": row ["Rating"],
                "date": row["Date"]
        })

        ids.append(str(i))
        documents.append(document)    

 #creation by conversion to documents

# initialising vector store using chroma
vector_store = Chroma(
    collection_name = "restaurant_reviews",
    persist_directory = db_location,
    embedding_function = embeddings
)

# adding data to the vector store if not added 
# it will automatically embed all our data for us and add to vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)


# making our vector store usable by LLM
# retriever will allow us to retrieve documents
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)