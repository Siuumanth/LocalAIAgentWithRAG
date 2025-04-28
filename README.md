


### Pizza Restaurant RAG Bot
This is a simple Retrieval-Augmented Generation (RAG) project where you can ask questions about a pizza restaurant, and get answers based on real customer reviews. 
It reads a CSV file of restaurant reviews, embeds them into a vector database (Chroma) using Ollama's mxbai-embed-large model, and retrieves the most relevant reviews when you ask a question.
These reviews are then passed to a Llama3.2 model to generate a detailed answer. It’s a basic but complete setup that shows how RAG works — grounding an LLM’s response on actual data instead of letting it guess.

Followed  [Tech with Tims tutorial](https://www.youtube.com/watch?v=E4l91XKQSgw&pp=0gcJCYQJAYcqIYzv)
