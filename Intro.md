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