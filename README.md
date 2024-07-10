# RAGTAG: 
### Streamlining Trip Planning with AI-powered Chatbot
![bloom-visualisation](https://github.com/zazzane/RAGTAG/assets/111720488/769f99c5-cc79-498f-b2de-5ba26e7c329f)

RAGTAG is a trip planning chatbot designed to help you leverage the power of past travel experiences for future adventures. It utilizes a Retrieval-Augmented Generation (RAG) approach powered by the gpt4o large language model (LLM) to analyze a corpus of trip reports. By interacting with RAGTAG, you can:

1. Gain insights from past trips: Ask detailed questions about past destinations you've visited and uncover interesting facts you might have missed. </br>
2. Focus on specific aspects: Explore information related to local government programs, green initiatives, and their impact on the environment, society, and the economy of a city.</br>
3. Save time on research: RAGTAG analyzes vast amounts of data to provide you with the information you need quickly and efficiently.</br>

### Core Technologies:

Database: Neo4j (hybrid search: vector similarity search + knowledge graph) - Neo4j serves as the foundation for storing and connecting data points extracted from trip reports. It utilizes a combination of vector similarity search for efficient retrieval and a knowledge graph to understand relationships between data points, providing a more comprehensive picture of a destination.</br>

Large Language Model (LLM): GPT-4o -  RAGTAG employs GPT-4o, an openAI model, to understand the nuances of user queries and generate insightful responses based on the retrieved information from the Neo4j database. It was also used to generate graph documents from raw document data.</br>

### Libraries:

OpenAI Embeddings - This library assisted in generating numerical representations of text data from trip reports, enabling GPT-4o to process and understand the information more effectively.</br>

Langchain & Langgraph: While currently experimental, Langchain and Langgraph libraries were used to further enhance the capabilities of RAGTAG in areas like natural language processing, prompt engineering and knowledge graph construction.</br>

### Additional Exploration (Experimental):

LLamaindex & GROQ API LLMs (Mistral): Explored using LLamaindex and GROQ API to integrate additional LLMs like Mistral into RAGTAG. This could potentially expand the range of functionalities and capabilities offered by the chatbot.</br>

Google T5 LLM: Explored the use of Google T5 LLM to investigate alternative large language models that might offer different strengths or functionalities compared to GPT-4o.</br>
