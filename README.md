# Document-Based RAG System with Neo4j, Chroma, and Ollama

This repository hosts a Python-based Retrieval-Augmented Generation (RAG) system designed to answer questions by leveraging both a knowledge graph and a vector database. It uses Neo4j for structured knowledge representation, Chroma DB for semantic search, and Ollama for local Large Language Model (LLM) inference and embeddings.

## Features

  * **Intelligent Document Ingestion:** Processes `.md` (Markdown) and `.txt` (Plain Text) files, extracting key entities and relationships to build a comprehensive knowledge graph.
  * **Hybrid Retrieval Mechanism:** Combines the power of:
      * **Neo4j Knowledge Graph:** For retrieving structured relationships and entity neighborhoods relevant to your query.
      * **Chroma Vector Store:** For semantic similarity search on document chunks, ensuring contextually rich information retrieval.
  * **Local LLM Integration:** Utilizes [Ollama](https://ollama.ai/) to run open-source Large Language Models (LLMs) locally for both graph transformation and generating informed answers, maintaining data privacy and control.
  * **Extensible Graph Schema:** Configurable `ALLOWED_GRAPH_NODES`, `NODE_PROPERTIES`, and `ALLOWED_GRAPH_RELATIONSHIPS` to tailor the knowledge graph structure to your specific domain.
  * **Command-Line Interface:** Simple commands for ingesting new documents or asking questions directly from your terminal.

## Getting Started

Follow these steps to set up and run the RAG system locally.

### Prerequisites

Before you begin, ensure you have the following installed and running:

1.  **Python 3.9+:**
    ```bash
    python --version
    ```
2.  **Ollama:** Download and install Ollama from [ollama.ai](https://ollama.ai/).
      * Once installed, pull the LLM model specified in `OLLAMA_INFERENCE_MODEL` (e.g., `llama3.2:latest`).
        ```bash
        ollama pull llama3.2:latest
        ```
3.  **Neo4j Desktop or Docker:**
      * **Neo4j Desktop:** Download from [neo4j.com/download](https://neo4j.com/download/). Create a new local graph database instance.
      * **Docker (Recommended for quick setup):**
        ```bash
        docker run \
            --name neo4j-rag \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/<YOUR_NEO4J_PASSWORD> \
            -e NEO4J_dbms_connector_bolt_listen_address=:1480 \
            -e NEO4J_dbms_connector_http_listen_address=:1480 \
            -e NEO4J_dbms_connector_https_listen_address=:1480 \
            -e NEO4J_dbms_security_auth__enabled=false \
            -d neo4j:latest
        ```
        **Note:** The provided code uses port `1480` for Neo4j. Adjust your Docker run command or the `NEO4J_URI` constant in `constants.py` if your Neo4j instance uses a different port. Also, ensure `NEO4J_USERNAME` and `NEO4J_PASSWORD` in the code match your Neo4j credentials. For `NEO4J_dbms_security_auth__enabled=false`, you might need to adjust based on your Neo4j version and security settings, as it disables authentication.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure you have a `requirements.txt` file listing all dependencies like `langchain`, `neo4j`, `chromadb`, `ollama`, `unstructured`, `langchain-experimental`, `langchain-community`, `pydantic`).

### Configuration

Open the Python script and adjust the constants at the top of the file to match your setup:

```python
# --- Configuration Constants ---
MARKDOWN_PATH = "Markdown/FinalPaper.md" # Path to your main markdown document
TEXT_FILE_PATH = "./sample_data.txt" # Path to your sample text document

# LLM Settings
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_INFERENCE_MODEL = "llama3.2:latest" # Ensure this model is pulled in Ollama

# Graph Database Settings
NEO4J_URI = "bolt://localhost:1480" # Match your Neo4j instance URI
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "morpheus4j" # Your Neo4j password

# Vector Database Settings
PERSIST_DIRECTORY = os.path.join(".", "chroma_db") # Directory to store Chroma DB
COLLECTION_NAME = "my_rag_collection" # Name for your Chroma collection

# LLMGraphTransformer Settings
# These should to be adjusted depending on the type of document imported - these for sample_data.txt

ALLOWED_GRAPH_NODES = [
    "Person", "Organization", "Location", "Concept", "Project", "Team",
    "Strategy", "Career", "Technology", "Benefit", "Outcome", "Solution", "Problem", "Highlight", "Fact", "Tabel", "Figure", "Graph"
]
NODE_PROPERTIES = ["name", "description", "source", "type", "area", "role", "focus"]
ALLOWED_GRAPH_RELATIONSHIPS = [
    "MENTORS", "LOCATED_IN", "HEADS", "STUDIED_IN", "MANAGES_FROM",
    "HAS_STRATEGY", "PART_OF", "WORKS_IN", "FOCUSES_ON", "ASSOCIATED_WITH",
    "WROTE", "MENTIONS", "RELATED_TO"
]
```

## Usage

### 1\. Ingest Documents

This step will load your specified documents, create a knowledge graph in Neo4j, and build a Chroma vector store.

```bash
python graph_neo4j.py ingest
```

Example output:
```
Application starting...
Successfully connected to ChatOllama.
Successfully connected to Neo4jGraph.
Successfully connected to Neo4j.
Neo4j graph cleared.
Starting document processing...
Ingesting: ./sample_data.txt
Generating and saving Neo4j graph...
Text splitting documents...
Building graph from split documents...
Saving graph to Neo4j...
Neo4j graph generation and save complete.
Generating Neo4j vector store...
Generating Neo4j vector index...
Creating full-text index in Neo4j...
Storing embeddings in Chroma DB...
Storing embeddings in Chroma DB...
Chroma DB embedding storage complete.
Document ingestion complete.
Application finished.
```

### 2\. Ask Questions

After ingestion, you can query the system:

```bash
python graph_neo4j.py ask "who is in charge?"
```
Using the sample text file provide this would generate something like this:
```
Application starting...
Successfully connected to ChatOllama.
Successfully connected to Neo4jGraph.
Loading vector store (Chroma DB)...
Loading Chroma DB retriever...
Building entity extraction prompt...
The question: who is in charge?
Getting sub-graph and vector data...
Now asking LLM for the answer...
====== Graph Data ====== 
question: who is in charge?
entities: names=['Who', 'is', 'in', 'charge?']
full_text_query: Who~2
full_text_query: is~2
full_text_query: in~2
full_text_query: charge~2
graph_data: Mia - MENTORS -> Olivia
Mia - MENTORS -> Ethan
Wildlife Guardians - LOCATED_IN -> Nairobi
Wildlife Guardians - MENTORS -> Mia
Wildlife Guardians - MENTORS -> Mia
Mia - MENTORS -> Olivia
Mia - MENTORS -> Ethan
Wildlife Guardians - LOCATED_IN -> Nairobi
Wildlife Guardians - MENTORS -> Mia
Wildlife Guardians - MENTORS -> Mia
====== ========== ====== 

--- LLM Response ---
The question "who is in charge?" can be answered based on the context provided. 

According to the graph data, Mia is a mentor for Olivia and Ethan.

However, according to the vector data, there are several individuals mentioned as being in charge or holding leadership positions:

- Sophia heads the East African conservation team.
- Ava directs the Asian outreach program from New Delhi.
- Benjamin works closely with Sophia on habitat restoration efforts and reports directly to the Manaus office.
- Lucas spearheads anti-poaching initiatives throughout Kenya.

However, it is not explicitly stated who is in charge overall.
--------------------
Application finished.
```

Replace the question with your desired query. If no question is provided, it defaults to "What are the conclusions of this document?".

## Project Structure

```
.
├── graph_neo4j.py            # Main Python script for the RAG system
├── sample_data.txt           # Example plain text document
├── chroma_db/                # Directory where Chroma DB embeddings will be persisted
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```
