import os
import sys
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

# enable these tow lines to see langchain debug information
#from langchain.globals import set_debug
#set_debug(True)

# code to just output the message sent to the LLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Enable the following line to see message sent to the LLM
OUTPUT_LLM_MESSAGE = False

# --- Configuration Constants ---
TEXT_FILE_PATH = "./sample_data.txt"

# LLM Settings
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_INFERENCE_MODEL = "llama3.2:latest"

# Graph Database Settings
NEO4J_URI = "bolt://localhost:1480"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "morpheus4j"

# Vector Database Settings
PERSIST_DIRECTORY = os.path.join(".", "chroma_db")
COLLECTION_NAME = "collection_name"

# LLMGraphTransformer Settings
# These should to be adjusted depending on the type of document imported - these for sample_data.txt
ALLOWED_GRAPH_NODES = [
    "Person",
    "Organization",
    "Location",
    "Concept",
    "Project",
    "Team",
    "Strategy", # More specific than 'Concept' for 'conservation strategies'
    "Career" # For 'began his career'
]

NODE_PROPERTIES = [
    "name",          # For names of people, organizations, locations, projects, teams
    "description",   # For more detail about concepts, strategies, projects
    "type",          # To distinguish types of concepts or strategies if needed
    "area",          # For locations (e.g., "East African")
    "role",          # For a person's role (e.g., "head")
    "focus",         # For conservation strategies (e.g., "wildlife")
    "source"         # From LangChain's default, good for document origin
]
ALLOWED_GRAPH_RELATIONSHIPS = [
    "MENTORS",
    "LOCATED_IN",
    "MENTIONS"
    "HEADS",
    "STUDIED_IN", # Ethan began his career studying wildlife IN the Serengeti
    "MANAGES_FROM",
    "HAS_STRATEGY", # For an organization having a strategy
    "PART_OF",      # For a person being part of a team, or a team being part of an organization
    "WORKS_IN",     # For a person working in a location
    "FOCUSES_ON",   # For a person or team focusing on a concept/strategy
    "ASSOCIATED_WITH" # A general relationship for less specific connections
]

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".md": (UnstructuredMarkdownLoader, {"mode": "elements"}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

# --- Global Variables (to be initialized) ---
neoj_graph: Neo4jGraph = None
vector_retriever = None
entity_chain = None


# --- Pydantic Models ---
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="Identify all the key entities and main talking points that "
        "appear in the provided text content.",
    )
# --- Helper Functions ---
def load_single_document(file_path: str) -> List[Document]:
    """Loads a single document based on its file extension."""
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")


def initialize_neo4j_driver():
    """Initializes and returns a Neo4j GraphDatabase driver."""
    try:
        neo4j_driver = GraphDatabase.driver(
            uri=NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        neo4j_driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        return neo4j_driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)


def clear_neo4j_graph(neo4j_driver):
    """Clears all nodes and relationships from the Neo4j database."""
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Neo4j graph cleared.")


def load_documents_from_file(file_path: str) -> List[Document]:
    """Loads documents from a specified markdown or text file."""
    print(f"Ingesting: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return load_single_document(file_path)


def generate_and_save_neo4j_graph(
    documents: List[Document], neo4j_graph_client: Neo4jGraph, llm: ChatOllama
) -> bool:
    """
    Generates a graph from documents using an LLMGraphTransformer and saves it to Neo4j.
    """
    print("Text splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=24, separators=["\n\n", "\n", " "]
    )
    split_documents = text_splitter.split_documents(documents)

    print("Building graph from split documents...")
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_GRAPH_NODES,
        node_properties=NODE_PROPERTIES,
        allowed_relationships=ALLOWED_GRAPH_RELATIONSHIPS
    )
    graph_documents = transformer.convert_to_graph_documents(split_documents)

    print("Full graph data: ",graph_documents)

    print("Saving graph to Neo4j...")
    neo4j_graph_client.add_graph_documents(
        graph_documents, baseEntityLabel=True, include_source=True
    )
    print("Neo4j graph generation and save complete.")
    return True


def generate_neo4j_vector_retriever(
    embedding_function: OllamaEmbeddings,
) -> Neo4jVector:
    """Generates a vector index from the existing Neo4j graph."""
    print("Generating Neo4j vector index...")
    vector_index = Neo4jVector.from_existing_graph(
        embedding_function,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )
    return vector_index.as_retriever()


def generate_full_text_query(input_string: str) -> str:
    """
    Generates a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspellings.
    """
    words = [el for el in remove_lucene_chars(input_string).split() if el]
    if not words:
        return ""
    
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print("full_text_query:",full_text_query)
    return full_text_query.strip()


def retrieve_graph_data(
    neo4j_graph_client: Neo4jGraph, entity_extraction_chain, question: str
) -> str:
    """
    Collects the neighborhood of entities mentioned in the question from the Neo4j graph.
    """
    result_data = []
    entities = entity_extraction_chain.invoke({"question": question})
    print("question:", question)
    print("entities:", entities)
    for entity in entities.names:
        response = neo4j_graph_client.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result_data.extend([el["output"] for el in response])
    
    return "\n".join(result_data)


def full_retriever(question: str) -> str:
    """Combines graph-based and vector-based retrieval."""
    global neoj_graph, vector_retriever, entity_chain
    print("====== Graph Data ====== ")
    graph_data = retrieve_graph_data(neoj_graph, entity_chain, question)
    print("graph_data:", graph_data)
    print("====== ========== ====== ")
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]

    #print("vector_data:",vector_data)
    final_data = f"""Graph data:
{graph_data}
Vector data:
{"#Document ".join(vector_data)}
"""
    return final_data


def store_chroma_embedding(docs: List[Document], embedding_function: OllamaEmbeddings):
    """Creates and stores a Chroma vectorstore locally."""
    print("Storing embeddings in Chroma DB...")
    chroma_client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY, settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    chroma_client.reset()  # Clear all data in the vector database
    docs = filter_complex_metadata(docs)
    Chroma.from_documents(
        docs,
        embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        client=chroma_client,
    )
    print("Chroma DB embedding storage complete.")


def load_chroma_retriever(embedding_function: OllamaEmbeddings):
    """Loads a Chroma retriever from the persisted directory."""
    print("Loading Chroma DB retriever...")
    chroma_client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY, settings=Settings(anonymized_telemetry=False)
    )
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        client=chroma_client,
    )
    return db.as_retriever()


# --- Main Functions ---
def ingest_documents_to_dbs(llm: ChatOllama):
    """
    Ingests documents, generates a Neo4j graph, and creates a vector store.
    """
    global neoj_graph

    neo4j_driver = initialize_neo4j_driver()
    clear_neo4j_graph(neo4j_driver)

    print("Starting document processing...")
    docs = load_documents_from_file(TEXT_FILE_PATH)  # Using TEXT_FILE_PATH as per original
    
    print("Generating and saving Neo4j graph...")
    generate_and_save_neo4j_graph(docs, neoj_graph, llm)
    
    print("Generating Neo4j vector store...")
    global vector_retriever  # Declare as global to update the module-level variable
    vector_retriever = generate_neo4j_vector_retriever(
        OllamaEmbeddings(model=OLLAMA_INFERENCE_MODEL)
    )
    
    print("Creating full-text index in Neo4j...")
    neoj_graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
    )
    
    print("Storing embeddings in Chroma DB...")
    store_chroma_embedding(docs, OllamaEmbeddings(model=OLLAMA_INFERENCE_MODEL))

    neo4j_driver.close()
    print("Document ingestion complete.")


def ask_question_with_rag(llm: ChatOllama, question: str):
    """
    Answers a question using a Retrieval-Augmented Generation (RAG) approach
    with Neo4j graph and Chroma vector store.
    """
    global neoj_graph, vector_retriever, entity_chain

    print("Loading vector store (Chroma DB)...")
    embedding_function = OllamaEmbeddings(model=OLLAMA_INFERENCE_MODEL)
    vector_retriever = load_chroma_retriever(embedding_function)

    print("Building entity extraction prompt...")
    entity_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting key entities and talking points from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    entity_chain = entity_prompt | llm.with_structured_output(Entities)

    print(f"The question: {question}")
    print("Getting sub-graph and vector data...")

    rag_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    if OUTPUT_LLM_MESSAGE:
        def print_inputs_for_llm(input_dict: dict) -> dict:
            """Helper function to print the context and question before LLM."""
            print("\n--- Context and Question for LLM ---")
            print("Context:")
            print(input_dict.get("context"))
            print("\nQuestion:")
            print(input_dict.get("question"))
            print("-------------------------------------\n")
            return input_dict # Pass the input through unchanged

        rag_chain = (
            {
                "context": full_retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(print_inputs_for_llm) # This will receive the dictionary {"context": ..., "question": ...}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
    else:
        rag_chain = (
            {
                "context": full_retriever,
                "question": RunnablePassthrough(),
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )

    print("Now asking LLM for the answer...")
    response = rag_chain.invoke(input=question)
    print("\n--- LLM Response ---")
    print(response)
    print("--------------------")


def main():
    """Main function to handle command-line arguments and orchestrate the workflow."""
    global neoj_graph

    the_question = "What are the conclusions of this document?"
    ingest_data = False
    arg_error = True

    # Check command line arguments
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "ingest":
            ingest_data = True
            arg_error = False
        elif action == "ask":
            ingest_data = False
            if len(sys.argv) > 2:
                the_question = sys.argv[2]
                arg_error = False

    # If arguments are missing or invalid, show help and exit
    if len(sys.argv) == 1 or arg_error:
        print("Please enter an action.")
        print("\nUsage:")
        print("  ingest: generate graph and vector database from document")
        print(
            "  ask [question]: use LLM to interrogate graph database with a custom question"
        )
        sys.exit(1)

    print("Application starting...")

    # Initialize LLM connection
    try:
        llm = ChatOllama(
            base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
            model=OLLAMA_INFERENCE_MODEL,
            temperature=0,
        )
        # Verify LLM connection (e.g., by a simple invoke)
        llm.invoke("Hello") 
        print("Successfully connected to ChatOllama.")
    except Exception as e:
        print(f"Failed to connect to ChatOllama: {e}")
        sys.exit(1)

    # Initialize Neo4jGraph client
    try:
        neoj_graph = Neo4jGraph(
            url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
        )
        # Verify Neo4jGraph connection (e.g., by a simple query)
        neoj_graph.query("MATCH () RETURN 1 LIMIT 1")
        print("Successfully connected to Neo4jGraph.")
    except Exception as e:
        print(f"Failed to connect to Neo4jGraph: {e}")
        sys.exit(1)

    if ingest_data:
        ingest_documents_to_dbs(llm)
    else:
        ask_question_with_rag(llm, the_question)

    neoj_graph.close()
    print("Application finished.")


if __name__ == "__main__":
    main()
