

## Integration with Local LLM and Neo4j for Graph-Based RAG

These scripts can be extended to support Retrieval-Augmented Generation (RAG) using a local Large Language Model (LLM) and the Neo4j graph database.

### How It Works

- The `extractbuilding.py` script includes functions to extract building graph information and store it to a Neo4j Graph database.
The building elements and their relationships are imported from a JSON Lines file into a Neo4j database. It categorizes nodes such as rooms, corridors, doors, stairs, and spaces, and establishes relationships like DOOR, STAIR, DIRECT, and CONNECTED_VIA.
- `accessbuildingdb.py`: Accesses the NEO4J database to extract room connectivity information and computes shortest paths between rooms using NEo4J Cypher queries. It provides detailed room information, generates graph context for RAG (Retrieval-Augmented Generation), and visualizes routes.
 
- The output graph is presented in a **triplet-like format**, often used for representing knowledge graphs or relationships in a human/LLM-readable way. 
- The data output of the graph is therefore formatted into a structured text suitable for input to a local LLM.
- The LLM can then use this graph-based context to answer questions, generate summaries, or assist in navigation tasks.

### Benefits of representing a building using a graph database

- Enables intelligent reasoning over spatial data.
- Enhances chatbot or assistant capabilities with structured building knowledge.
- Supports offline and private deployments using local LLMs.

