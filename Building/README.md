

## Integration with Local LLM and Neo4j for Graph-Based RAG

These scripts can be extended to support Retrieval-Augmented Generation (RAG) using a local Large Language Model (LLM) and the Neo4j graph database.

### How It Works

- The `extractbuilding.py` script includes functions to extract graph context and route information from the Neo4j database.
The building elements and their relationships are imported from a JSON Lines file into a Neo4j database. It categorizes nodes such as rooms, corridors, doors, stairs, and spaces, and establishes relationships like DOOR, STAIR, DIRECT, and CONNECTED_VIA.
- `accessbuildingdb.py`: Accesses the NEO4J database to extract room connectivity information and computes shortest paths between rooms using NEo4J Cypher queries. It provides detailed room information, generates graph context for RAG (Retrieval-Augmented Generation), and visualizes routes.
 
- The output graph is presented in a **triplet-like format**, often used for representing knowledge graphs or relationships in a human/LLM-readable way. 
- The data output of the graph is therefore formatted into a structured text suitable for input to a local LLM.
- The LLM can then use this graph-based context to answer questions, generate summaries, or assist in navigation tasks.

### Example Use Case

1. **Graph Context Extraction**:
   - Use `generate_graph_rag_output(room_node, connections)` to create a textual representation of a room and its connections.
   - This output can be passed to an LLM to answer questions like "What rooms are connected to Room 0-1?"

2. **Route Context Extraction**:
   - Use `generate_route_rag_output(route_data)` to describe the shortest path between two rooms.
   - The LLM can use this to generate directions or explain the route.

### Benefits

- Enables intelligent reasoning over spatial data.
- Enhances chatbot or assistant capabilities with structured building knowledge.
- Supports offline and private deployments using local LLMs.

