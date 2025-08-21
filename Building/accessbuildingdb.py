import json
from neo4j import GraphDatabase

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "morpheus4j"

element_type = "BuildingElement"

# Initialize Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def get_room_neighbors(session, room_label):
    """
    Finds and returns the central room, its direct neighbors, and the connections between them.
    """
    # The Cypher query already returns the relationship object 'r'
    query = """
    MATCH (room {label: $room_label})-[r]-(neighbor)
    RETURN room, COLLECT({neighbor: neighbor, relationship: r}) AS connections
    """
    result = session.run(query, room_label=room_label)

    record = result.single()

    if record:
        central_room_node = record["room"] # This is the actual Neo4j Node object
        neighbor_connections = []
        for conn in record["connections"]:
            # OPTIONAL MATCH means conn['neighbor'] and conn['relationship'] could be None
            if conn["neighbor"] is None and conn["relationship"] is None:
                continue # Skip if no connections found for this room

            neighbor_node = conn["neighbor"]      # This is the actual Neo4j Node object
            relationship_obj = conn["relationship"] # This is the actual Neo4j Relationship object
            neighbor_connections.append({
                "neighbor_node": neighbor_node,
                "relationship_obj": relationship_obj
            })
        return central_room_node, neighbor_connections
    else:
        return None, []


# --- print_room_info (updated to use Node objects) ---
def print_room_info(room_node, connections):
    """
    Prints the details of the room and its connections to the console.
    Expects room_node as a neo4j.Node object and connections as list of dicts with neighbor_node and relationship_obj.
    """
    # MATCH (room {label: '1-A'})  OPTIONAL MATCH (room)-[r]-(neighbor) return room, neighbor
    # MATCH (room {label: '1-A'})  OPTIONAL MATCH (room)<-[r]-(neighbor) return room, neighbor
    # MATCH (room {label: '1-A'})  OPTIONAL MATCH (room)-[r]->(neighbor) return room, neighbor
    if not room_node:
        print("Room not found.")
        return

    print("\n" + "="*50)
    print(f"Information for Room: {room_node.get('label', 'N/A')} (ID: {room_node.get('internalId', 'N/A')})")
    print(f"Title: {room_node.get('title', 'N/A')}")
    print(f"Value: {room_node.get('value', 'N/A')}")
    print("-" * 50)

    if connections:
        print("Connected to:")
        for conn in connections:
            neighbor_node = conn["neighbor_node"]
            relationship_obj = conn["relationship_obj"]

            print(f"  - Neighbor: {neighbor_node.get('label', 'N/A')} (ID: {neighbor_node.get('internalId', 'N/A')})")
            print(f"    Connection Type: :{relationship_obj.type}") # Access type directly
            print(f"    Connection Name: {relationship_obj.get('name', 'N/A')}")
            if 'weight' in relationship_obj:
                print(f"    Weight: {relationship_obj.get('weight')}")
            print("-" * 20)
    else:
        print("No direct connections found.")
    print("="*50 + "\n")


def generate_graph_rag_output(room_node, connections):
    """
    Generates a textual output suitable for a Graph RAG call to an LLM.
    Formats the room and its connections into a clear, structured string.

    Args:
        room_node (neo4j.Node): The central room node.
        connections (list): A list of dictionaries, each containing 'neighbor_node' (neo4j.Node)
                            and 'relationship_obj' (neo4j.Relationship).

    Returns:
        str: A formatted string representing the graph context.
    """
    if not room_node:
        return "No relevant graph context found for the requested room."

    output_lines = []
    output_lines.append(f"--- Graph Context for Room: '{room_node.get('label', 'N/A')}' ---")

    # 1. Central Node Details
    output_lines.append("\nCentral Node Details:")
    # Get labels excluding 'BuildingElement' if other specific labels exist
    #node_labels = [label for label in room_node.labels if label != "BuildingElement"]
    #output_lines.append(f"  Type(s): {', '.join(node_labels) if node_labels else 'BuildingElement'}")
    output_lines.append(f"  Type(s): {room_node.get('type', 'BuildingElement')}")
    output_lines.append(f"  Label: {room_node.get('label', 'N/A')}")
    output_lines.append(f"  Internal ID: {room_node.get('internalId', 'N/A')}")
    if room_node.get('title'):
        output_lines.append(f"  Title: {room_node['title']}")
    if room_node.get('color'):
        output_lines.append(f"  Color: {room_node['color']}")
    if room_node.get('shape'):
        output_lines.append(f"  Shape: {room_node['shape']}")
    if room_node.get('value') is not None:
        output_lines.append(f"  Value: {room_node['value']}")

    # 2. Direct Connections
    if connections:
        output_lines.append("\nDirect Connections (Triplets - Source --[RELATIONSHIP]-> Target [Properties]):")
        for conn in connections:
            neighbor_node = conn["neighbor_node"]
            relationship_obj = conn["relationship_obj"]

            # Determine direction of the relationship for the triplet
            source_node_id = relationship_obj.start_node['internalId']
            target_node_id = relationship_obj.end_node['internalId']

            # Use internalId for precise comparison
            if source_node_id == room_node['internalId']:
                source_label = room_node.get('label', 'N/A')
                target_label = neighbor_node.get('label', 'N/A')
            elif target_node_id == room_node['internalId']:
                source_label = neighbor_node.get('label', 'N/A')
                target_label = room_node.get('label', 'N/A')
            else:
                # This case should ideally not happen if query is correct
                source_label = "UNKNOWN_SOURCE"
                target_label = "UNKNOWN_TARGET"

            # Build relationship properties string
            rel_props = []
            for prop_key, prop_val in relationship_obj.items():
                # Exclude internal ID if present as a property, or other non-descriptive props
                if prop_key not in ['internalId', 'id', 'name', 'color', 'weight']:
                    rel_props.append(f"{prop_key}: {prop_val}")
            
            # Special handling for common properties
            if relationship_obj.get('name'):
                rel_props.insert(0, f"name: '{relationship_obj['name']}'") # Put name first for clarity
            if relationship_obj.get('color'):
                rel_props.append(f"color: '{relationship_obj['color']}'")
            if relationship_obj.get('weight') is not None:
                rel_props.append(f"weight: {relationship_obj['weight']}")


            rel_properties_str = f" [{', '.join(rel_props)}]" if rel_props else ""

            output_lines.append(
                f"- ({source_label}) --[:{relationship_obj.type}]--> ({target_label}){rel_properties_str}"
            )
    else:
        output_lines.append("\n  No direct connections found.")

    output_lines.append("\n--- End of Graph Context ---")
    return "\n".join(output_lines)


# --- find_room_route ---
def find_room_route(session, start_room_label, end_room_label):
    """
    Finds the shortest path between two rooms in the graph.

    Args:
        session (neo4j.Session): The Neo4j database session.
        start_room_label (str): The 'label' property of the starting room node.
        end_room_label (str): The 'label' property of the ending room node.

    Returns:
        dict: A dictionary containing path details.
              - 'path_found': True if a path was found, False otherwise.
              - 'start_room_label': The label of the start room.
              - 'end_room_label': The label of the end room.
              - 'nodes': List of neo4j.Node objects on the path (if found).
              - 'relationships': List of neo4j.Relationship objects on the path (if found).
              - 'total_weight': Sum of 'weight' property on relationships (if found and weights exist).
    """
    # Max 10 hops is a reasonable limit to prevent very long computations for complex graphs
    # or finding irrelevant paths. Adjust as needed.
    query = """
    MATCH (startNode {label: $start_room_label})
    MATCH (endNode {label: $end_room_label})
    OPTIONAL MATCH p = shortestPath((startNode)-[*..10]-(endNode))
    RETURN p
    """
    result = session.run(query, start_room_label=start_room_label, end_room_label=end_room_label)
    record = result.single()

    if record and record["p"]:
        path = record["p"] # This is a neo4j.Path object

        total_weight = sum(rel.get('weight', 0) for rel in path.relationships if rel.get('weight') is not None)

        return {
            "path_found": True,
            "start_room_label": start_room_label,
            "end_room_label": end_room_label,
            "nodes": list(path.nodes),
            "relationships": list(path.relationships),
            "total_weight": total_weight
        }
    else:
        return {
            "path_found": False,
            "start_room_label": start_room_label,
            "end_room_label": end_room_label
        }


# --- generate_route_rag_output ---
def generate_route_rag_output(route_data):
    """
    Generates a textual output suitable for a Graph RAG call to an LLM, focusing on a specific route.

    Args:
        route_data (dict): The dictionary returned by find_room_route.

    Returns:
        str: A formatted string representing the graph route context.
    """
    start_label = route_data.get('start_room_label', 'N/A')
    end_label = route_data.get('end_room_label', 'N/A')

    output_lines = []
    output_lines.append(f"--- Graph Route Context from '{start_label}' to '{end_label}' ---")

    if not route_data['path_found']:
        output_lines.append(f"  No direct path found between '{start_label}' and '{end_label}' within the search limit.")
        output_lines.append("--- End of Graph Route Context ---")
        return "\n".join(output_lines)

    nodes = route_data['nodes']
    relationships = route_data['relationships']
    total_weight = route_data.get('total_weight')

    output_lines.append(f"\nPath found from '{start_label}' to '{end_label}':")
    output_lines.append(f"  Total Hops: {len(relationships)}")
    if total_weight is not None:
        output_lines.append(f"  Total Path Weight: {total_weight}")

    # Section 1: Visual representation of the path
    output_lines.append("\nVisual Path Sequence:")
    formatted_path_elements = []
    # The path.nodes and path.relationships are ordered.
    # The first node in path.nodes is the start, then relationship, then next node, etc.
    for i in range(len(nodes)):
        node = nodes[i]
        formatted_path_elements.append(f"({node.get('label', node.get('title', 'N/A'))})") # Prefer label, then title
        if i < len(relationships):
            rel = relationships[i]
            rel_props = []
            if rel.get('name'): rel_props.append(f"name:'{rel['name']}'")
            if rel.get('weight') is not None: rel_props.append(f"weight:{rel['weight']}")
            props_str = f"[{', '.join(rel_props)}]" if rel_props else ""
            
            # Determine direction of this specific relationship segment relative to the path
            # Assuming standard shortestPath output where rels point from start to end of segment
            # For undirected relationships, this might need more robust handling
            formatted_path_elements.append(f"--[:{rel.type}]{props_str}-->")
    
    output_lines.append("  " + " ".join(formatted_path_elements))


    # Section 2: Detailed components of the path
    output_lines.append("\nDetailed Path Components:")
    for i, node in enumerate(nodes):
        #node_type = ', '.join([lbl for lbl in node.labels if lbl != 'BuildingElement']) or 'BuildingElement'
        node_type = node.get('type', 'BuildingElement')
        output_lines.append(f"  Node {i+1}: '{node.get('label', 'N/A')}' (Type: {node_type})")
        if node.get('title'): output_lines.append(f"    Title: {node['title']}")
        if node.get('internalId'): output_lines.append(f"    Internal ID: {node['internalId']}")
        if node.get('value') is not None: output_lines.append(f"    Value: {node['value']}")

    for i, rel in enumerate(relationships):
        source_label = rel.start_node.get('label', rel.start_node.get('title', 'N/A'))
        target_label = rel.end_node.get('label', rel.end_node.get('title', 'N/A'))
        output_lines.append(f"  Relationship {i+1}: ({source_label}) --[:{rel.type}]--> ({target_label})")
        if rel.get('name'): output_lines.append(f"    Name: {rel['name']}")
        if rel.get('color'): output_lines.append(f"    Color: {rel['color']}")
        if rel.get('weight') is not None: output_lines.append(f"    Weight: {rel['weight']}")

    output_lines.append("--- End of Graph Route Context ---")
    return "\n".join(output_lines)

# Start Neo4j session and import data
with driver.session() as session:
    # Example 1: Query for Room '0-1'
    room_label_to_query_1 = "0-1"
    room, connections = get_room_neighbors(session, room_label_to_query_1)
    print_room_info(room, connections)

    # Example 2: Query for B - level 1
    room_label_to_query_2 = "B - level 1"
    room, connections = get_room_neighbors(session, room_label_to_query_2)
    print_room_info(room, connections)

    # graph RAG
    room_label_to_query_2 = generate_graph_rag_output(room, connections)
    print(f"\n--- RAG Output for '{room_label_to_query_2}' ---\n{room_label_to_query_2}")

    # Example 2: Query for 0-A
    room_label_to_query_3 = "0-A"
    room, connections = get_room_neighbors(session, room_label_to_query_3)
    print_room_info(room, connections)

    # graph RAG
    room_label_to_query_3 = generate_graph_rag_output(room, connections)
    print(f"\n--- RAG Output for '{room_label_to_query_3}' ---\n{room_label_to_query_3}")

    # Example 1: Route between Room 0-1 and Room 0-3
    route_data_1 = find_room_route(session, "0-1", "0-3")
    rag_output_route_1 = generate_route_rag_output(route_data_1)
    print(f"\n--- RAG Output for Route '0-1' to '0-3' ---\n{rag_output_route_1}")

    # Example 2: Route between Corridor 0-A and Room 1-10
    route_data_2 = find_room_route(session, "0-A", "1-10")
    rag_output_route_2 = generate_route_rag_output(route_data_2)
    print(f"\n--- RAG Output for Route 'Corridor 0-A' to 'Room 1-10' ---\n{rag_output_route_2}")
# Close the driver
driver.close()
