import json
from neo4j import GraphDatabase

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "morpheus4j"

# Path to your JSON Lines file
JSONL_FILE = 'building_data.jsonl'

# Initialize Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# --- Function to clear the database ---
def clear_database(tx):
    print("Clearing existing database data...")
    tx.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")

def import_building_data(tx, data):
    nodes_created = 0
    relationships_processed = 0
    relationships_created = 0
    relationships_matched_nodes_found = 0
    relationships_matched_nodes_not_found = 0
    node_type = "BuildingElement"
    for subdata in data:
        for item in subdata:
            if 'label' in item and 'id' in item:  # This is a node
                # Determine the node label based on the 'label' or 'title'
                # For simplicity, let's use a general 'BuildingElement' label,
                # and specific labels like 'Room', 'Corridor', 'Door', 'Stairs', 'Space'
                # based on the title.
                
                node_label = "Unknown"
                if 'title' in item:
                    a_name = item['title']
                else:
                    a_name = item['label']
                if True:
                    print("reading node:",a_name)
                
                    if a_name.startswith("Room"):
                        node_label = "Room"
                    elif a_name.startswith("Bath"):
                        node_label = "Bathroom"
                    elif a_name.startswith("W/C") or a_name.startswith("Women") or a_name.startswith("Men"):
                        node_label = "WC"
                    elif a_name.startswith("Dining"):
                        node_label = "Dining"
                    elif a_name.startswith("Conference") or a_name.startswith("Lounge") or a_name.startswith("Kitchen") or a_name.startswith("Daycare"):
                        node_label = "Facility"
                    elif a_name.startswith("Office"):
                        node_label = "Office"
                    elif a_name.startswith("Store") or a_name.startswith("Storage"):
                        node_label = "Storage"
                    elif a_name.startswith("Corridor"):
                        node_label = "Corridor"
                    elif a_name.startswith("Stair"):
                        node_label = "Stair"
                    elif a_name.startswith("Doors_"): # Doors have different labels
                        node_label = "Door"
                    elif a_name.startswith("Space"):
                        node_label = "Space"
                    elif a_name.startswith("Entrance"):
                        node_label = "Entrance"
                    elif a_name.startswith("Lift"):
                        node_label = "Elevator"
                    elif "Lobby" in a_name:
                        node_label = "Lobby"
                    elif 'shape' in item and item['shape']=="dot":
                        node_label = "ExternalExit"
                        
                    # Create/Merge Node
                    tx.run(f"""
                        MERGE (n:{node_type} {{internalId: $id}})
                        SET n.label = $label,
                            n.type  = $alabel,
                            n.title = $title,
                            n.color = $color,
                            n.area =  $area
                    """, alabel= node_label, id=item['id'], label=item.get('label'), title=a_name,
                         color=item.get('color'),area=item.get('value'))


            elif 'from' in item and 'to' in item:  # This is a relationship
                # Define relationship type and properties
                rel_type = "CONNECTED_VIA" # Default
                if 'title' in item:
                    relationships_processed += 1
                    rel_type = "CONNECTED_VIA" # Default relationship type

                    if item.get('color') == '#FF0000':
                        rel_type = "STAIR"
                    elif item.get('color') == '#00FF00':
                        rel_type = "DIRECT"
                    elif item.get('color') == '#FF00FF':
                        rel_type = "ESCALATOR"
                    elif item.get('color') == '#CCCCCC':
                        rel_type = "LIFT"
                    elif item.get('title', '').startswith("Lift"):
                        rel_type = "LIFTDOOR"
                    elif item.get('title', '').startswith("Doors_"):
                        rel_type = "DOOR"

                    result = tx.run(f"""
                        MATCH (a:{node_type} {{internalId: $fromId}})
                        MATCH (b:{node_type} {{internalId: $toId}})
                        WITH a, b, $title AS rel_title, $color AS rel_color, $weight AS rel_weight, '{rel_type}' AS dynamic_rel_type_str
                        MERGE (a)-[r:{rel_type}]->(b)
                        ON CREATE SET
                            r.name = $title,
                            r.color = $color,
                            r.weight = $weight
                        ON MATCH SET
                            r.name = $title,
                            r.color = $color,
                            r.weight = $weight
                        RETURN r
                    """, fromId=item['from'], toId=item['to'],
                         title=item.get('title'), color=item.get('color'), weight=item.get('weight'))
                    '''
                    # add a two way relationship
                    result = tx.run(f"""
                        MATCH (a:{node_type} {{internalId: $fromId}})
                        MATCH (b:{node_type} {{internalId: $toId}})
                        WITH a, b, $title AS rel_title, $color AS rel_color, $weight AS rel_weight, '{rel_type}' AS dynamic_rel_type_str
                        MERGE (b)-[r:{rel_type}]->(a)
                        ON CREATE SET
                            r.name = $title,
                            r.color = $color,
                            r.weight = $weight
                        ON MATCH SET
                            r.name = $title,
                            r.color = $color,
                            r.weight = $weight
                        RETURN r
                    """, fromId=item['from'], toId=item['to'],
                         title=item.get('title'), color=item.get('color'), weight=item.get('weight'))
                    '''
                    # Check if any relationship was returned (meaning nodes were found and relationship handled)
                    # The RETURN r was implicitly removed from the query for clarity, as it's not strictly needed for MERGE.
                    # We can still check if rows were affected, but the previous `result.single()` was more about if the MATCH found something.
                    # A simpler check is to count the relationships after, or rely on the warning if no match.
                    if result.single():
                        relationships_created += 1
                        relationships_matched_nodes_found += 1
                        print(f"Relationship created for from_id={item['from']}, to_id={item['to']}. ")
                    else:
                        relationships_matched_nodes_not_found += 1
                        # This print statement will help identify relationships where nodes weren't found
                        print(f"Warning: No relationship created for from_id={item['from']}, to_id={item['to']}. ")
                else:
                    print("reading arc unknown:",item['from'],item['to'])


# Read JSON Lines data
all_data = []
try:
    with open(JSONL_FILE, 'r') as f:
        print("File open")
        for line in f:
            all_data.append(json.loads(line))
except FileNotFoundError:
    print(f"Error: The file '{JSONL_FILE}' was not found. Please ensure it's in the correct directory.")
    exit()

# Start Neo4j session and import data
with driver.session() as session:
    # --- Clear the database first ---
    session.execute_write(clear_database)
    
    print("Beginning data import...")
    session.execute_write(import_building_data, all_data)
    print("Data import complete!")

# Close the driver
driver.close()
