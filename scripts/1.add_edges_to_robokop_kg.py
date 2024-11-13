####HELPER QUERIES
"""
//check if node id exists
MATCH (n {id: ''})
RETURN n LIMIT 1;

// check if edge exists
MATCH (a {id: ''})-[r]-(b {id: ''})
RETURN r LIMIT 1

//clear all nodes and relationships
MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r

"""

uri = "bolt://localhost:7687"  # or your Neo4j instance URI
user = "neo4j"  # replace with your username
password = "password"  # replace with your password

####FUNCTIONS
import os
import pandas as pd
from neo4j import GraphDatabase


def get_unique_entities(df, entity_cols):
    unique_entities = df.drop_duplicates(subset=entity_cols, keep='first')
    return list(unique_entities[entity_cols].itertuples(index=False, name=None))


def connect_to_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password), database="robokopkg")
    return driver


def remove_text_mining_edges(driver):
    with driver.session() as session:
        cypher = """
            MATCH ()-[r]->() 
            WHERE r.agent_type = "text_mining_agent"
            DELETE r
        """
        session.run(cypher)
    print("Text mined edges removed successfully")


def remove_subclass_edges(driver, batch_size=10000):
    with driver.session() as session:
        # Loop to delete relationships in batches
        while True:
            # Cypher query to delete a batch of relationships
            cypher = """
                MATCH ()-[r]->()
                WHERE type(r) = "biolink:subclass_of"
                WITH r LIMIT $batch_size
                DELETE r
                RETURN count(r) AS deleted_count
            """
            # Execute the Cypher query
            result = session.run(cypher, batch_size=batch_size)

            # Check if any relationships were deleted in this batch
            deleted_count = result.single()["deleted_count"]
            print(f"Deleted {deleted_count} relationships in this batch.")

            # Exit the loop if no more relationships are deleted
            if deleted_count == 0:
                break

    print("All subclass_of edges deleted.")


def create_nodes(driver, node_tuples, node_type):
    with driver.session() as session:
        for node_id, node_name in node_tuples:
            cypher = f"""
                MERGE (n:`{node_type}` {{id: $node_id}})
                ON CREATE SET n.name = $node_name
            """
            session.run(cypher, node_id=node_id, node_name=node_name)
        # print(f"Executing Cypher query:\n{cypher}")
        # print(f"With parameters: node_id={node_id}, node_name={node_name}")
    print(f"{node_type} nodes checked and created as necessary.")


def create_relationships(driver, df, node1_label, node2_label, rel_type, node1_id_field, node2_id_field,
                         knowledge_source):
    with driver.session() as session:
        for _, row in df.iterrows():
            # Extract data from each row
            node1_id = row[node1_id_field]
            node2_id = row[node2_id_field]

            # Generalized Cypher query for creating relationships
            cypher = f"""
                MATCH (n1:`{node1_label}` {{id: $node1_id}})
                MATCH (n2:`{node2_label}` {{id: $node2_id}})
                MERGE (n1)-[r:`{rel_type}`]->(n2)
                ON CREATE SET r.primary_knowledge_source = $knowledge_source
            """
            # Debugging output
            # print(f"Executing Cypher query:\n{cypher}")
            #  print(f"With parameters: node1_id={node1_id}, node2_id={node2_id}, knowledge_source={knowledge_source}\n")

            # Execute the Cypher query
            session.run(cypher, node1_id=node1_id, node2_id=node2_id, knowledge_source=knowledge_source)
    print(f"{rel_type} relationships created from DataFrame successfully.")


def remove_island_nodes(driver):
    with driver.session() as session:
        cypher = """
            MATCH (n)
            WHERE NOT (n)--()
            DELETE n
        """
        session.run(cypher)
    print("Nodes with no connections removed successfully")

def get_unique_nodes():   
    query = "MATCH (n) RETURN DISTINCT n.id AS node_id, n.name AS node_name"
    with driver.session() as session:
        result = session.run(query)
        data = [(record["node_id"], record["node_name"]) for record in result]
    df = pd.DataFrame(data, columns = ["node_id", "node_name"])
    df.to_parquet(os.path.dirname(os.getcwd())+"all_node_ids_mapped_to_node_names.parquet")

def get_therapeutic_triples():   
    query = """MATCH (c:`biolink:ChemicalEntity`)-[r0:`biolink:directly_physically_interacts_with`]-(g:`biolink:GeneOrGeneProduct`)-[r1]-(d:`biolink:DiseaseOrPhenotypicFeature`), (c)-[r2:`biolink:treats`]-(d) WHERE properties(c)["CHEBI_ROLE_pharmaceutical"] IS NOT NULL AND properties(r2)["primary_knowledge_source"] IN ["infores:drugcentral", "everycure_indication_list"] RETURN DISTINCT c.name, c.id, g.name, g.id, d.name, d.id"""
    with driver.session() as session:
        result = session.run(query)
        data = [(record["c.name"], record["c.id"], record['g.name'], record['g.id'], record['d.name'], record['d.id']) for record in result]
    df = pd.DataFrame(data, columns = ["drug_name", "drug_id", "target_name", "target_id", "disease_name", "disease_id"])
    df.to_csv(os.path.dirname(os.getcwd())+"therapeutic_triples_name_id.csv")

drug_disease_df = pd.read_csv(os.path.dirname(os.getcwd()) + '/edges_to_add/drug-disease-list-treats.csv')
unique_drugs_everycure = get_unique_entities(drug_disease_df, ['drug_id', 'drug_name'])

unique_diseases_everycure = get_unique_entities(drug_disease_df, ['disease_id', 'disease_name'])

drug_target_df = pd.read_csv(os.path.dirname(os.getcwd()) + '/edges_to_add/normalized_ttd_drug_target_edges.csv')
unique_drugs_ttd = get_unique_entities(drug_target_df, ['drug_id', 'drug_name'])

unique_targets_ttd = get_unique_entities(drug_target_df, ['target_id', 'target_name'])

target_disease_df = pd.read_csv(os.path.dirname(os.getcwd()) + '/edges_to_add/normalized_ttd_target_disease_edges.csv')
unique_targets_ttd_2 = get_unique_entities(target_disease_df, ['target_id', 'target_name'])

unique_diseases_ttd = get_unique_entities(target_disease_df, ['disease_id', 'disease_name'])

unique_drugs_all = list(set(unique_drugs_everycure + unique_drugs_ttd))
unique_targets_all = list(set(unique_targets_ttd + unique_targets_ttd_2))
unique_diseases_all = list(set(unique_diseases_ttd + unique_diseases_everycure))

if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    driver = connect_to_neo4j(uri, user, password)
    
    create_nodes(driver, unique_drugs_all, "biolink:ChemicalEntity")
    create_nodes(driver, unique_diseases_all, "biolink:DiseaseOrPhenotypicFeature")
    create_nodes(driver, unique_targets_all, "biolink:GeneOrGeneProduct")

    remove_text_mining_edges(driver)
    remove_subclass_edges(driver)

    create_relationships(driver, drug_disease_df, "biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature",
                         "biolink:treats", "drug_id", "disease_id", "everycure_indication_list")

    create_relationships(driver, drug_target_df, "biolink:ChemicalEntity", "biolink:GeneOrGeneProduct",
                         "biolink:directly_physically_interacts_with", "drug_id", "target_id",
                         "therapeutic_target_database")

    create_relationships(driver, target_disease_df, "biolink:GeneOrGeneProduct", "biolink:DiseaseOrPhenotypicFeature",
                         "biolink:target_for", "target_id", "disease_id", "therapeutic_target_database")

    remove_island_nodes(driver)
    
    # Close the connection after processing
    
    get_unique_nodes()
    get_therapeutic_triples()
    driver.close()

###RUN THIS CYPHER QUERY IN NEO4J BROWSER TO EXPORT GRAPH TO CSV
""""
CALL apoc.export.csv.query(
    "
    MATCH (n)-[r]->(m)
    RETURN n.id AS node1_id, m.id AS node2_id, type(r) AS relationship_type
    ",
    "enriched_refined_robokop_kg.csv", 
    {useTypes: true, quotes: false}
)
"""
###MOVE CSV FILE FROM:
# Neo4j Desktop/Application/relate-data/dbmss/dbms-cf8ab5ab-90f0-4054-ba35-5bd3107d01b4/import
### TO:
# this directory
