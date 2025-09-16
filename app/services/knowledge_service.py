import spacy
from typing import List, Dict
from neo4j import GraphDatabase

class KnowledgeService:
    def __init__(self, uri, user, password):
        """
        Initializes the knowledge service.
        - Loads the spaCy NLP model.
        - Establishes a connection to the Neo4j database.
        """
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            print("Error: spaCy model 'en_core_web_sm' not found.")
            self.nlp = None

    def close(self):
        """Closes the database connection driver."""
        self._driver.close()

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extracts named entities from a block of text using spaCy.
        """
        if not self.nlp:
            return []

        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_
            })
        return entities

    def ingest_entities_and_relationships(self, entities: List[Dict], document_name: str):
        """
        Ingests entities and creates relationships to a central document node.
        """
        with self._driver.session(database="kmrl") as session:
            session.run("MERGE (d:Document {name: $doc_name})", doc_name=document_name)
            
            for entity in entities:
                # --- THIS IS THE CORRECTED QUERY ---
                session.run(
                    """
                    MATCH (d:Document {name: $doc_name})
                    MERGE (e:%s {name: $entity_name})
                    MERGE (e)-[:MENTIONED_IN]->(d)
                    """ % entity['label'],
                    entity_name=entity['text'],
                    doc_name=document_name
                )
        print(f"Successfully ingested {len(entities)} entities and their relationships for document '{document_name}'.")

# --- Test Block for the Knowledge Service ---
if __name__ == "__main__":
    # --- IMPORTANT: UPDATE YOUR NEO4J DETAILS HERE ---
    NEO4J_URI = "neo4j://localhost:7687"  # This is usually the default
    NEO4J_USER = "neo4j"                 # This is the default username
    NEO4J_PASSWORD = "p2cXjyEQUWRoumIThsMJKz1othrWa54YFDefG2k2h24"   # <-- CHANGE THIS to the password you created
    
    # Initialize the service with your database credentials
    service = KnowledgeService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    if service.nlp:
        sample_text = """
        Since its first commercial run in 2017, KMRL has grown into a complex enterprise.
        John Doe from the Engineering department submitted a report to the Ministry of Housing & Urban Affairs
        in Kochi last week regarding a purchase from a vendor named Acme Corp.
        """
        
        # 1. Extract entities from the text
        print("\n--- Extracting Entities ---")
        extracted_entities = service.extract_entities(sample_text)
        for entity in extracted_entities:
            print(f"- Text: '{entity['text']}', Type: '{entity['label']}'")
        
        # 2. Ingest the extracted entities into Neo4j
        print("\n--- Ingesting into Neo4j ---")
        document_id = "Sample.pdf"
        service.ingest_entities_and_relationships(extracted_entities, document_id)
    # 3. Close the database connection
    service.close()
    print("\nDatabase connection closed.")