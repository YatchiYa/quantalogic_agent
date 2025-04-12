from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

def test_connection():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "testLegalPlace"
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            print("Connection successful!")
            print(result.single()[0])
        driver.close()
    except ServiceUnavailable as e:
        print(f"Connection failed (service unavailable): {e}")
    except Exception as e:
        print(f"Connection failed: {type(e).__name__} - {str(e)}")

if __name__ == "__main__":
    test_connection()
