import psycopg2
from psycopg2 import OperationalError


def create_connection():
    try:
        connection = psycopg2.connect(
            user="raguae",
            password="raguae",
            host="localhost",
            port="5432",
            database="vector_db"
        )
        print("Connection to PostgreSQL DB successful")
        return connection
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        return None

connection = create_connection()
if connection:
    connection.close()

