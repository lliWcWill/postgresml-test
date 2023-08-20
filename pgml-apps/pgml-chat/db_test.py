import psycopg2
from psycopg2 import OperationalError

# Replace with your DATABASE_URL
DATABASE_URL = "postgres://postgresml:4rjt68h3ek5rukv@ee6d42ff-c236-4946-bf10-b5155ca572ba.db.cloud.postgresml.org:38600/digestionfield"

try:
    # Connect to the database
    conn = psycopg2.connect(DATABASE_URL)
    print("Successfully connected to the database!")
except OperationalError as e:
    print(f"The error '{e}' occurred")

finally:
    # Close the connection
    if conn:
        conn.close()
        print("Database connection closed.")

