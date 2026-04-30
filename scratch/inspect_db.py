import sqlite3
import os

db_path = '../data/incidents.db'

if not os.path.exists(db_path):
    print(f"File {db_path} not found.")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Tables in database:")
        for table in tables:
            table_name = table[0]
            print(f"\n--- Table: {table_name} ---")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print("Columns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            rows = cursor.fetchall()
            print("\nSample Data (up to 5 rows):")
            for row in rows:
                print(f"  {row}")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")
