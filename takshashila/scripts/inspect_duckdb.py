import duckdb

DB_PATH = "/Users/shlokjain/pollu/takshashila/duckdb/pollu.duckdb"

def run_queries():
    con = duckdb.connect(DB_PATH)

    # 1. List all tables
    print("=== TABLES ===")
    tables = con.execute("SHOW TABLES").fetchall()
    for t in tables:
        print(t[0])
    print()

    # 2. Describe each table
    for (table_name,) in tables:
        print(f"=== SCHEMA: {table_name} ===")
        schema = con.execute(f"DESCRIBE {table_name}").fetchdf()
        print(schema)
        print()

    # 3. Sample data from each table
    for (table_name,) in tables:
        print(f"=== SAMPLE ROWS: {table_name} ===")
        sample = con.execute(
            f"SELECT * FROM {table_name} LIMIT 5"
        ).fetchdf()
        print(sample)
        print()

    # 4. Example analytical queries (edit freely)
    print("=== ROW COUNTS ===")
    for (table_name,) in tables:
        count = con.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        print(f"{table_name}: {count}")
    print()

    # 5. Example filter / aggregation (adjust column names as needed)
    # Example assumes columns like grid_id, timestamp_hour, pm25 may exist
    try:
        print("=== EXAMPLE AGGREGATION ===")
        df = con.execute("""
            SELECT *
            FROM pm25_observation
            WHERE grid_id = 'g00000001'
            AND timestamp = TIMESTAMP '2026-01-18 16:00:00+00:00';
        """).fetchdf()
        print(df)
    except Exception as e:
        print("Aggregation query failed (schema may differ):", e)

    con.close()

if __name__ == "__main__":
    run_queries()
