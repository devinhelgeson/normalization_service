import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
MIGRATIONS_DIR = Path("db/migrations")

def run_migrations():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    for migration_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        print(f"🔧 Running {migration_file.name}...")
        with open(migration_file, "r", encoding="utf-8") as f:
            sql = f.read()
            cur.execute(sql)
            conn.commit()

    cur.close()
    conn.close()
    print("✅ All migrations applied.")

if __name__ == "__main__":
    run_migrations()