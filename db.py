import sqlite3
import threading

DB_PATH = "people_count.db"
lock = threading.Lock()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS people_count (
            id INTEGER PRIMARY KEY,
            total_in INTEGER DEFAULT 0,
            total_out INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("INSERT OR IGNORE INTO people_count (id) VALUES (1)")
    conn.commit()
    conn.close()

def increment_in():
    with lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE people_count SET total_in = total_in + 1")
        conn.commit()
        conn.close()

def increment_out():
    with lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE people_count SET total_out = total_out + 1")
        conn.commit()
        conn.close()

def read_counts():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT total_in, total_out FROM people_count WHERE id=1")
    row = c.fetchone()
    conn.close()
    return row if row else (0, 0)
