import sqlite3
import datetime
from database.connection import hash_password, SQLITE_DB_PATH

db_path = str(SQLITE_DB_PATH)
print("Connecting to DB:", db_path)

con = sqlite3.connect(db_path)
cur = con.cursor()

# 1. Clear activity logs completely
cur.execute("DELETE FROM user_activity_logs")
print("Cleared user_activity_logs")

# 2. Clear all user sessions
cur.execute("DELETE FROM user_sessions")
print("Cleared user_sessions")

# 3. Clear all users
cur.execute("DELETE FROM users")
print("Cleared all users")

# 4. Reset autoincrement sequence counters
try:
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('users', 'user_activity_logs', 'user_sessions')")
    print("Reset sqlite_sequence counters")
except Exception as e:
    print("Sequence reset notice:", e)

# 5. Insert fresh admin user
pwd_hash, salt = hash_password("admin123")
now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
cur.execute(
    "INSERT INTO users (id, username, password_hash, salt, role, created_at) VALUES (1, ?, ?, ?, ?, ?)",
    ("admin", pwd_hash, salt, "admin", now_str)
)
print("Created fresh admin user (id=1, username=admin, role=admin)")

con.commit()
con.execute("VACUUM")
con.close()
print("Database reset completed successfully!")
