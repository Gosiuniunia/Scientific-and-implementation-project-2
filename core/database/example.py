import hashlib
import uuid
import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()


password = "stud"
password_hash = hashlib.sha256(password.encode()).hexdigest()

user_id = str(uuid.uuid4())

cursor.execute("""
INSERT INTO users (id, username, password_hash)
VALUES (?, ?, ?)
""", (user_id, "stud", password_hash))

conn.commit()


analysis_id = str(uuid.uuid4())

cursor.execute("""
INSERT INTO color_analysis (id, user_id, name, result)
VALUES (?, ?, ?, ?)
""", (analysis_id, user_id, "moja analiza", "winter"))

conn.commit()


cursor.execute("""
SELECT * FROM users""")

rows = cursor.fetchall()
for row in rows:
    print(row)