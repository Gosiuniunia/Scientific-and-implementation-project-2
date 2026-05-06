import json
import os
import hashlib
import sqlite3
import uuid


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def connect_to_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    return cursor, conn

def register_user(username, password):
    cursor, conn = connect_to_db("database.db")

    password_hash = hash_password(password)
    user_id = str(uuid.uuid4())

    try:
        cursor.execute("""
        INSERT INTO users (id, username, password_hash)
        VALUES (?, ?, ?)
        """, (user_id, username, password_hash))

        conn.commit()
        return True, "Account created"

    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed" in str(e):
            return False, "User already exists"
        else:
            return False,  "DB error:" + e


def login_user(username, password):
    cursor, conn = connect_to_db("database.db")

    password_hash = hash_password(password)

    cursor.execute("""
    SELECT password_hash FROM users WHERE username = ?
    """, (username,))

    result = cursor.fetchone()
    conn.close()

    if result is None:
        return False, "User not found"

    stored_hash = result[0]

    if stored_hash == password_hash:
        return True, ""
    else:
        return False, "Wrong password"
