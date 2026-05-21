import json
import os
import hashlib
import sqlite3
import uuid


def hash_password(password):
    """
    Returns the SHA-256 hash of the given password.
    Args:
        password (str): The password to be hashed.
    Returns:
        str: The hexadecimal representation of the hashed password.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def connect_to_db(db_name):
    """
    Connects to the specified SQLite database and returns the cursor and connection objects.
    Args:
        db_name (str): The name of the database file to connect to.
    Returns:
        tuple: A tuple containing the cursor and connection objects.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    return cursor, conn

def register_user(username, password):
    """
    Registers a new user with the given username and password.
    Args:
        username (str): The desired username for the new account.
        password (str): The password for the new account.
    Returns:
        tuple: A tuple containing a boolean indicating success and a message string.
    """
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
            return False, "DB error: " + str(e)
    finally:
        conn.close()


def login_user(username, password):
    """
    Logins a user by verifying the provided username and password against the database records.
    Args:
        username (str): The username of the user trying to log in.
        password (str): The password provided for authentication.
    Returns:
        tuple: A tuple containing a boolean indicating success and a message string.
    """
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
