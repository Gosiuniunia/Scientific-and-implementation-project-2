import pytest
import sqlite3
import os
from core.utils.login import register_user, login_user, hash_password, connect_to_db
from core.utils.history import save_color_analysis, get_user_analyses

@pytest.fixture(autouse=True)
def manage_db():
    """
    Fixture zarządza bazą danych: tworzy tabele przed testem i czyści je po teście.
    """
    db_name = "database.db"
    
    # 1. Inicjalizacja tabel (jeśli nie istnieją)
    cursor, conn = connect_to_db(db_name)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, 
            username TEXT UNIQUE, 
            password_hash TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS color_analysis (
            id TEXT PRIMARY KEY, 
            user_id TEXT, 
            name TEXT, 
            result TEXT, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close() # Zamykamy, aby nie blokować pliku

    yield # Tu wykonują się testy

    # 2. Czyszczenie danych po każdym teście (zamiast usuwania pliku)
    cursor, conn = connect_to_db(db_name)
    cursor.execute("DELETE FROM users")
    cursor.execute("DELETE FROM color_analysis")
    conn.commit()
    conn.close()

def test_registration_and_login_flow():
    # Test rejestracji
    success, msg = register_user("test_user", "password123")
    assert success is True
    
    # Test logowania
    success, msg = login_user("test_user", "password123")
    assert success is True
    
    # Test błędnego hasła
    success, msg = login_user("test_user", "wrong_password")
    assert success is False
    assert msg == "Wrong password"

def test_history_saving_and_loading():
    user_id = "unique_user_id"
    
    # Test zapisu
    save_color_analysis(user_id, "spring", "Mój test")
    
    # Test odczytu
    history = get_user_analyses(user_id)
    assert len(history) == 1
    assert history[0][1] == "Mój test"
    assert history[0][2] == "spring"

def test_hash_password():
    password = "secret_password"
    hashed = hash_password(password)
    assert hashed != password
    assert len(hashed) == 64  # SHA-256 zawsze ma 64 znaki w hex

def test_register_existing_user():
    register_user("duplicate", "pass1")
    success, msg = register_user("duplicate", "pass2")
    assert success is False
    assert "already exists" in msg

def test_login_non_existent_user():
    """Pokrycie linii 54: próba logowania użytkownika, którego nie ma w bazie."""
    success, msg = login_user("non_existent_person", "any_password")
    assert success is False
    assert msg == "User not found"

def test_db_error_handling():
    """TODO linijka 36"""

    success, msg = register_user( None, "pass")
    assert success is False
    return 