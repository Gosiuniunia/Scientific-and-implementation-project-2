from core.utils.login import connect_to_db
import uuid


def save_color_analysis(user_id, result, name=None):
    cursor, conn = connect_to_db("database.db")

    analysis_id = str(uuid.uuid4())

    cursor.execute("""
    INSERT INTO color_analysis (id, user_id, name, result)
    VALUES (?, ?, ?, ?)
    """, (analysis_id, user_id, name, result))

    conn.commit()
    conn.close()

    print("Analiza zapisana")


def get_user_analyses(user_id):
    cursor, conn = connect_to_db("database.db")

    cursor.execute("""
    SELECT id, name, result, created_at
    FROM color_analysis
    WHERE user_id = ?
    ORDER BY created_at DESC
    """, (user_id,))

    results = cursor.fetchall()
    conn.close()

    return results