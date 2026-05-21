from core.utils.login import connect_to_db
import uuid


def save_color_analysis(user_id, result, name=None):
    """
    Saves the color analysis result to the database for a specific user.
    Args:
        user_id (str): The ID of the user.
        result (str): The result of the color analysis.
        name (str, optional): An optional name for the analysis. Defaults to None.
    """
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
    """
    Retrieves all color analysis results for a specific user from the database.
    Args:
        user_id (str): The ID of the user.
    Returns:
        list: A list of tuples containing the analysis ID, name, result, and creation date.
    """
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