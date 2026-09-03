from memory.db import get_connection

def save_conversation(user_input: str, intent: str, response: str) -> None:
    conn = get_connection()
    conn.execute(
        'INSERT INTO conversations (user_input, intent, response) VALUES (?, ?, ?)',
        (user_input, intent, response)
    )
    conn.commit()
    conn.close()

def save_memory(category: str, key: str, value: str) -> None:
    conn = get_connection()
    conn.execute(
        'INSERT OR REPLACE INTO memories (category, key, value) VALUES (?, ?, ?)',
        (category, key, value)
    )
    conn.commit()
    conn.close()
