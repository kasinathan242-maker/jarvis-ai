from memory.db import get_connection

def get_recent_conversations(limit: int = 5) -> list:
    conn = get_connection()
    cursor = conn.execute(
        'SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?',
        (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def search_memory(keyword: str) -> list:
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM memories WHERE key LIKE ? OR value LIKE ?",
        (f'%{keyword}%', f'%{keyword}%')
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
