from sql_generator import generate_sql
from database import get_connection

sql = generate_sql("What's the average price of petrol cars by year?")
conn = get_connection()
rows = conn.execute(sql).fetchall()

from backend.sql_generator import generate_sql, SQLGenerationError
from backend.database import get_connection

try:
    sql = generate_sql("What's the average price by fuel type?")
    rows = get_connection().execute(sql).fetchall()
except SQLGenerationError as e:
    print(f"Could not generate query: {e}")