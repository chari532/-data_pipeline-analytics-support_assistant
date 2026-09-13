import sqlite3
import pandas as pd

conn = sqlite3.connect("books.db")

queries = {}

queries["q_1"] = '''SELECT title, price_inr, rating FROM books WHERE rating >= 4 ORDER BY price_inr DESC LIMIT 10'''
queries["q_2"] = '''SELECT DISTINCT category_id FROM books'''
queries["q_3"] = '''SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 40 ORDER BY price_gbp ASC'''
queries["q_4"] = '''SELECT title, rating FROM books WHERE rating IN (1,5) ORDER BY rating'''
queries["q_5"] = """SELECT c.category_name, b.title, b.rating, b.price_inr
               FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
               ORDER BY b.rating DESC, b.price_inr DESC, b.title ASC
               LIMIT 10"""
queries["q_6"] = """SELECT c.category_name, ROUND(AVG(b.price_inr), 2) AS avg_price_inr, COUNT(*) AS num_books
               FROM books AS  b
               JOIN categories AS c ON b.category_id = c.category_id
               GROUP BY c.category_name
               ORDER BY avg_price_inr DESC"""

results = {}

print("=" * 70)
for name, sql in queries.items():
    print(f"\n--- {name} ---\n")
    try:
        df_result = pd.read_sql(sql, conn)
        results[name] = df_result
        print(df_result.to_string(index=False))
    except Exception as e:
        print(f"Error executing {name}: {e}")
    print("-" * 70)


print("\n" + "=" * 70)
print("VERIFYING pd.read_sql vs pd.merge EQUIVALENCE (join query)")
print("=" * 70)


books_df = pd.read_sql("SELECT * FROM books", conn)
categories_df = pd.read_sql("SELECT * FROM categories", conn)


merged_df = pd.merge(books_df, categories_df, on="category_id")
merged_df = merged_df[["category_name", "title", "rating", "price_inr"]]
merged_df = merged_df.sort_values(
    by=["rating", "price_inr","title"], ascending=[False, False, True]
).head(10).reset_index(drop=True)

sql_join_result = results["q_5"].reset_index(drop=True)

print("\n--- Result from pd.read_sql (SQL JOIN) ---")
print(sql_join_result.to_string(index=False))

print("\n--- Result from pd.merge (in-memory pandas) ---")
print(merged_df.to_string(index=False))

are_equal = sql_join_result.equals(merged_df)
print(f"\nAre the two results equivalent? {are_equal}")

conn.close()