import sqlite3
import pandas as pd

DB_PATH = "books.db"


def create_schema(conn):
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        )
    """)
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER REFERENCES categories(category_id)
        )
    """)
    conn.commit()
    print("Schema created: categories, books")


def load_data(conn, csv_path="books_clean.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["category"])
    unique_categories = sorted(df["category"].unique())
    categories_df = pd.DataFrame({"category_name": unique_categories})
    categories_df.index += 1
    categories_df.index.name = "category_id"
    categories_df.to_sql("categories", conn, if_exists="append", index=True)
    cat_map = {name: idx for idx, name in enumerate(unique_categories, start=1)}
    df["category_id"] = df["category"].map(cat_map)
    books_df = df[["title", "price_gbp", "price_inr", "rating", "category_id","in_stock"]].copy()
    books_df["in_stock"] = df["in_stock"].astype(int)
    books_df.index += 1
    books_df.index.name = "book_id"
    books_df.to_sql("books", conn, if_exists="append", index=True)
    conn.commit()
    print(f"Loaded {len(categories_df)} categories and {len(books_df)} books into DB")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    load_data(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM books")
    print("books count:", cursor.fetchone())
    cursor.execute("SELECT COUNT(*) FROM categories")
    print("categories count:", cursor.fetchone())
    cursor.execute("SELECT DISTINCT category_id FROM books")
    print("distinct category_ids:", cursor.fetchall())
    conn.close()
    print(f"Database saved as {DB_PATH}")