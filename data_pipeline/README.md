# Data Pipeline Module (`/data_pipeline`)

## What this does

Scrapes book data from books.toscrape.com, cleans it into proper types, converts prices to INR
using a fixed baseline rate, loads it into a normalized SQLite database, and runs a set of SQL
queries against it -- verified against equivalent pandas operations.

## Install

```bash
pip install -r requirements.txt
```

Requires: `requests`, `beautifulsoup4`, `pandas`

## How to run (in order)

```bash
python scrape.py      # scrapes 3 categories -> books_raw.csv
python clean.py        # cleans + converts currency -> books_clean.csv
python load_db.py       # builds SQLite schema and loads data -> books.db
python queries.py       # runs 6 SQL queries + pandas verification
```

To save the query output:
```bash
python queries.py > query_output.txt
```

## Data source

Scraped 69 books across 3 categories (Travel: 11, Mystery: 32, Historical Fiction: 26) from
books.toscrape.com, a public site built for scraping practice. No login/API key required.

## Currency conversion

**Fixed baseline rate: 1 GBP = 105.50 INR.**
This is a project-defined constant used for this assignment, not a live/historical market rate.
It requires no external API call or network access. Applied as:
`price_inr = price_gbp * 105.50`

## Cleaning decisions

- `price`: currency symbol stripped, converted to `float` -> `price_gbp`.
- `star_rating`: word rating ("One".."Five") mapped to integer 1-5 -> `rating`.
- `availability`: text checked for "in stock" -> boolean `in_stock`.
- **Missing/unparseable values**: none occurred in the final 69-row dataset (all fields parsed
  cleanly). The pipeline is defensively written to handle failures if they occur:
  - Unparseable `price_gbp` values are median-imputed.
  - Unparseable `star_rating` values are dropped, since a rating can't be meaningfully imputed.

## Database schema

Two tables with a primary/foreign key relationship:

```sql
categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL,
      rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))
```

`load_db.py` drops and recreates both tables on every run (idempotent -- safe to re-run without
creating duplicate rows).

## SQL queries (`queries.py`)

Six queries covering all required clauses:
1. `WHERE` + `ORDER BY` + `LIMIT` -- top-rated, most expensive books
2. `DISTINCT` -- distinct category IDs present
3. `BETWEEN` -- books in a mid-range GBP price band
4. `IN` -- books rated either 1 or 5 stars
5. `JOIN` -- top-rated books per category
6. `JOIN` + `GROUP BY` (bonus) -- average price per category

Query 5's result is also reproduced using `pd.merge()` on in-memory DataFrames (no SQL) and
verified equal to the `pd.read_sql()` result -- both include `title` as an explicit tiebreaker
in the sort so ties (equal rating + price) order identically in both approaches.

## Files in this module

| File | Purpose |
|---|---|
| `scrape.py` | Scrapes books.toscrape.com |
| `clean.py` | Cleans and converts scraped data |
| `load_db.py` | Builds SQLite schema and loads data |
| `queries.py` | Runs SQL queries + pandas verification |
| `books_raw.csv` | Raw scraped data |
| `books_clean.csv` | Cleaned data |
| `books.db` | SQLite database |
| `query_output.txt` | Saved output of all queries |

## Status
Module 1 complete and tested.

## Author Notes
This module was built, tested, and verified end-to-end as part of the Zepto Data & AI Platform capstone project.