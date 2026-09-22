### Multi-Module Data & AI Platform

This repository contains a comprehensive multi-module platform structured as a single unified project. It consists of three independent functional modules situated at the root directory, coordinated via a centralized orchestration interface. 

### Project Structure

text

.
├── requirements.txt         # Consolidated global dependencies (Stated Choice)
├── main.py                  # Central platform orchestrator script
├── README.md                # Root project documentation (This file)
│
├── data_pipeline/           # Module 1: Books Web Scraper & SQLite Database
│   ├── scrape.py            # Scrapes books.toscrape.com into books_raw.csv
│   ├── clean.py             # Cleans data & converts currency into books_clean.csv
│   ├── load_db.py           # Builds SQLite schema and loads cleaned data
│   └── queries.py           # Runs 6 SQL queries + equivalent pandas verification
│
├── analytics/               # Module 2: Titanic Predictive Modeling Engine
│   ├── eda.py               # Loads, profiles, cleans data -> titanic.csv + plots/
│   └── modeling.py          # Pipelines, classifiers, tuning, and regression side-task
│
└── support_assistant/       # Module 3: Zepto Local GenAI RAG Service
    ├── ingest.py            # Local document chunking and vector indexing to ChromaDB
    ├── graph.py             # LangGraph-driven routing and generation orchestration
    └── main.py              # FastAPI application server interface

Use code with caution.

### Setup & Installation

### Dependency Strategy Notice

* **Chosen Approach:** This project uses a single **consolidated root-level requirements.txt** file to handle all sub-system dependencies simultaneously. No individual, separate requirements files are used per module.

To set up the complete execution environment, run the following commands: 

bash

# 1. Clone the repository
git clone <your-public-github-repo-link>
cd <repo-name>

# 2. Create and activate a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install all dependencies across all three modules
pip install -r requirements.txt

Use code with caution.

### Module Overview & Execution Instructions

You can manage and run all three systems automatically via the centralized root orchestrator by executing: 

bash

python main.py

Use code with caution.

Alternatively, you can navigate into individual directories or target the files manually as detailed below: 

### Module 1: Data Pipeline (/data_pipeline)

* **What it does:** Scrapes book data from books.toscrape.com, cleans raw elements, converts prices to INR using a fixed baseline rate (1 GBP = 105.50 INR), loads data into a normalized SQLite database, and verifies SQL queries against equivalent pandas operations.
* **How to run (in order):** 

bash

python data_pipeline/scrape.py       # Scrapes 3 categories -> books_raw.csv
python data_pipeline/clean.py        # Cleans + converts currency -> books_clean.csv
python data_pipeline/load_db.py      # Builds SQLite schema and loads data -> books.db
python data_pipeline/queries.py      # Runs 6 SQL queries + pandas verification

Use code with caution.
* **To save query output:** 

bash

python data_pipeline/queries.py > data_pipeline/query_output.txt

Use code with caution.

### Module 2: Analytics Engine (/analytics)

* **What it does:** Profiles and cleans the classic Titanic dataset, tells a visual data story about survival factors, and executes a full machine learning modeling pipeline (Logistic Regression, Decision Trees, Random Forests) alongside an imbalance-handling assessment and a fare prediction regression task.
* **How to run (in order):** 

bash

python analytics/eda.py              # Part A: Profiling, cleaning, charts -> titanic.csv
python analytics/modeling.py         # Part B: Preprocessing, training, tuning, saving pipeline

Use code with caution.

*Note: If internet access is unavailable to download the dataset during evaluation, modeling.py fallback logic automatically reads the locally committed offline titanic.csv text asset.*

### Module 3: Support Assistant (/support_assistant)

* **What it does:** A complete Retrieval-Augmented Generation (RAG) assistant for Zepto utilizing a local dictionary vector index in ChromaDB, orchestrated by LangGraph, and served through schema-validated JSON FastAPI endpoints.
* **How to run (in order):** 

bash

# Step A: Embed the policy documents locally into ChromaDB
python support_assistant/ingest.py

# Step B: Spin up the FastAPI service webserver
uvicorn support_assistant.main:app --host 127.0.0.1 --port 8000 --reload

Use code with caution.
* **Interactive UI:** Open your browser and navigate to http://127.0.0.1:8000/docs to test endpoints via the visual Swagger interface.
* **Docker Deployment alternative:** 

bash

docker build -t zepto-support-assistant support_assistant/
docker run -p 7860:7860 zepto-support-assistant

Use code with caution.
