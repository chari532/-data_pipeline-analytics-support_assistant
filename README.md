### Multi-Module Data & AI Platform

This repository contains a comprehensive multi-module data and artificial intelligence platform structured as a single unified project. It consists of three functional modules situated at the root directory, coordinated via a centralized orchestration interface. 

### Project Structure

text

.
├── requirements.txt         # Consolidated global dependencies (Stated Choice)
├── main.py                  # Central platform orchestrator script
├── README.md                # Root project documentation (This file)
│
├── data_pipeline/           # Module 1: Books Web Scraper & SQLite Database
│   ├── scrape.py            # Web scraper for books.toscrape.com
│   ├── clean.py             # Data cleaning and currency normalization
│   ├── load_db.py           # SQLite database schema initialization and loading
│   └── queries.py           # SQL execution and pandas verification engine
│
├── analytics/               # Module 2: Titanic Predictive Modeling Engine
│   ├── eda.py               # Exploratory Data Analysis and visualization
│   └── modeling.py          # Machine learning classification and regression pipeline
│
└── support_assistant/       # Module 3: Zepto Local GenAI RAG Service
    ├── ingest.py            # Local document chunking and vector indexing
    ├── graph.py             # LangGraph-driven routing and generation orchestration
    └── main.py              # FastAPI application server interface

Use code with caution.

### Setup & Installation

### Dependency Strategy Notice

* **Chosen Approach:** This project uses a single **consolidated root-level requirements.txt** file to handle all sub-system dependencies simultaneously.

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

### How to Run the Modules

You can execute each module individually through standard execution paths, or manage them automatically via the centralized root orchestrator. 

### Centralized Orchestration (Recommended)

To run or launch any of the three modules from a single unified interface, execute the root coordinator tool: 

bash

python main.py

Use code with caution.

### Manual Individual Execution

### Module 1: Data Pipeline

Processes book data sequentially through extraction, transformations, and persistence: 

bash

python data_pipeline/scrape.py
python data_pipeline/clean.py
python data_pipeline/load_db.py
python data_pipeline/queries.py

Use code with caution.

### Module 2: Analytics Engine

Performs data profiling, produces visualizations, and trains the machine learning classifiers: 

bash

python analytics/eda.py
python analytics/modeling.py

Use code with caution.

### Module 3: Support Assistant

Indexes local knowledge base documents and hosts the REST API web framework: 

bash

# Vectorize local text documents into ChromaDB
python support_assistant/ingest.py

# Spin up the local FastAPI service deployment
uvicorn support_assistant.main:app --host 127.0.0.1 --port 8000 --reload

Use code with caution.

Once the service is active, visit the interactive API dashboard directly at http://127.0.0.1:8000/docs.
