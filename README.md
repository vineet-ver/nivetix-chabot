# 🚀 NIVETIX LLM-FREE CHATBOT ENGINE

This is a production-grade, 100% LLM-Free hybrid NLP Semantic Search engine. It bypasses massive OpenAI API fees by substituting LLM generation with a highly-tuned **Scikit-Learn (TF-IDF + LinearSVC)** Intent Classifier and a **FAISS Euclidean vector knowledge store**. It perfectly handles Hinglish, major typos, and slang.

## ⚙️ Architecture Requirements
Works flawlessly on Python 3.9 through Python 3.13. No rigid ML dependencies.

## 🚀 Setup Instructions

### 1. Initialize the Environment
Open a terminal in the `chatbot-backend` directory and run:
```bash
# Create a virtual environment
python -m venv venv

# Activate it
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build the FAISS Concept Index
This parses your `../public/knowledge.json` and mathematically projects it into high-dimensional vectors.
```bash
python faiss_indexer.py
```
*You will only ever need to run this command when you manually update the `knowledge.json` file in your Next.js directory!*

### 3. Train the NLP Intent Model
We must train the Scikit-Learn logic vector mapping using our `nlu.yml` (packed with specific syntax and broken English mappings).
```bash
python intent_classifier.py
```
*This will generate a highly optimized `intent_model.pkl` in your directory in under a second.*

### 4. Deploy the Server
```bash
source venv/Scripts/activate
uvicorn main:app --reload
```
*This exposes FastAPI port 8000 locally.*

---

## 💻 API Usages Overview

### Primary Chat Endpoint
**POST** `http://localhost:8000/chat`
```json
{
    "message": "websit kitne ka",
    "session_id": "client_1234"
}
```

*Expected output: The system will hit Preprocessing, map `websit` to `website` and `kitne ka` to `price`, fire through Rasa, lock onto `ask_price`, and output the domain pricing tier automatically!*

### Diagnostics Endpoint
**GET** `http://localhost:8000/health`
