# Message Triage System

A small FastAPI service that classifies hospital-related messages (appointments, billing, reports, complaints) and opens tickets in SQLite.

## Tech Stack
- FastAPI backend
- Scikit-learn (TF-IDF + Logistic Regression)
- SQLite persistence

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data and train the model:
   ```bash
   python generate_data.py
   python train.py
   ```

3. Run the API:
   ```bash
   python -m uvicorn app:app --reload
   ```
   The server listens on http://127.0.0.1:8000.

## API Endpoints

- `POST /ml/predict`: Run classification without storing a ticket.
- `POST /messages/ingest`: Create a ticket; low-confidence predictions (< 0.7) are flagged for manual triage.
- `GET /tickets`: List tickets, optionally filter by `label` or `status`.
- `PATCH /tickets/{id}`: Update ticket status (e.g., mark as resolved).

Example ingest request:
```bash
curl -X POST http://127.0.0.1:8000/messages/ingest \
     -H "Content-Type: application/json" \
     -d '{"from": "+123", "text": "I need a doctor"}'
```

On the synthetic dataset, the model reaches roughly 0.86 macro F1.
