import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("Waiting for server to start...")
    time.sleep(10)
    
    # 1. Health Check
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {resp.status_code} - {resp.json()}")
    except Exception as e:
        print(f"Server not up yet? {e}")
        return

    # 2. Predict
    print("\n--- Testing Prediction ---")
    texts = [
        "I need an appointment with Dr. Rao",
        "How much is the bill?",
        "Where is my blood test report?",
        "The food was terrible"
    ]
    for t in texts:
        resp = requests.post(f"{BASE_URL}/ml/predict", json={"text": t})
        print(f"Input: '{t}' -> {resp.json()}")

    # 3. Ingest
    print("\n--- Testing Ingestion ---")
    ticket_ids = []
    
    # High confidence example
    resp = requests.post(f"{BASE_URL}/messages/ingest", json={
        "from": "+123456789",
        "text": "I want to schedule a checkup for tomorrow morning."
    })
    print(f"Ingested High Conf: {resp.json()}")
    if resp.status_code == 200:
        ticket_ids.append(resp.json()['id'])

    # Low confidence/ambiguous example
    resp = requests.post(f"{BASE_URL}/messages/ingest", json={
        "from": "+987654321",
        "text": "The quick brown fox jumps over the lazy dog." 
    })
    print(f"Ingested Ambiguous: {resp.json()}")
    if resp.status_code == 200:
        ticket_ids.append(resp.json()['id'])

    # 4. List Tickets
    print("\n--- Testing List Tickets ---")
    resp = requests.get(f"{BASE_URL}/tickets")
    print(f"All Tickets: {len(resp.json())} items found.")
    
    resp = requests.get(f"{BASE_URL}/tickets?status=open")
    print(f"Open Tickets: {len(resp.json())} items found.")

    # 5. Resolve Ticket
    if ticket_ids:
        t_id = ticket_ids[0]
        print(f"\n--- Resolving Ticket {t_id} ---")
        resp = requests.patch(f"{BASE_URL}/tickets/{t_id}", json={"status": "resolved"})
        print(f"Resolve Response: {resp.json()}")
        
        # Verify status
        resp = requests.get(f"{BASE_URL}/tickets")
        tickets = resp.json()
        for t in tickets:
            if t['id'] == t_id:
                print(f"Ticket {t_id} status in DB: {t['status']}")

if __name__ == "__main__":
    test_api()
