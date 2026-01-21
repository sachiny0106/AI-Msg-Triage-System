import os
from datetime import datetime
from typing import List, Optional

import joblib
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# --- Configuration ---
DATABASE_URL = "sqlite:///./tickets.db"
MODEL_PATH = "models/model.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"

# --- Database Setup ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TicketDB(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, index=True) # mapped from 'from'
    text = Column(String)
    label = Column(String)
    confidence = Column(Float)
    status = Column(String, default="open")
    created_at = Column(DateTime, default=datetime.utcnow)
    triage_required = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- ML Model Loading ---
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Model not found. Predictions will fail.")
except Exception as e:
    print(f"Error loading model: {e}")

# --- Pydantic Models ---
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float

class IngestRequest(BaseModel):
    text: str
    sender: str = Field(alias="from") 

class TicketResponse(BaseModel):
    id: int
    sender: str = Field(alias="from")
    text: str
    label: str
    confidence: float
    status: str
    created_at: datetime
    triage_required: bool
    resolved_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

class ResolveRequest(BaseModel):
    status: str

# --- App & Endpoints ---
app = FastAPI(title="AI Message Triage System")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ml/predict", response_model=PredictResponse)
def predict_category(request: PredictRequest):
    if not model:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    # Predict
    prediction = model.predict([request.text])[0]
    
    # Get probability if supported
    try:
        probabilities = model.predict_proba([request.text])[0]
        confidence = float(max(probabilities))
    except AttributeError:
        confidence = 1.0
    
    return {"label": prediction, "confidence": confidence}

@app.post("/messages/ingest", response_model=TicketResponse)
def ingest_message(request: IngestRequest, db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    # ML Logic
    prediction = model.predict([request.text])[0]
    try:
        probs = model.predict_proba([request.text])[0]
        confidence = float(max(probs))
    except:
        confidence = 1.0

    triage_required = confidence < 0.7

    # Create Ticket
    db_ticket = TicketDB(
        sender=request.sender,
        text=request.text,
        label=prediction,
        confidence=confidence,
        status="open",
        triage_required=triage_required,
        created_at=datetime.utcnow()
    )
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)
    
    return db_ticket

@app.get("/tickets", response_model=List[TicketResponse])
def list_tickets(
    label: Optional[str] = None, 
    status: Optional[str] = None, 
    db: Session = Depends(get_db)
):
    query = db.query(TicketDB)
    if label:
        query = query.filter(TicketDB.label == label)
    if status:
        query = query.filter(TicketDB.status == status)
    
    return query.all()

@app.patch("/tickets/{id}")
def resolve_ticket(id: int, request: ResolveRequest, db: Session = Depends(get_db)):
    ticket = db.query(TicketDB).filter(TicketDB.id == id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    if request.status == "resolved":
        ticket.status = "resolved"
        ticket.resolved_at = datetime.utcnow()
    else:
        ticket.status = request.status
        
    db.commit()
    db.refresh(ticket)
    
    return {
        "id": ticket.id,
        "status": ticket.status,
        "resolved_at": ticket.resolved_at
    }
