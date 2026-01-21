import pandas as pd
import random

categories = {
    "appointment": [
        "I need to book an appointment with Dr. Rao",
        "Can I schedule a visit for tomorrow?",
        "Is Dr. Smith available next Monday?",
        "Booking an appointment for cardiology",
        "I want to see a general physician",
        "Need a slot for 10 AM",
        "Appointment for vaccination",
        "I want to book a checkup",
        "Is the dermatologist available?",
        "Pls cancel my appointment",
        "Reschedule appointment to next week?",
        "Do you have any openings this afternoon",
        "Urgent appointment needed for back pain",
        "I need to consult for fever",
        "Book a dental cleaning",
        "Eye checkup appointment",
        "Pediatrician appointment for my child",
        "Orthopedic consultation booking",
        "ENT specialist appointment",
        "Neurologist visit scheduling",
        "Gynaecologist appointment request",
        "Psychiatrist consultation booking",
        "Follow-up appointment booking",
        "New patient registration and appointment",
        "Emergency appointment request"
    ],
    "billing": [
        "How much strictly is the surgery?",
        "Send me the invoice for my last visit",
        "Is insurance accepted here?",
        "Payment issue with credit card",
        "What is the cost of an MRI",
        "Charged twice for consultation",
        "Billing department contact number",
        "Do you accept cash payments?",
        "Cost of blood test",
        "Refund request for cancelled appointment",
        "My bill seems incorrect",
        "Can I pay online?",
        "Insurance claim status",
        "Outstanding balance inquiry",
        "Itemized bill request",
        "Co-payment amount details",
        "Discount for senior citizens?",
        "Payment receipt not received",
        "Installment payment options",
        "Billing query regarding lab tests",
        "Consultation fee details",
        "Room charges per day",
        "Pharmacy bill clarification",
        "Surgery package price",
        "Late payment penalty?"
    ],
    "reports": [
        "I haven't received my blood test report yet",
        "When will X-ray results be ready",
        "Email me the lab report",
        "Download link for my MRI scan",
        "Status of my pathology report",
        "Is the CT scan report generated?",
        "Hard copy of the report needed",
        "Report collection timings",
        "Covid test result inquiry",
        "Urine culture report delay",
        "Access reports online",
        "My report has a spelling mistake",
        "Biopsy result waiting time",
        "Send reports to my doctor directly",
        "Glucose test results",
        "Thyroid profile report status",
        "Lipid profile results",
        "Liver function test report",
        "Kidney function test report",
        "Vitamin D test result",
        "Hemoglobin test report",
        "Dengue test result status",
        "Typhoid test report availability",
        "Ultrasound report collection",
        "Endoscopy report status"
    ],
    "complaint": [
        "The waiting time was too long",
        "Rude behavior by the receptionist",
        "Hospital premises were not clean",
        "Doctor didn't explain diagnosis well",
        "Parking issue",
        "Pharmacy didn't have the medicines",
        "Nurse was not attentive",
        "AC was not working in the waiting area",
        "Food quality in the cafeteria was poor",
        "Billing process is too slow",
        "Appointment was delayed by 2 hours",
        "Unprofessional staff conduct",
        "Washrooms were dirty",
        "Noise level in the ward was high",
        "Bed linen was not changed",
        "Water dispenser empty",
        "Security guard was rude",
        "TV in the room not working",
        "Call button not responding",
        "Wrong medicine given at pharmacy",
        "Disappointed with the service",
        "Manager needed for a complaint",
        "Fan making noise in the room",
        "Mosquitoes in the lobby",
        "Slippery floor hazard"
    ]
}

data = []
for label, examples in categories.items():
    for ex in examples:
        data.append({"text": ex, "label": label})
        
    # Add variations to ensure dataset size > 100 and help generalization
    for ex in examples: 
         data.append({"text": ex + " please", "label": label})
         data.append({"text": "Hi, " + ex, "label": label})

df = pd.DataFrame(data)
# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

print(f"Generated {len(df)} rows.")
print(df["label"].value_counts())

df.to_csv("data/messages.csv", index=False)
print("Saved to data/messages.csv")
