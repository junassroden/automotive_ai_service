
import json
from pathlib import Path

dataset_path = Path("data/chatbot_intents.json")

new_patterns = [
    "Which maintenance components should I inspect first?",
    "What servicing does this car need?",
    "Can you check this vehicle for maintenance issues?",
    "What should I prioritize when servicing this vehicle?",
    "What maintenance should I perform on this car?",
    "What parts of my vehicle need inspection?",
    "Which parts should I service first?",
    "What vehicle components need maintenance?",
    "Check this car for possible maintenance problems",
    "Analyze this vehicle's maintenance needs",
    "Tell me what needs to be serviced on this vehicle",
    "What should I replace or inspect on this car?",
    "Which components need attention on this vehicle?",
    "Can you predict what maintenance this car needs?",
    "Give me a maintenance prediction for this vehicle",
    "What are the recommended repairs or inspections?",
    "Identify the maintenance needs of this vehicle",
    "Can you evaluate this car's maintenance requirements?",
    "What should my mechanic check on this vehicle?",
    "Which parts might require servicing soon?",
    "What maintenance work is recommended for this car?",
    "Help me determine which parts need maintenance",
    "Review this vehicle for recommended maintenance",
    "What should I have checked during the next service?",
    "Can you assess this car's maintenance condition?",
    "Find the components that need inspection",
    "Tell me the recommended maintenance for my car",
    "Which car parts should I check for wear?",
    "Predict the maintenance requirements of my vehicle",
    "What service does my vehicle require?",
    "Can you identify possible maintenance concerns?",
    "Give me the maintenance recommendations for this car",
    "What should I inspect on my vehicle at this mileage?",
    "Which components should I prioritize for servicing?",
    "Analyze my car and recommend maintenance",
    "What parts should be checked during vehicle servicing?",
    "Help me plan maintenance for this vehicle",
    "Determine which components may need attention",
    "Can you run a maintenance assessment on this car?",
    "What maintenance checks are recommended for my vehicle?"
]

with dataset_path.open("r", encoding="utf-8") as file:
    dataset = json.load(file)

intents = dataset["intents"]
target_intent = next(
    (intent for intent in intents if intent["tag"] == "vehicle_prediction"),
    None
)

if target_intent is None:
    raise ValueError("The vehicle_prediction intent was not found.")

existing = {
    pattern.strip().casefold()
    for pattern in target_intent["patterns"]
}

added = 0

for pattern in new_patterns:
    if pattern.casefold() not in existing:
        target_intent["patterns"].append(pattern)
        existing.add(pattern.casefold())
        added += 1

with dataset_path.open("w", encoding="utf-8") as file:
    json.dump(dataset, file, indent=2, ensure_ascii=False)

print("Vehicle prediction training data updated.")
print(f"New patterns added: {added}")
print(f"Total vehicle prediction patterns: {len(target_intent['patterns'])}")
