
import json
from pathlib import Path

dataset_path = Path("data/chatbot_intents.json")

new_patterns = {
    "brakes": [
        "How do I check my brakes?",
        "Why are my brakes making a squealing noise?",
        "How can I tell if my brake pads are worn?",
        "When should I inspect my braking system?",
        "What are signs of brake problems?",
        "My car vibrates when I brake. What should I check?",
        "Why does my brake pedal feel soft?",
        "How do I know if my brakes need servicing?",
        "What should I check if my car takes longer to stop?",
        "Can you explain common brake maintenance?",
        "My brakes are grinding. What should I do?",
        "How often should the braking system be inspected?",
        "What can cause a car to pull while braking?",
        "Which brake components require regular inspection?",
        "How can I recognize a brake fluid problem?",
        "What are the warning signs of failing brakes?",
        "Should I replace worn brake pads immediately?",
        "Why does my vehicle shake when stopping?",
        "How can I maintain my car's braking system?",
        "What brake issues should I ask a mechanic to inspect?"
    ],
    "cooling": [
        "Why is my engine overheating?",
        "How do I check the engine coolant?",
        "What causes a car engine to run hot?",
        "When should I replace engine coolant?",
        "How can I identify cooling system problems?",
        "Why is my coolant level dropping?",
        "What should I inspect if the temperature rises?",
        "How do I maintain the radiator?",
        "What are the signs of a coolant leak?",
        "Why is steam coming from under the hood?",
        "How often should the radiator be inspected?",
        "What does an engine temperature warning mean?",
        "How can I prevent engine overheating?",
        "What cooling components need regular maintenance?",
        "Why does my car lose coolant?",
        "How do I recognize a faulty cooling system?",
        "What should I check if the engine temperature fluctuates?",
        "Can a radiator problem damage an engine?",
        "What maintenance does the cooling system need?",
        "When should a mechanic inspect the radiator?"
    ],
    "maintenance_schedule": [
        "What maintenance schedule should I follow?",
        "When should I service my car?",
        "How often should a vehicle receive maintenance?",
        "What routine maintenance does a car need?",
        "When is my next vehicle service due?",
        "How often should I change engine oil?",
        "What should be checked during regular servicing?",
        "Can you explain a basic car maintenance schedule?",
        "How often should I inspect my vehicle?",
        "What maintenance should I do every month?",
        "What needs to be checked during a periodic service?",
        "When should I schedule preventive maintenance?",
        "How do I plan routine vehicle servicing?",
        "Which maintenance tasks should be done regularly?",
        "What should be included in a service checklist?",
        "How frequently should I maintain my car?",
        "What is a recommended vehicle service interval?",
        "How can I organize my vehicle maintenance schedule?",
        "What routine checks should I perform on my car?",
        "When should I bring my vehicle to a mechanic?"
    ],
    "safety": [
        "What safety checks should I perform before driving?",
        "What should I inspect before a long road trip?",
        "How do I check whether my car is safe to drive?",
        "What should I do if a warning light appears?",
        "What vehicle problems mean I should stop driving?",
        "How can I check my tires before driving?",
        "What should I inspect before using a rental car?",
        "Which vehicle safety items should I check regularly?",
        "What should I do if my brakes suddenly stop working properly?",
        "How can I recognize an unsafe vehicle condition?",
        "What checks should I make before starting a trip?",
        "When should I avoid driving my vehicle?",
        "What should I do if my engine overheats while driving?",
        "How do I perform a basic vehicle safety inspection?",
        "What are important pre-trip vehicle checks?",
        "How can I check my headlights and brake lights?",
        "What should I do if I notice a serious mechanical problem?",
        "What safety equipment should I keep in my car?",
        "How do I identify problems that make driving dangerous?",
        "What vehicle checks help prevent roadside breakdowns?"
    ],
    "capabilities": [
        "What can this AI chatbot do?",
        "How can you help me with car maintenance?",
        "What features does this automotive assistant have?",
        "Can you explain what this AI system is for?",
        "What tasks can you help me with?",
        "Can you predict vehicle maintenance needs?",
        "Can you analyze my fleet?",
        "Can you help me understand prediction scores?",
        "What automotive questions can I ask you?",
        "How does this chatbot help fleet managers?",
        "Can you recommend vehicle inspections?",
        "What kinds of maintenance advice can you provide?",
        "Can you help me prioritize vehicle servicing?",
        "What can this system analyze?",
        "How do I use this automotive AI assistant?",
        "Can you answer questions about brakes and fluids?",
        "Can you help with preventive maintenance planning?",
        "What information can I get from this chatbot?",
        "Can this assistant explain vehicle maintenance recommendations?",
        "What are the main functions of this AI assistant?"
    ]
}

with dataset_path.open("r", encoding="utf-8") as file:
    dataset = json.load(file)

intents = dataset["intents"]
added_by_intent = {}

for tag, patterns in new_patterns.items():
    target = next(
        (intent for intent in intents if intent["tag"] == tag),
        None
    )

    if target is None:
        raise ValueError(f"Intent not found: {tag}")

    existing = {
        pattern.strip().casefold()
        for pattern in target["patterns"]
    }

    added = 0

    for pattern in patterns:
        normalized = pattern.strip().casefold()

        if normalized not in existing:
            target["patterns"].append(pattern)
            existing.add(normalized)
            added += 1

    added_by_intent[tag] = {
        "added": added,
        "total": len(target["patterns"])
    }

with dataset_path.open("w", encoding="utf-8") as file:
    json.dump(dataset, file, indent=2, ensure_ascii=False)

print("Training dataset expanded successfully.")

for tag, result in added_by_intent.items():
    print(
        f"{tag}: added {result['added']}, "
        f"total patterns {result['total']}"
    )
