
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "chatbot_intents.json"

ADDITIONAL_PATTERNS = {
    "greeting": [
        "hello there",
        "hi assistant",
        "hey chatbot",
        "good evening",
        "good day",
        "hello automotive assistant",
        "hi can you help me",
        "hey I need some help",
        "good morning assistant",
        "is anyone there",
        "hello I have a question",
        "hi there how are you",
        "can I ask you something",
        "are you available to help",
        "let us get started"
    ],
    "vehicle_prediction": [
        "which maintenance components should I inspect first",
        "what should I check on this car first",
        "can you assess this vehicle for maintenance",
        "identify possible maintenance needs for this car",
        "check this car for recommended servicing",
        "what parts may need attention on this vehicle",
        "run a maintenance assessment for this car",
        "give me a maintenance recommendation for this vehicle",
        "what should the mechanic inspect on this car",
        "help me assess the condition of this vehicle",
        "find maintenance items that may need servicing",
        "which parts should I prioritize for inspection",
        "review this vehicle and suggest maintenance",
        "can you check what servicing this car needs",
        "evaluate this vehicle using its mileage and details",
        "what maintenance work should I consider for this car",
        "tell me which components may need attention",
        "check the maintenance needs of my vehicle",
        "inspect this vehicle using the available data",
        "what should I prioritize when servicing this vehicle"
    ],
    "fleet_analysis": [
        "which fleet vehicles should be serviced first",
        "rank the cars by maintenance priority",
        "show me vehicles that may need attention",
        "analyze maintenance needs across all vehicles",
        "which fleet units have the highest priority",
        "help me prioritize workshop inspections",
        "compare maintenance priorities across the fleet",
        "which vehicles should go to the workshop",
        "identify high priority vehicles in my fleet",
        "review the fleet for possible maintenance needs",
        "sort vehicles by recommended maintenance priority",
        "which cars need attention before the others",
        "give me a fleet maintenance overview",
        "help me decide which vehicle to service first",
        "analyze all available vehicles for maintenance",
        "which fleet units should I inspect soon",
        "prioritize maintenance across the rental vehicles",
        "show the vehicles with the most maintenance recommendations",
        "help plan inspections for the whole fleet",
        "compare the vehicles that need servicing"
    ],
    "explain_score": [
        "how should I interpret this model score",
        "why did the system recommend this component",
        "what does the confidence value tell me",
        "does a high score mean the part is broken",
        "how reliable is this maintenance prediction",
        "what is the difference between a score and a diagnosis",
        "why is this component marked as a priority",
        "how does the system calculate its recommendation",
        "does this percentage guarantee a failure",
        "what does a low prediction score mean",
        "how should a mechanic use this result",
        "why was this maintenance item flagged",
        "is the score the actual chance of failure",
        "can I trust this prediction without an inspection",
        "explain why this item was recommended for servicing",
        "what does the recommendation threshold mean",
        "how does the threshold affect the result",
        "what does the model confidence represent",
        "does a higher score always mean greater danger",
        "how do I understand the prediction results"
    ],
    "brakes": [
        "when should I inspect my brake pads",
        "what are signs of worn brake pads",
        "my brakes squeal when I stop",
        "how do I know if brake fluid needs attention",
        "what maintenance is needed for the braking system",
        "the brake pedal feels soft what should I do",
        "why does my car vibrate when braking",
        "when should brake components be checked",
        "what should be inspected in the brake system",
        "my brakes feel less responsive",
        "how can I recognize a possible brake problem",
        "what maintenance does the clutch need",
        "when should clutch operation be inspected",
        "what can cause unusual braking sounds",
        "should a mechanic inspect noisy brakes",
        "how often should brakes be checked",
        "what should I do if brake fluid is leaking",
        "can worn brake pads affect vehicle safety",
        "what are warning signs of clutch wear",
        "which brake components need regular inspection"
    ],
    "oil": [
        "how do I know when engine oil needs changing",
        "what happens if a vehicle has low engine oil",
        "when should the oil filter be inspected",
        "what does an engine oil warning light mean",
        "how often should engine oil be checked",
        "what maintenance is needed for engine lubrication",
        "why is checking oil level important",
        "what should I do before adding engine oil",
        "can old oil affect engine performance",
        "how can I check the engine oil level",
        "what happens when oil changes are delayed",
        "when should a mechanic inspect an oil leak",
        "does this vehicle need an oil service",
        "what is the purpose of an oil filter",
        "how should I plan regular oil changes",
        "why should the correct oil specification be used",
        "what can cause the engine oil level to drop",
        "how do I recognize a possible oil leak",
        "what should be included in an oil service",
        "when should engine lubrication be checked"
    ],
    "cooling": [
        "what should I do if the engine temperature rises",
        "how do I check the coolant level safely",
        "what are signs of a cooling system problem",
        "why is the engine temperature too high",
        "when should the radiator be inspected",
        "what does a coolant warning light indicate",
        "how can I recognize a coolant leak",
        "why is coolant important for the engine",
        "what maintenance does the cooling system need",
        "what should I do if the vehicle overheats",
        "how often should coolant be checked",
        "can low coolant cause engine overheating",
        "what should a mechanic inspect in the radiator",
        "why might the engine run hotter than normal",
        "how do I identify a possible cooling problem",
        "what happens if the cooling system is neglected",
        "when should the cooling system be serviced",
        "what are common signs of radiator trouble",
        "how can coolant leaks affect a vehicle",
        "what should I know about engine temperature warnings"
    ],
    "filters_fluids": [
        "when should the engine air filter be replaced",
        "what does a dirty cabin filter affect",
        "how often should the fuel filter be inspected",
        "what is the purpose of transmission fluid",
        "when should spark plugs be checked",
        "what maintenance does the washer drain plug need",
        "how do I know if a filter needs servicing",
        "what fluids should be checked during maintenance",
        "why is transmission fluid condition important",
        "what happens when an air filter becomes clogged",
        "when should the pollen filter be changed",
        "what should be checked during a fluid service",
        "how do spark plugs affect engine operation",
        "what is included in routine filter maintenance",
        "when should the fuel system filter be serviced",
        "why should transmission fluid be maintained",
        "what are signs of a clogged air filter",
        "how do I plan filter replacements",
        "which vehicle fluids need regular inspection",
        "what should a mechanic check during fluid maintenance"
    ],
    "safety": [
        "should I drive if the brakes are not responding normally",
        "what should I do if the steering suddenly feels different",
        "can I keep driving when the engine overheats",
        "does an AI prediction prove a vehicle is safe",
        "what should I do if I suspect a serious vehicle fault",
        "is it safe to ignore a brake warning light",
        "what if the tires look damaged",
        "when should I stop driving and seek an inspection",
        "can this chatbot certify that my car is roadworthy",
        "what should I do if I notice a fluid leak",
        "is a maintenance score a safety inspection",
        "should a mechanic verify these recommendations",
        "what if the engine makes an unusual noise",
        "can I rely on AI instead of a qualified mechanic",
        "what should I do when a critical warning appears",
        "does a successful prediction mean there is no fault",
        "how should I handle a possible steering problem",
        "can the system guarantee my vehicle is safe",
        "when is professional vehicle inspection necessary",
        "what should I do if a vehicle feels unsafe"
    ],
    "capabilities": [
        "what automotive tasks can you assist with",
        "can you recommend vehicle maintenance",
        "can you analyze multiple rental vehicles",
        "can you explain maintenance prediction results",
        "can you help prioritize fleet servicing",
        "what types of vehicle questions can I ask",
        "can you identify possible maintenance priorities",
        "can you explain why a component was flagged",
        "can you help me understand vehicle service needs",
        "what does this automotive assistant specialize in",
        "can you analyze vehicle details and mileage",
        "can you answer questions about filters and fluids",
        "can you help with brake maintenance questions",
        "what can the fleet analysis feature do",
        "can you guarantee that a vehicle will not fail",
        "can you replace a mechanic's inspection",
        "which maintenance topics do you support",
        "can you use vehicle information to make recommendations",
        "how can this AI help a rental fleet manager",
        "what are the limits of your maintenance predictions"
    ],
    "maintenance_schedule": [
        "how do I create a regular vehicle service plan",
        "what maintenance records should I maintain",
        "when should I schedule the next inspection",
        "how can I plan servicing for rental cars",
        "what should a routine vehicle inspection include",
        "how often should a fleet vehicle be serviced",
        "how do I track upcoming maintenance",
        "what information belongs in a service history",
        "how can I avoid missing scheduled maintenance",
        "what should be checked during routine servicing",
        "how should I organize maintenance dates",
        "what is the benefit of keeping service records",
        "how do I plan preventive maintenance for vehicles",
        "when should I review a vehicle's maintenance history",
        "how can mileage help plan a service interval",
        "what should be recorded after vehicle servicing",
        "how can I schedule inspections for several cars",
        "what should I check before a rental car is released",
        "how do I keep track of recurring service tasks",
        "what is a good process for planning fleet maintenance"
    ],
    "goodbye": [
        "goodbye for now",
        "bye assistant",
        "thanks that helped",
        "thank you for the explanation",
        "I have no more questions",
        "that is all for now",
        "see you later",
        "thanks for the maintenance advice",
        "I am finished for now",
        "okay goodbye",
        "thanks for checking the vehicle",
        "thank you for your assistance",
        "I will ask again later",
        "that answers my question",
        "thanks I understand now",
        "end this conversation",
        "I am done asking questions",
        "alright take care",
        "thanks for your time",
        "farewell"
    ]
}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or not isinstance(data.get("intents"), list):
        raise ValueError("Expected a JSON object containing an intents list.")

    intent_map = {}

    for item in data["intents"]:
        tag = item.get("tag")
        patterns = item.get("patterns")

        if not tag or not isinstance(patterns, list):
            raise ValueError("Each intent must contain a tag and patterns list.")

        if tag in intent_map:
            raise ValueError(f"Duplicate intent tag found: {tag}")

        intent_map[tag] = item

    missing_tags = sorted(set(ADDITIONAL_PATTERNS) - set(intent_map))

    if missing_tags:
        raise ValueError(
            "These expected intent tags were not found: "
            + ", ".join(missing_tags)
        )

    added_counts = {}

    for tag, new_patterns in ADDITIONAL_PATTERNS.items():
        existing = intent_map[tag]["patterns"]
        seen = {pattern.strip().casefold() for pattern in existing}
        added = 0

        for pattern in new_patterns:
            normalized = pattern.strip().casefold()

            if normalized and normalized not in seen:
                existing.append(pattern.strip())
                seen.add(normalized)
                added += 1

        added_counts[tag] = added

    DATA_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    total_patterns = sum(
        len(item["patterns"])
        for item in data["intents"]
    )

    print("Chatbot training data expanded successfully.")
    print(f"Intent count: {len(data['intents'])}")
    print(f"Total pattern count: {total_patterns}")
    print("\nAdded patterns per intent:")

    for tag, count in added_counts.items():
        print(f"{tag}: {count}")

    print("\nResponses and existing intent tags were preserved.")


if __name__ == "__main__":
    main()
