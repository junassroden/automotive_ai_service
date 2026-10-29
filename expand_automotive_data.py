
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "chatbot_intents.json"
RANDOM_SEED = 42
TARGET_EXAMPLES = 3000

random.seed(RANDOM_SEED)

SYMPTOMS = {
    "exhaust smoke": [
        "my exhaust is producing smoke",
        "smoke is coming from my tailpipe",
        "my car is blowing smoke",
        "there is smoke coming out of the exhaust",
        "why is my vehicle producing exhaust smoke",
        "my exhaust started smoking recently",
        "the tailpipe keeps releasing smoke",
        "my vehicle has unusual exhaust smoke"
    ],
    "engine overheating": [
        "my engine is overheating",
        "the temperature gauge is too high",
        "my car engine is running hot",
        "the temperature warning appeared",
        "steam is coming from under the hood",
        "my engine temperature keeps rising",
        "my car gets too hot while driving",
        "the engine temperature is above normal"
    ],
    "engine noise": [
        "my engine is making a strange noise",
        "there is a knocking sound from the engine",
        "my car engine sounds unusual",
        "I hear rattling from under the hood",
        "the engine is making a ticking noise",
        "my engine started making loud sounds",
        "there is an unusual sound when the engine runs",
        "why is my engine making noise"
    ],
    "starting problems": [
        "my car will not start",
        "the engine struggles to start",
        "my vehicle has trouble starting",
        "the starter clicks but the engine does not start",
        "my car takes several attempts to start",
        "the engine cranks but will not start",
        "my vehicle suddenly stopped starting",
        "why does my car struggle to start"
    ],
    "vehicle vibration": [
        "my car shakes while driving",
        "the vehicle vibrates at high speed",
        "my steering wheel is shaking",
        "the car shakes when idling",
        "I feel unusual vibration from the engine",
        "my vehicle vibrates when accelerating",
        "the car shakes when I press the brakes",
        "why is my vehicle shaking"
    ],
    "fluid leaks": [
        "my car is leaking fluid",
        "there is liquid underneath my vehicle",
        "I noticed a fluid leak in my car",
        "something is dripping from the engine",
        "there is a puddle under my parked vehicle",
        "my vehicle keeps losing fluid",
        "I found an unknown liquid under my car",
        "why is my car leaking"
    ],
    "transmission problems": [
        "my transmission shifts roughly",
        "the gearbox makes unusual noises",
        "my car struggles to change gears",
        "the transmission seems to be slipping",
        "my vehicle jerks when changing gears",
        "the automatic transmission feels unusual",
        "my car has trouble engaging a gear",
        "why does my vehicle shift roughly"
    ],
    "battery and electrical problems": [
        "my battery keeps going flat",
        "the dashboard lights are flickering",
        "my car battery warning light is on",
        "the headlights are unusually dim",
        "my vehicle has electrical problems",
        "the engine clicks when I try to start it",
        "my car loses electrical power",
        "why does my car battery keep dying"
    ],
    "steering problems": [
        "my steering wheel feels stiff",
        "the car pulls to one side",
        "my steering feels loose",
        "the steering wheel is difficult to turn",
        "my vehicle wanders while driving",
        "I hear noise when turning the steering wheel",
        "my steering suddenly feels different",
        "why is my car pulling to one side"
    ],
    "brake symptoms": [
        "my brakes are grinding",
        "the brake pedal feels soft",
        "my car takes longer to stop",
        "the brakes squeal when I stop",
        "the vehicle pulls when braking",
        "my brake pedal vibrates",
        "my brakes do not feel normal",
        "why are my brakes making noise"
    ],
    "air conditioning problems": [
        "my car air conditioner is not cold",
        "the AC blows warm air",
        "my vehicle air conditioning stopped working",
        "the AC makes a strange noise",
        "my car AC cools inconsistently",
        "the air conditioner has a bad smell",
        "why is my car AC not cooling",
        "the cabin is not getting cold"
    ],
    "fuel and performance problems": [
        "my car is using too much fuel",
        "the engine loses power while driving",
        "my vehicle hesitates when accelerating",
        "the engine sputters when I accelerate",
        "my car has poor fuel economy",
        "the engine feels weak",
        "my vehicle struggles uphill",
        "why is my car losing power"
    ]
}

QUESTIONS = [
    "{symptom}. What could be causing it?",
    "{symptom}. What should I check?",
    "{symptom}. What might be wrong with my vehicle?",
    "{symptom}. How should I troubleshoot this?",
    "{symptom}. What could be the possible cause?",
    "{symptom}. Should I have it inspected?",
    "Can you help me understand this problem: {symptom}?",
    "I noticed this issue: {symptom}. What should I do?",
    "What does it mean when {symptom}?",
    "How serious could this problem be: {symptom}?",
    "My vehicle has this issue: {symptom}. What are the possibilities?",
    "Can you explain possible causes when {symptom}?",
    "I am concerned because {symptom}. What should I know?",
    "What are common reasons for this symptom: {symptom}?",
    "Help me troubleshoot my car because {symptom}.",
    "I need automotive troubleshooting advice. {symptom}.",
    "What should a mechanic investigate if {symptom}?",
    "Could maintenance issues cause this: {symptom}?",
    "How can I respond safely if {symptom}?",
    "Please explain what could cause this: {symptom}."
]

def load_dataset():
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload.get("intents"), list):
        raise ValueError("Dataset must contain an intents list.")
    return payload

def main():
    payload = load_dataset()
    intents = payload["intents"]

    target = next(
        (item for item in intents if item.get("tag") == "troubleshooting"),
        None
    )

    if target is None:
        target = {
            "tag": "troubleshooting",
            "answer": (
                "I can help explain possible causes of vehicle symptoms, "
                "but I cannot confirm a mechanical fault from a message alone. "
                "Tell me the symptom details, when it occurs, and any warning "
                "lights. Stop safely and seek qualified assistance for serious "
                "brake, steering, overheating, smoke, or other safety concerns."
            ),
            "patterns": []
        }
        intents.append(target)

    existing = {
        pattern.strip().casefold()
        for item in intents
        for pattern in item.get("patterns", [])
        if isinstance(pattern, str)
    }

    candidates = []

    for symptom, examples in SYMPTOMS.items():
        for example in examples:
            for template in QUESTIONS:
                phrase = template.format(symptom=example)
                normalized = phrase.casefold().strip()

                if normalized not in existing:
                    candidates.append(phrase)
                    existing.add(normalized)

    random.shuffle(candidates)
    needed = max(0, TARGET_EXAMPLES - len(target["patterns"]))
    added = candidates[:needed]
    target["patterns"].extend(added)

    backup_path = DATA_PATH.with_suffix(".backup.json")
    if not backup_path.exists():
        backup_path.write_text(
            json.dumps(load_dataset(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    DATA_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Automotive troubleshooting data updated.")
    print(f"New examples added: {len(added)}")
    print(f"Troubleshooting examples: {len(target['patterns'])}")
    print(f"Total patterns: {sum(len(i.get('patterns', [])) for i in intents)}")
    print(f"Backup: {backup_path}")

if __name__ == "__main__":
    main()
