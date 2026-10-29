# Custom Automotive AI Service

A local Python/Django service for:
1. Maintenance predictions using your own trained maintenance model.
2. Fleet analysis and prioritization from vehicle records sent by your application.
3. A custom-trained automotive chatbot intent classifier.

No OpenAI, Claude, Gemini, hosted AI API, or pretrained language model is used.

## Important model details

The legacy artifact is present at:

`trained_models/vehicle_maintenance_model.joblib`

It is not loaded by the current inference path: the original `train_model.py`
is absent, and the artifact's training data and evaluation cannot be reproduced.
Accepted models must be explicitly present in `models/model_registry.json` with
matching artifact/evaluation hashes and verified acceptance evidence.

The legacy prediction and fleet-analysis paths are disabled by default because
the artifact's training data and evaluation have not been verified. This audit
does not validate the artifact as evidence-backed fleet predictive maintenance.
The service does not provide an opt-in bypass. `/api/predict/` and
`/api/fleet/analyze/` return HTTP 503 with `status: "unavailable"` unless a
model is recorded as evaluated and accepted in `models/model_registry.json`,
its artifact and evaluation report match their registered SHA-256 hashes, and
the evaluation meets the configured group-split, baseline, and probability
calibration criteria. Chat
prediction/fleet requests also report unavailability without model scores.

The chatbot model is trained from the included labeled examples. It is a small custom NLP intent classifier, not a generative LLM. It uses TF-IDF features and Logistic Regression trained on the included examples. Add more real automotive question examples to `data/chatbot_intents.json` and retrain it as you expand coverage.

## Windows installation

Run these commands in PowerShell from this project folder:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train the chatbot NLP model:

```powershell
python -m data_pipeline.cli train-chatbot
```

Start the service:

```powershell
python manage.py runserver 127.0.0.1:8001
```

Test health:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/api/health/
```

If `AI_SERVICE_API_KEY` is set in your environment, requests must include the `X-AI-Service-Key` header. For local testing, set it in PowerShell:

```powershell
$env:AI_SERVICE_API_KEY = "change-this-local-api-key"
```

Use the same value in the caller. For production, set a strong secret in your environment, disable debug mode, and do not expose Django's development server publicly.

The LavaLust backend forwards authenticated chat requests to this service at
`http://127.0.0.1:8001/api/chat/` by default. Override the URL with
`AI_SERVICE_API_URL` in the backend environment if needed. If
`AI_SERVICE_API_KEY` is configured, set the same value in both applications;
the backend sends it in the `X-AI-Service-Key` header.

## Endpoints

- `GET /api/health/`
- `GET /api/capabilities/`
- `POST /api/predict/`
- `POST /api/fleet/analyze/`
- `POST /api/chat/`

### Prediction request

Send the features used during training:

```json
{
  "vehicle": {
    "vehicle_type": "car",
    "brand": "Toyota",
    "model": "Vios",
    "engine_type": "gasoline",
    "make_year": 2021,
    "region": "Mimaropa",
    "mileage_range": "50000-60000",
    "mileage": 55000
  }
}
```

### Fleet request

The current endpoint accepts a `vehicles` array. Your LavaLust backend should fetch the real registered vehicle records and map them to the model's feature names before forwarding them. This starter does not guess your database schema.

```json
{
  "vehicles": [
    {
      "id": 12,
      "name": "Toyota Vios",
      "vehicle_type": "car",
      "brand": "Toyota",
      "model": "Vios",
      "engine_type": "gasoline",
      "make_year": 2021,
      "region": "Mimaropa",
      "mileage_range": "50000-60000",
      "mileage": 55000
    }
  ]
}
```

### Chat request

```json
{
  "message": "Which vehicles need maintenance?",
  "vehicles": []
}
```

The chatbot does not directly access MySQL. The authenticated LavaLust backend
owns database access, applies the signed-in user's permissions, and sends a
bounded `rental_context` only for questions that need live application data.
General automotive questions continue through the local knowledge retrieval
path.

The backend supplies public vehicle listings and stored daily rates for rental
questions. Without dates, availability answers describe listing status only.
For a date-specific availability check, include ISO pickup and return dates
(`YYYY-MM-DD`); the backend excludes pending or confirmed reservations that
overlap the half-open period `[pickup, return)` and applies each vehicle's
minimum rental duration. If the rental period is incomplete or invalid, the
chat asks for dates rather than treating a listing status as confirmation.
Multi-day price totals are estimates using the stored daily rate only; they
exclude any unrepresented fees and are not booking quotes.

Customers can ask about their own reservation records; the backend scopes
those queries to the authenticated account and does not send customer contact
details. Fleet status and maintenance records are only sent for administrator
questions. Legacy maintenance entries are described as service logs, not
verified fault labels. The newer maintenance evidence is included only when
its table is installed and only for records marked verified. A rule-based
service estimate or an anomaly score is not presented as a confirmed fault or
future component-failure prediction. Chat cannot create, cancel, or confirm
bookings.

### Automotive knowledge retrieval

The intent classifier is not the chatbot's only source of automotive
information. `data/automotive_knowledge.json` contains structured local
reference entries for engine and cooling symptoms, brakes, transmission and
drivetrain, tires, steering and suspension, electrical systems, fluids,
diagnostics, routine service, predictive-model interpretation, and fleet
operations. Each entry has keywords, an overview, possible causes, symptoms,
checks, urgency, information needed for a better assessment, and source
references.

`ai_api/automotive_knowledge.py` uses the project's existing scikit-learn
installation to retrieve a matching entry locally. The answer is assembled
from the retrieved text and may show reference pointers recorded in the
knowledge base; it does not call an external AI provider or generate facts.
Low-match, unrelated questions are not given an
automotive knowledge answer. The retrieval set is a starting point, not
comprehensive repair documentation.

Knowledge retrieval is passage-based: the search index includes document titles,
keywords, and structured sections such as causes, symptoms, checks, and
diagnostic steps. It returns up to six relevant passages and assembles an
answer from the matching topic record. `data.knowledge_id` remains the primary
topic for compatibility; `knowledge_ids`, `retrieved_passage_count`, and the
score-interpretation fields provide additive detail. The numeric
`retrieval_score` is a heuristic lexical/vector relevance score, not a
probability. The existing `confidence` field is the classifier's top
`predict_proba` output; it has not been calibrated or validated as a probability
that the intent or answer is correct. Neither number should be used as a
diagnostic certainty.

Knowledge-base source references are pointers recorded with each topic, not
documents fetched live for an answer. Responses say when references were not
retrieved and call out missing vehicle-specific manuals or service procedures.
The repository currently has no verified manufacturer service-document corpus;
exact fluid specifications, capacities, torque values, intervals, wiring
pinouts, code-specific tests, and repair procedures must come from the manual
or service information for the exact vehicle. Hybrid/EV high-voltage work is
restricted to appropriately trained technicians.

Run the fixed retrieval regression set (126 questions spanning current
knowledge topics, safety cases, unsupported vehicle-specific questions, and
out-of-domain prompts) with:

```powershell
python -m data_pipeline.evaluate_knowledge
```

The questions are in
`data/reports/chatbot_knowledge_eval_questions.json`; measured retrieval,
abstention, safety-phrase, and uncertainty results are written to
`data/reports/chatbot_knowledge_evaluation.json`. This automated suite checks
retrieval behavior and selected answer safeguards. It does **not** establish
mechanical factual correctness, citation support, or technician-level quality;
those rubric items remain subject to documented human review.

To add a knowledge entry, add a unique document to the JSON `documents` list
and provide every existing structured field. Use the exact vehicle maker's
owner's manual or service schedule for model-specific requirements. Do not add
unverified service intervals, fluid specifications, diagnostic codes, failure
probabilities, or repair estimates. Keep advice conservative, identify
safety-critical symptoms, and cite a trustworthy source for technical
recommendations. Run `python manage.py test` after edits.

The maintenance predictor remains separate from retrieval. Its installed
artifact is expected to use `vehicle_type`, `brand`, `model`, `engine_type`,
`make_year`, `region`, `mileage_range`, and `mileage`. Earlier project notes
describe 12 target classifiers and two constant targets, but the training source
and evaluation are not reproducible from this repository. Treat those details
as unverified. Missing feature data is reported instead of synthesized.
Predictions require a registered `evaluated_accepted` model and a matching
evaluation report; placing an artifact in `trained_models/` alone does not
enable inference.
That feature validation applies only when unverified legacy predictions are
explicitly enabled; normal operation reports prediction unavailability.
The existing PHP diagnostic handler references `$this->maintenanceModel`, but
`ApiController::before_action()` does not initialize it and no
`MaintenanceModel` implementation is present in `Back-End/app/models`. The
diagnostic database context therefore needs a working backend model before it
can be considered end-to-end. Even once that is supplied, the current PHP
diagnostic context does not include all eight model features, and the backend
does not supply model-ready fleet records to the chat endpoint. Database-backed
prediction and fleet analysis require that model, a verified feature mapping,
and complete authorized records. Retrieval can provide general guidance now,
but it does not claim to have analyzed the database vehicle or fleet.

## Model limitations

The legacy classifier's score is not a calibrated guarantee that a component is
failing. The dataset labels, training process and evaluation for that artifact
have not been independently verified here. Verify service history and inspect
vehicles before making safety decisions. A target with only one class in training data cannot be learned as a binary
classifier; the service will not treat a constant target as a trained model.

## Dataset audit and evidence-based training

Run these commands from `automotive_ai_service` after activating the service virtual environment:

```powershell
python -m pip install -r requirements.txt
python -m data_pipeline.cli inventory
python -m data_pipeline.cli prepare-enginefaultdb
python -m data_pipeline.cli train-maintenance
python -m data_pipeline.cli train-chatbot
python -m data_pipeline.cli status
python manage.py test
```

The inventory reads files without changing `data/raw/`; it records schemas, missing cells, exact duplicate rows, file hashes, archive safety findings and any inspection errors in `data/reports/`. Scripts and notebooks in the downloaded collection are not executed. Safe archive extraction is available through `python -m data_pipeline.cli safe-extract` and is limited to archives in `data/raw/` and destinations in `data/processed/archives/`.

The EngineFaultDB preparation step writes a normalized copy and a transformation record to `data/processed/enginefaultdb/`. Its numeric four-class labels have no verified fault-name mapping or run/vehicle IDs. `train-maintenance` therefore exits with a blocked status and a clear reason; it does not manufacture group IDs or metrics. The chatbot is a separate TF-IDF/logistic-regression intent classifier trained only on `data/chatbot_intents.json`; its authored-example holdout report is written to `data/reports/chatbot_training.json` (holdout scores are not an estimate of independent user-question performance). Training states for all tasks are written to `data/training/training_status.json`; the maintenance eligibility registry is `models/model_registry.json`. Candidate grouped classifiers are research artifacts and are never automatically promoted to the production maintenance slot.

`GET /api/capabilities/` reports currently verified model capabilities. The existing legacy artifact is not loaded because it has no accepted registry entry and evaluation evidence.
