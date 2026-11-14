# Answer quality: current

86 of 100 cases passed (114 turns).
Answer length: mean 61.4 words, median 60, max 114.

| Metric | Passed | Rate |
|---|---|---|
| accuracy | 72/85 | 0.847 |
| conciseness | 103/104 | 0.99 |
| consistency | 6/7 | 0.857 |
| context_awareness | 9/9 | 1.0 |
| expected_kind | 112/112 | 1.0 |
| hallucination | 15/15 | 1.0 |
| no_leakage | 114/114 | 1.0 |
| relevance | 95/95 | 1.0 |
| retrieval | 94/94 | 1.0 |
| safety | 20/20 | 1.0 |

## Failed cases

- **overheat_follow_up** "Why is my car overheating?" -> answer/engine_overheating (96 words): failed accuracy
- **misfire** "My engine is misfiring" -> answer/engine_misfire (80 words): failed accuracy
- **battery_dies_overnight** "My battery dies overnight" -> answer/battery_charging_starting (72 words): failed accuracy
- **blown_fuse** "My radio stopped working, could it be a blown fuse?" -> answer/blown_fuse (57 words): failed accuracy
- **trans_slip** "Transmission slipping" -> answer/transmission_slip (57 words): failed accuracy
- **warning_light_general** "What does the check engine light mean?" -> answer/dashboard_warning_obd (30 words): failed accuracy
- **rental_inspection** "What should I check before renting a car?" -> answer/pre_rental_inspection (48 words): failed accuracy
- **rental_damage** "I scratched the rental car, what should I do?" -> answer/rental_damage_reporting (12 words): failed accuracy
- **follow_up_cost** "How much will it cost to fix?" -> answer/transmission_slip (36 words): failed accuracy
- **orange_engine_symbol** "what's that orange engine symbol on my dash" -> answer/dashboard_warning_obd (39 words): failed accuracy
- **jerky_shift** "my car jerks when shifting from 2nd to 3rd" -> answer/transmission_slip (36 words): failed accuracy
- **ran_over_nail** "i ran over a nail" -> answer/tire_puncture (38 words): failed accuracy
- **temp_light_follow_ups** "how much will that cost" -> answer/engine_overheating (32 words): failed accuracy
- **replaced_plugs** "I already replaced the spark plugs" -> answer/engine_misfire (84 words): failed conciseness
