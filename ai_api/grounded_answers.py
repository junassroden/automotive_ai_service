"""Answers built only from the records the Back-End retrieved for a plan.

Every figure in an answer comes from the retrieval results (prices, deposits and totals
are computed by the Back-End with the reservation pricing rule). validate_amounts()
rejects an answer that mentions a peso amount not present in the results, and the
caller then falls back to a plain listing of the records. When a record is missing or
a tool failed, the answer says exactly what could not be verified.
"""
import datetime
import re


ADMIN_ONLY = ("That information is limited to administrators, so I can't show it for this account. "
              "I can help with vehicle listings, prices, availability and your own reservations.")
DATA_ERROR = ("I couldn't read {what} from the database just now, so I can't give a verified answer. "
              "Please try again in a moment.")
AMOUNT = re.compile(r"PHP\s?([\d,]+(?:\.\d{1,2})?)")
# Reservation statuses as the system guide defines them.
STATUS_MEANINGS = {
    "pending": "it is awaiting payment, and it is confirmed automatically once the payment is verified",
    "confirmed": "the vehicle is reserved for your dates",
    "cancelled": "it was cancelled and no longer holds the vehicle",
    "rejected": "it was not accepted, so it does not hold the vehicle",
    "completed": "the rental has finished and the vehicle was returned",
}
# Payment states as My Reservations shows them (ReservationModel::attach_payment_summary).
PAYMENT_MEANINGS = {
    "paid": "the payment is verified and the reservation is confirmed.",
    "pending": "a payment was started but hasn't been verified yet. If you've finished paying, it usually updates shortly; "
               "otherwise continue the payment.",
    "failed": "the last payment attempt didn't go through or expired, so it isn't paid yet. You can retry the payment.",
    "unpaid": "it hasn't been paid yet, so it stays pending until you pay.",
    "not_required": "no payment is due for it.",
}
STATUS_LABELS = {"overdue": "Overdue", "approaching": "Due within 30 days", "not_due": "Not due",
                 "insufficient_history": "Not enough service history"}
# Explanations shown once per conversation (the conversation memory records which were shown).
NOTES = {
    "price_rule": "Totals are the daily rate times the number of days; tell me how many days for an estimate.",
    "quote_rule": ("This is the daily rate times the number of days. The system stores no discounts or extra fees; "
                   "the rental team confirms the final amount."),
    "deposit_rule": ("The deposit is shown separately and is not added to the rental total. The system does not record "
                     "whether or when a deposit is refunded; the rental team handles that, so please confirm with them."),
    "dates_hint": "For specific dates, give me the pickup and return dates and I'll check them against existing bookings.",
    "schedule_estimate": "These are schedule estimates, not confirmed faults; check the manufacturer schedule.",
    "booked_value": "These are booked reservation totals, not money received; only verified online payments count as paid.",
    "no_payments": "This is the booked amount. Whether each reservation is paid shows on My Reservations and Payments (/users/payments); ask \"did my payment go through?\" to check here.",
    "service_log": "Service-log entries record work done; they are not verified fault diagnoses.",
}


class Notes:
    def __init__(self, shown):
        self.shown, self.added = set(shown or []), []

    def add(self, lines, key):
        if key not in self.shown and key not in self.added:
            self.added.append(key)
            lines.append(NOTES[key])


class UngroundedAnswer(Exception):
    pass


def money(value):
    return f"PHP {float(value):,.2f}"


def name(row):
    return " ".join(str(row.get(key)) for key in ("make", "model") if row.get(key)).strip() or f"Vehicle #{row.get('id')}"


def labels(vehicles):
    """Names that tell vehicles with the same make and model apart (year, colour, plate or id)."""
    counts = {}
    for v in vehicles:
        counts[name(v)] = counts.get(name(v), 0) + 1
    out = {}
    for v in vehicles:
        label = name(v)
        if counts[label] > 1:
            extra = ", ".join(str(part) for part in (v.get("year"), v.get("color"), v.get("plate_number") or f"#{v.get('id')}") if part)
            label = f"{label} ({extra})"
        out[id(v)] = label
    return out


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _upper_first(text):
    """Capitalises only the first letter ("GPS-equipped" stays "GPS-equipped"; str.capitalize would give "Gps-equipped")."""
    return text[:1].upper() + text[1:] if text else text


def compose(plan, results, role, vocabulary=None, guide=None, fleet_risk=None, shown_notes=()):
    """(answer, data). guide() -> (text, sources) | None; fleet_risk(vehicles) -> answer text."""
    results = results if isinstance(results, list) else []
    by_tool = {item.get("tool"): item for item in results if isinstance(item, dict)}
    intent = plan.get("intent")
    data = {"plan_intent": intent, "retrieval_status": {tool: item.get("status") for tool, item in by_tool.items()}}

    failure = _failure(by_tool, intent)
    if failure:
        return failure, data

    handler = {
        "vehicles": _vehicles, "own_bookings": _own_bookings, "reservation_summary": _reservation_summary,
        "fleet_status": _fleet_status, "maintenance_due": _maintenance_due, "maintenance_records": _maintenance_records,
        "maintenance_recurring": _maintenance_recurring, "maintenance_inspections": _maintenance_inspections,
        "maintenance_prediction": _maintenance_prediction, "maintenance_costs": _maintenance_costs,
        "vehicle_status": _vehicle_status, "revenue": _revenue, "utilization": _utilization,
        "reassessment_due": _reassessment_due,
        "maintenance_condition": _maintenance_condition, "maintenance_priority": _maintenance_priority,
        "maintenance_retire": _maintenance_retire,
    }.get(intent)
    if handler is None:
        return "I couldn't work out which records answer that. Could you rephrase it?", data
    notes = Notes(shown_notes)
    answer = handler(plan, by_tool, role, vocabulary or {}, guide, fleet_risk, notes)
    if intent in ("maintenance_priority", "maintenance_retire") and _condition_missing(by_tool) \
            and not answer.startswith(CONDITION_UNAVAILABLE):
        answer = CONDITION_UNAVAILABLE + ". This is based on the service schedule and maintenance records only.\n\n" + answer
    data["notes_shown"] = notes.added
    try:
        validate_amounts(answer, results)
        validate_answers_question(plan, answer, by_tool)
    except UngroundedAnswer:
        answer = _fallback(by_tool)
        data["validation"] = "fallback_listing"
    return answer, data


def validate_amounts(answer, results):
    """Every peso amount in the answer must appear in the retrieved records."""
    known = set()
    _numbers(results, known)
    for match in AMOUNT.finditer(answer):
        value = round(float(match.group(1).replace(",", "")), 2)
        if value not in known:
            raise UngroundedAnswer(f"PHP {match.group(1)} is not in the retrieved records")


def validate_answers_question(plan, answer, by_tool):
    """The last check before replying: a vehicle question found in the records must name that vehicle in the answer."""
    if plan.get("intent") != "vehicles" or plan.get("purpose") == "attribute":
        return  # An attribute answer may rightly be "only administrators can see that".
    item = by_tool.get("vehicles.quote") or by_tool.get("vehicles.search") or {}
    found = [str(v.get("model") or "").casefold() for v in (item.get("data") or {}).get("vehicles") or []]
    low = answer.casefold()
    for model in (plan.get("requested") or {}).get("models") or []:
        if str(model).casefold() in found and str(model).casefold() not in low:
            raise UngroundedAnswer(f"the answer does not mention {model}")


def _numbers(value, known):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        known.add(round(float(value), 2))
    elif isinstance(value, str):
        try:
            known.add(round(float(value), 2))
        except ValueError:
            pass
    elif isinstance(value, dict):
        for item in value.values():
            _numbers(item, known)
        # A quote's rental charge plus its deposit is the one derived amount answers may state.
        if all(isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool)
               for key in ("base_total", "security_deposit")):
            known.add(round(float(value["base_total"]) + float(value["security_deposit"]), 2))
    elif isinstance(value, list):
        for item in value:
            _numbers(item, known)


def _failure(by_tool, intent):
    if not by_tool:
        return DATA_ERROR.format(what="the records for that question")
    statuses = {item.get("status") for item in by_tool.values()}
    if statuses & {"ok"}:
        return None
    first = next(iter(by_tool.values()))
    if "denied" in statuses:
        return ADMIN_ONLY
    if first.get("reason") == "invalid_dates":
        return ("Those dates don't work: the pickup date must be today or later and the return date after it, "
                "written as YYYY-MM-DD (for example 2026-12-01 to 2026-12-04).")
    if first.get("reason") == "rental_days_required":
        return "For how many days? Tell me the number of days or the pickup and return dates, and I'll calculate the total."
    what = "the vehicle listings" if intent == "vehicles" else "your reservations" if intent == "own_bookings" else "those records"
    return DATA_ERROR.format(what=what)


# --------------------------------------------------------------------------- vehicles

def _vehicles(plan, by_tool, role, vocabulary, guide, fleet_risk, notes):
    result = by_tool.get("vehicles.quote") or by_tool.get("vehicles.search")
    data = result.get("data") or {}
    vehicles = data.get("vehicles") or []
    requested = plan.get("requested") or {}
    purpose = plan.get("purpose")
    args = (plan.get("retrievals") or [{}])[0].get("args") or {}
    scope = "in the fleet" if role == "admin" else "available for rent right now"
    label = labels(vehicles)
    lines = []

    if plan.get("asks_why_status"):
        lines.append("The system doesn't record why a vehicle has a status: an administrator sets it when editing the vehicle "
                     "(/admin/vehicles), so check with whoever changed it. These are the vehicles with that status now:")
    if plan.get("stock_check") and vehicles and (requested.get("models") or requested.get("makes") or requested.get("categories")) \
            and not requested.get("unknown_names"):
        # "Do you have the Vios?" is answered yes first; its rate follows.
        lines.append(f"Yes, we have the {_join(sorted({name(v) for v in vehicles}))}.")
    for unknown in requested.get("unknown_names") or []:
        lines.append(f"I couldn't find a {unknown} {scope}.")

    if not vehicles and purpose == "recommend":
        need = f" for {args['min_seats']} or more people" if args.get("min_seats") else ""
        return (f"No vehicle{need} can be rented right now, so I can't recommend one. "
                "Ask me about other dates, or about a smaller group, and I'll check again.")
    if not vehicles:
        asked = requested.get("models") or requested.get("makes") or requested.get("categories") or requested.get("unknown_names")
        wanted = _join(asked) if asked else "a vehicle matching that"
        if args.get("min_seats") and not asked:
            lines.append(f"No vehicle that seats {args['min_seats']} or more people is {scope}.")
        elif args.get("max_daily_rate") is not None and not asked:
            lines.append(f"No vehicle with a daily rate at or below {money(args['max_daily_rate'])} is {scope}.")
        elif _feature_words(args) and not asked:
            lines.append(f"No {_feature_words(args)} vehicle is {scope}.")
        elif not requested.get("unknown_names"):
            lines.append(f"I couldn't find {wanted} {scope}.")
        listed = _vocabulary_names(vocabulary)
        lines.append((("Vehicles in the fleet: " if role == "admin" else "Vehicles currently listed: ") + _join(listed) + ".")
                     if listed else "No vehicles are listed at the moment.")
        return " ".join(line for line in lines if line)

    if requested.get("unknown_names") and not (requested.get("makes") or requested.get("models") or requested.get("categories")):
        lines.append(("Vehicles you can rent instead: " if role != "admin" else "Vehicles in the fleet instead: ")
                     + _join(label[id(v)] for v in vehicles) + ". Ask me about any of them for prices and details.")
        return "\n".join(lines)

    if purpose == "recommend":
        lines += _recommend_lines(vehicles, args, label)
        return "\n".join(lines)
    if purpose == "attribute":
        lines += _attribute_lines(plan.get("attribute"), vehicles, label, role)
    elif purpose == "ranking" and plan.get("rank_by") == "seats":
        by_seats = sorted(vehicles, key=lambda v: v.get("seating_capacity") or 0, reverse=True)
        top = by_seats[0]
        if plan.get("long_trip"):
            lines.append("No comfort rating is recorded, but roomier vehicles are usually easier on a long drive, so here they are by size.")
        tied = [label[id(v)] for v in by_seats if v.get("seating_capacity") == top.get("seating_capacity")]
        lines.append(f"The {_join(tied)} {'seat' if len(tied) > 1 else 'seats'} the most people: {top.get('seating_capacity')} seats.")
        if len(by_seats) > 1:
            lines += [f"- {label[id(v)]} ({v.get('category') or 'vehicle'}): {v.get('seating_capacity')} seats, "
                      f"{money(v['daily_rental_price'])} per day" for v in by_seats]
    elif purpose == "feature_check":
        lines += _feature_lines(plan.get("feature_check") or {}, vehicles, label,
                                listing=not (requested.get("models") or requested.get("makes")))
    elif purpose == "compare":
        lines += _compare_lines(plan.get("compare") or {}, vehicles, label)
    elif purpose == "deposit":
        if len(vehicles) == 1:
            v = vehicles[0]
            lines.append(f"The {label[id(v)]} has a {money(v['security_deposit'])} deposit (daily rate {money(v['daily_rental_price'])})."
                         if v.get("security_deposit") else f"No security deposit is listed for the {label[id(v)]}.")
        else:
            amount = lambda v: float(v.get("security_deposit") or 0)  # noqa: E731
            listed = vehicles
            if plan.get("deposit_rank"):
                with_deposit = sorted((v for v in vehicles if amount(v) > 0), key=amount, reverse=plan["deposit_rank"] == "desc")
                if with_deposit:
                    top = with_deposit[0]
                    word = "highest" if plan["deposit_rank"] == "desc" else "lowest"
                    lines.append(f"The {word} deposit is the {label[id(top)]}'s: {money(top['security_deposit'])}.")
                listed = with_deposit + [v for v in vehicles if amount(v) <= 0]
            lines.append("Security deposits from the vehicle records:")
            shown = [v for v in listed if amount(v) > 0]
            none_listed = [v for v in listed if amount(v) <= 0]
            lines += [f"- {label[id(v)]}: {money(v['security_deposit'])} deposit (daily rate {money(v['daily_rental_price'])})" for v in shown]
            if none_listed:
                # Twenty "no deposit is listed" lines say less than one line with the count (of every match, not only those returned).
                others = max(int(data.get("total_matching") or 0), len(vehicles)) - len(shown)
                lines.append(f"- {_join(label[id(v)] for v in none_listed)}: no security deposit is listed" if others <= 3
                             else f"- {others} other vehicles have no security deposit listed")
        notes.add(lines, "deposit_rule")
    elif purpose in {"quote", "availability"} and vehicles[0].get("quote"):
        # A car whose status is not "available" can't be rented on any date; that is not a booking.
        off_fleet = [v for v in vehicles if v.get("status") not in (None, "", "available")]
        rentable = [v for v in vehicles if v not in off_fleet]
        if purpose == "quote" and not data.get("period"):
            # "How much is the Hiace for 1 day?" is a price question: quote every car asked about, and say which can't be booked.
            rentable, off_fleet_note = vehicles, off_fleet
        else:
            off_fleet_note = None
        if not rentable:
            # "How about the Xpander?" when every Xpander is marked unavailable: no estimate for cars nobody can book.
            return "\n".join(lines + _off_fleet_lines(off_fleet, label, every=True))
        if purpose == "availability" and data.get("period"):
            period = data["period"]
            free = [label[id(v)] for v in rentable if v.get("available_for_period")]
            taken = [label[id(v)] for v in rentable if v.get("available_for_period") is False and v not in off_fleet]
            dates = f"{period['start_date']} to {period['end_date']}"
            too_short = [v for v in rentable if v.get("available_for_period") and not (v.get("quote") or {}).get("meets_minimum", True)]
            if free and len(too_short) == len(free):
                # "Is the Vios available tomorrow?": free that day, but one day is shorter than any rental allowed,
                # so the shortest rental that can be booked from that day is quoted instead.
                minimum = max(v["quote"]["minimum_days"] for v in too_short)
                lines.append(f"{_join(free)} {'is' if len(free) == 1 else 'are'} free on {period['start_date']}, but rentals are "
                             f"at least {minimum} days.")
                lines += _minimum_rental_lines(too_short, period["start_date"], minimum, label)
                lines.append("Tell me your return date and I'll check that the whole period is free.")
                if taken:
                    lines.append(f"{_join(taken)} {'is' if len(taken) == 1 else 'are'} already booked on {period['start_date']}.")
                lines += _off_fleet_lines(off_fleet, label)
                notes.add(lines, "quote_rule")
                return "\n".join(lines)
            if free:
                lines.append(f"Yes: {_join(free)} {'is' if len(free) == 1 else 'are'} free from {dates}.")
            elif not taken:
                lines.append(f"None of those vehicles can be rented for {dates}.")
            if taken:
                lines.append(f"{_join(taken)} {'is' if len(taken) == 1 else 'are'} already booked for {dates}.")
        lines += _quote_lines(rentable, data, label)
        lines += _off_fleet_lines(off_fleet, label, every=len(off_fleet) == len(vehicles)) if off_fleet_note is not None \
            else _off_fleet_lines(off_fleet, label)
        if plan.get("include_deposit"):
            for v in vehicles:
                quote = v["quote"]
                if quote.get("security_deposit"):
                    lines.append(f"{'The ' + label[id(v)] + ': r' if len(vehicles) > 1 else 'R'}ental {money(quote['base_total'])} + "
                                 f"security deposit {money(quote['security_deposit'])} = "
                                 f"{money(quote['base_total'] + quote['security_deposit'])} including the deposit.")
        notes.add(lines, "quote_rule")
    elif purpose == "availability" or purpose == "status":
        lines += _availability_lines(vehicles, data, role, purpose, label)
        if plan.get("pending_pickup") and not data.get("period"):
            # "Is the Staria available on 2026-12-05?": the pickup is known; the period needs a return date.
            lines.append(f"To check {plan['pending_pickup']} against existing bookings I need the return date too: rentals are 3 to 7 days "
                         "(up to 21 as a special-duration request). When will you return it?")
        elif role != "admin" and not data.get("period"):
            notes.add(lines, "dates_hint")
    elif purpose == "details":
        if len(vehicles) == 1:
            lines.append(f"The {label[id(vehicles[0])]}: {_specs(vehicles[0])}.")
        else:
            heading = "Vehicles that seat at least {} people".format(args["min_seats"]) if args.get("min_seats") else None
            if heading or _feature_words(args):
                lines.append(f"{heading or _upper_first(_feature_words(args)) + ' vehicles'} {scope}:")
            lines += [f"- {label[id(v)]}: {_specs(v)}" for v in vehicles]
    else:
        if plan.get("hourly"):
            lines.append("Rentals are charged per day; the system has no hourly rates, so a few hours count as a full day.")
        if plan.get("rate_period"):
            lines.append("Only daily rates are recorded; the system has no weekly or monthly rates or discounts. "
                         "A longer rental is the daily rate times the number of days.")
        count = max(int(data.get("total_matching") or 0), len(vehicles)) if plan.get("count_question") else None
        lines += _price_lines(vehicles, {**args, "_count": count, "_total": data.get("total_matching")}, label, role, scope)
        if purpose == "price" and len(vehicles) > 0:
            notes.add(lines, "price_rule")
    if data.get("truncated") and not any(line.startswith("...and ") for line in lines) \
            and not (role == "admin" and purpose in ("availability", "status") and not data.get("period")) \
            and not (purpose == "deposit" and plan.get("deposit_rank")):
        # The admin availability summary already says how many others there are.
        lines.append(f"Showing {len(vehicles)} of {data.get('total_matching')} matching vehicles; name a make, model or status to narrow it down.")
    return "\n".join(lines)


def _deposit(amount):
    """ "PHP 3,000.00 deposit", or that none is listed (a 0.00 deposit is a missing value, not a free deposit)."""
    return f"{money(amount)} deposit" if amount and float(amount) > 0 else "no deposit listed"


def _minimum_rental_lines(vehicles, start_date, minimum, label):
    """The shortest bookable rental from that day, with each vehicle's daily rate. Totals are only stated when the
    Back-End has quoted them (every peso amount must come from the records), so the rate is given here instead."""
    end = (datetime.date.fromisoformat(start_date) + datetime.timedelta(days=minimum)).isoformat()
    lines = [f"The shortest booking from that day is {start_date} to {end} ({minimum} days), at:"]
    for v in vehicles:
        quote = v.get("quote") or {}
        rate = quote.get("daily_rate") or v.get("daily_rental_price")
        deposit = quote.get("security_deposit", v.get("security_deposit"))
        lines.append(f"- {label[id(v)]}: {money(rate)} per day"
                     + (f", plus a {money(deposit)} security deposit" if deposit and float(deposit) > 0 else ""))
    return lines


def _off_fleet_lines(vehicles, label, every=False):
    """Cars an administrator has marked unavailable (or in maintenance): not bookable on any date, and not booked.
    every=True when they are all the vehicles asked about (no "other")."""
    if not vehicles:
        return []
    groups = {}
    for v in vehicles:
        groups.setdefault(v.get("status") or "unavailable", []).append(label[id(v)])
    lines = []
    for status, names in sorted(groups.items()):
        status_text = "in maintenance" if status == "maintenance" else f"marked {status}"
        who = _join(names) if len(names) <= 4 else f"{len(names)} {'' if every else 'other '}vehicles"
        lines.append(f"{who} {'is' if len(names) == 1 else 'are'} {status_text}, so {'it' if len(names) == 1 else 'they'} "
                     f"can't be booked on any date until {'its' if len(names) == 1 else 'their'} status is changed.")
    return lines


def _price_lines(vehicles, args, label, role, scope):
    sort = args.get("sort")
    lines = []
    first = vehicles[0]
    if sort == "price_asc":
        lines.append(f"The cheapest is the {label[id(first)]} at {money(first['daily_rental_price'])} per day.")
    elif sort == "price_desc":
        lines.append(f"The most expensive is the {label[id(first)]} at {money(first['daily_rental_price'])} per day.")
    elif sort in ("mileage_desc", "mileage_asc"):
        word = "highest" if sort == "mileage_desc" else "lowest"
        km = f"{first['current_mileage']:,} km" if first.get("current_mileage") is not None else "no mileage recorded"
        lines.append(f"The {word}-mileage vehicle is the {label[id(first)]}" + (f" ({first['plate_number']})" if first.get("plate_number") else "") + f" at {km}.")
        lines += [f"- {label[id(v)]}: " + (f"{v['current_mileage']:,} km" if v.get("current_mileage") is not None else "no mileage recorded")
                  for v in vehicles]
        return lines
    elif sort in ("year_desc", "year_asc"):
        lines.append(f"The {'newest' if sort == 'year_desc' else 'oldest'} is the {label[id(first)]} ({first.get('year')}).")
    elif args.get("max_daily_rate") is not None:
        lines.append(f"Within {money(args['max_daily_rate'])} per day, you can rent: "
                     + _join(f"{label[id(v)]} ({money(v['daily_rental_price'])}/day)" for v in vehicles) + ".")
        return lines
    elif len(vehicles) == 1:
        v = vehicles[0]
        features = _feature_words(args)
        if features and not (args.get("models") or args.get("makes")):
            has_deposit = v.get("security_deposit") and float(v["security_deposit"]) > 0
            lines.append(f"Yes: the {label[id(v)]} is the {features} vehicle {scope}, at {money(v['daily_rental_price'])} per day"
                         + (f" with a {money(v['security_deposit'])} deposit." if has_deposit else "; no security deposit is listed for it."))
        else:
            lines.append(f"The {label[id(v)]} is {money(v['daily_rental_price'])} per day, with a {money(v['security_deposit'])} deposit."
                         if v.get("security_deposit") and float(v["security_deposit"]) > 0
                         else f"The {label[id(v)]} is {money(v['daily_rental_price'])} per day; no security deposit is listed for it.")
        return lines
    if role == "admin" and sort in ("price_asc", "price_desc") and first.get("status") not in (None, "", "available"):
        # The cheapest car in the whole fleet may be one nobody can rent right now; say so, and name the rentable one.
        rentable = next((v for v in vehicles if v.get("status") == "available"), None)
        lines[-1] = lines[-1][:-1] + f", but it is marked {first['status']}."
        if rentable is not None:
            lines.append(f"The {'cheapest' if sort == 'price_asc' else 'most expensive'} rentable one is the {label[id(rentable)]} "
                         f"at {money(rentable['daily_rental_price'])} per day.")
    features = _feature_words(args)
    if args.get("_count"):
        total = args["_count"]  # "How many cars do you have?"
        lines.append(f"There {'is' if total == 1 else 'are'} {total} vehicle{'s' if total != 1 else ''} {scope}:")
    else:
        lines.append(f"{_upper_first(features)} vehicles {scope}:" if features else ("Vehicles in the fleet:" if role == "admin" else "Daily rates:"))
    shown = vehicles[:LIST_LIMIT_AFTER_ANSWER] if sort in ("price_asc", "price_desc") and len(vehicles) > LIST_LIMIT_AFTER_ANSWER + 1 else vehicles
    lines += [f"- {label[id(v)]} ({v.get('category') or 'vehicle'}, {v.get('seating_capacity')} seats): "
              f"{money(v['daily_rental_price'])} per day, {_deposit(v['security_deposit'])}"
              + (f", status {v['status']}" if role == "admin" and v.get("status") else "") for v in shown]
    if len(shown) < len(vehicles):
        more = max(int(args.get("_total") or 0), len(vehicles)) - len(shown)
        lines.append(f"...and {more} more. Ask \"show all cars\" for the full list.")
    return lines


# A "cheapest"/"most expensive" answer is the first line; a few runners-up are enough after it.
LIST_LIMIT_AFTER_ANSWER = 5


def _feature_words(args):
    parts = []
    parts += [value.lower() for value in args.get("fuel_types") or []]
    parts += [value.lower() for value in args.get("transmissions") or []]
    parts += [value.lower() for value in args.get("colors") or []]
    if args.get("has_gps") is True:
        parts.append("GPS-equipped")
    if args.get("has_gps") is False:
        parts.append("non-GPS")
    if args.get("has_aircon"):
        parts.append("air-conditioned")
    if args.get("min_year"):
        parts.append(f"{args['min_year']}-or-newer")
    return " ".join(parts)


ATTRIBUTE_NAMES = {"plate_number": "plate number", "minimum_rental_duration": "minimum rental period", "year": "model year",
                   "color": "colour", "seating_capacity": "seating capacity", "fuel_type": "fuel type", "transmission": "transmission",
                   "current_mileage": "mileage figure"}


def _attribute_value(attribute, v):
    value = v.get(attribute)
    if value in (None, ""):
        return None
    if attribute == "minimum_rental_duration":
        return f"{value} day" + ("" if int(value) == 1 else "s")
    if attribute == "seating_capacity":
        return f"{value} seats"
    if attribute == "current_mileage":
        return f"{int(value):,} km"
    return str(value)


def _attribute_lines(attribute, vehicles, label, role):
    """Only the field that was asked for, or why it is not shown."""
    what = ATTRIBUTE_NAMES.get(attribute, "that detail")
    if attribute in ("plate_number", "current_mileage") and role != "admin":
        return [f"{what.capitalize()}s are shown only to administrators, so I can't share them on this account. "
                "I can tell you a vehicle's model, year, colour, seats, features and daily rate."]
    lines = []
    for v in vehicles:
        value = _attribute_value(attribute, v)
        if value is None:
            lines.append(f"The {label[id(v)]}'s {what} is not recorded.")
        elif attribute == "minimum_rental_duration":
            lines.append(f"The {label[id(v)]} has a minimum rental period of {value}.")
        elif attribute == "seating_capacity":
            lines.append(f"The {label[id(v)]} has {value}.")
        elif attribute == "fuel_type":
            lines.append(f"The {label[id(v)]} uses {value}.")
        elif attribute == "year":
            lines.append(f"The {label[id(v)]} is a {value} model.")
        else:
            lines.append(f"The {label[id(v)]}'s {what} is {value}.")
    if len(lines) > 1:
        lines = [f"{what.capitalize()} by vehicle:"] + [f"- {line}" for line in lines]
    return lines


COUNTABLE_EQUIPMENT = {"dash camera", "backup camera", "child seat", "sunroof", "touchscreen", "roof rack"}


def _has_equipment(v, terms):
    listed = [re.sub(r"[^a-z0-9]", "", str(item).casefold()) for item in (v.get("additional_features") or [])]
    wanted = [re.sub(r"[^a-z0-9]", "", term.casefold()) for term in terms]
    return any(term and term in item for item in listed for term in wanted)


def _feature_lines(check, vehicles, label, listing=False):
    feature, value = check.get("feature"), check.get("value")
    lines = []
    if feature == "seats":
        # "Is the Vios a 7 seater?" / "Can 7 people fit in the Vios?"
        for v in vehicles:
            seats = v.get("seating_capacity")
            if seats is None:
                lines.append(f"The {label[id(v)]}'s seating capacity is not recorded.")
            elif int(seats) >= int(value):
                lines.append(f"Yes, the {label[id(v)]} has {seats} seats, so it fits {value}.")
            else:
                lines.append(f"No, the {label[id(v)]} has {seats} seats, so it does not fit {value}.")
        return lines
    if feature == "equipment":
        terms = list(check.get("terms") or []) + [str(value)]
        # "a dash camera", but "Bluetooth" and "parking sensors" take no article.
        item = f"a {value}" if value in COUNTABLE_EQUIPMENT else str(value)
        if listing:
            having = [label[id(v)] for v in vehicles if _has_equipment(v, terms)]
            if having:
                return [f"Vehicles that list {item} among their features: {_join(having)}."]
            return [f"None of the listed vehicles has {item} in its recorded features, so I can't confirm one. "
                    "Please check with the rental team."]
        for v in vehicles:
            if _has_equipment(v, terms):
                lines.append(f"Yes, the {label[id(v)]} lists {item} among its features.")
            else:
                lines.append(f"The {label[id(v)]}'s recorded features don't include {item}, so I can't confirm it has it; "
                             "please check with the rental team.")
        return lines
    for v in vehicles:
        car = label[id(v)]
        if feature == "has_gps":
            has = bool(v.get("gps_navigation"))
            lines.append(f"{'Yes' if has else 'No'}, the {car} {'has' if has else 'does not have'} GPS navigation.")
        elif feature == "has_aircon":
            has = bool(v.get("air_conditioning"))
            lines.append(f"{'Yes' if has else 'No'}, the {car} {'has' if has else 'does not have'} air conditioning.")
        else:
            column = {"transmissions": "transmission", "fuel_types": "fuel_type", "colors": "color"}[feature]
            actual = str(v.get(column) or "not recorded")
            wanted = [str(item).lower() for item in value]
            matches = actual.lower() in wanted
            noun = {"transmissions": "transmission", "fuel_types": "fuel", "colors": "colour"}[feature]
            if len(wanted) > 1:
                # "Diesel or gasoline?" is not a yes/no question.
                lines.append(f"The {car}'s {noun} is {actual}.")
            else:
                lines.append(f"{'Yes' if matches else 'No'}, the {car}'s {noun} is {actual}.")
    return lines


def _compare_lines(compare, vehicles, label):
    if len(vehicles) < 2:
        return [f"I found only the {label[id(v)]} to compare." for v in vehicles[:1]]
    by_price = sorted(vehicles, key=lambda v: v["daily_rental_price"], reverse=True)
    lines = []
    if compare.get("by") in ("price", "all"):
        high, low = by_price[0], by_price[-1]
        if high["daily_rental_price"] == low["daily_rental_price"]:
            lines.append(f"They cost the same per day: {money(high['daily_rental_price'])}.")
        elif compare.get("lead") == "cheaper":
            # "Which one is cheaper?" / "alin ang mas mura?": the answer names the cheaper vehicle first.
            if len(vehicles) > 2:
                lines.append(f"The {label[id(low)]} is the cheapest at {money(low['daily_rental_price'])} per day; "
                             f"the most expensive is the {label[id(high)]} at {money(high['daily_rental_price'])} per day.")
            else:
                lines.append(f"The {label[id(low)]} is cheaper at {money(low['daily_rental_price'])} per day; "
                             f"the {label[id(high)]} is {money(high['daily_rental_price'])} per day.")
        else:
            lines.append(f"The {label[id(high)]} is more expensive at {money(high['daily_rental_price'])} per day; "
                         f"the {label[id(low)]} is {money(low['daily_rental_price'])} per day.")
    if compare.get("by") in ("seats", "all"):
        by_seats = sorted(vehicles, key=lambda v: v.get("seating_capacity") or 0, reverse=True)
        lines.append(f"The {label[id(by_seats[0])]} seats {by_seats[0].get('seating_capacity')}; "
                     + "; ".join(f"the {label[id(v)]} seats {v.get('seating_capacity')}" for v in by_seats[1:]) + ".")
    if compare.get("by") == "all":
        lines += [f"- {label[id(v)]}: {_specs(v)}" for v in vehicles]
    return lines


def _quote_lines(vehicles, data, label):
    days = data.get("rental_days")
    period = data.get("period")
    plural = "s" if days != 1 else ""
    when = f" ({period['start_date']} to {period['end_date']}, return date not counted)" if period else ""

    def extras(v, quote):
        text = ""
        if not quote.get("meets_minimum"):
            text += f"; it has a {quote['minimum_days']}-day minimum, so {days} day{plural} can't be booked"
        elif quote.get("exceeds_maximum"):
            text += f"; rentals can't be longer than {quote.get('maximum_days', 21)} days, so {days} days can't be booked"
        elif quote.get("special_request_required"):
            text += (f"; rentals over {quote.get('standard_maximum_days', 7)} days must be sent as a special-duration request "
                     "on the reservation form")
        if v.get("available_for_period") is False:
            text += "; it is already booked for those dates"
        elif v.get("available_for_period") is True:
            text += "; it is free for those dates"
        return text

    if len(vehicles) == 1:
        v = vehicles[0]
        quote = v["quote"]
        return [f"Renting the {label[id(v)]} for {days} day{plural}{when} costs {money(quote['base_total'])} "
                f"({money(quote['daily_rate'])} x {days} day{plural}){deposit(quote)}{extras(v, quote)}."]
    lines = [f"Estimate for {days} day{plural}{when}:"]
    for v in vehicles:
        quote = v["quote"]
        lines.append(f"- {label[id(v)]}: {money(quote['base_total'])} ({money(quote['daily_rate'])} x {days} day{plural})"
                     f"{deposit(quote)}{extras(v, quote)}")
    return lines


def deposit(quote):
    amount = quote.get("security_deposit")
    return (f", plus a {money(amount)} security deposit shown separately" if amount and float(amount) > 0
            else "; no security deposit is listed")


def _recommend_lines(vehicles, args, label, limit=3):
    """Up to three rentable vehicles that fit, each with why, then what would narrow it down.

    With a group size, the closest fit comes first (enough seats without paying for a much bigger car); otherwise
    the cheapest, the roomiest and the best equipped are picked so the choices differ."""
    need = args.get("min_seats")
    seats = lambda v: v.get("seating_capacity") or 0  # noqa: E731
    price = lambda v: float(v.get("daily_rental_price") or 0)  # noqa: E731
    if need:
        ranked = sorted(vehicles, key=lambda v: (seats(v) - need, price(v)))
    else:
        cheapest = min(vehicles, key=price)
        roomiest = max(vehicles, key=lambda v: (seats(v), -price(v)))
        equipped = max(vehicles, key=lambda v: (bool(v.get("gps_navigation")) + bool(v.get("air_conditioning"))
                                              + len(v.get("additional_features") or []), -price(v)))
        ranked = list(dict.fromkeys([id(cheapest), id(roomiest), id(equipped)] + [id(v) for v in sorted(vehicles, key=price)]))
        ranked = [next(v for v in vehicles if id(v) == key) for key in ranked]
    picks = ranked[:limit]
    cheapest_price = min(price(v) for v in vehicles)
    most_seats = max(seats(v) for v in vehicles)

    def reasons(v):
        why = []
        if need and seats(v) - need <= 1:
            why.append(f"fits {need} with {seats(v)} seats")
        elif seats(v) == most_seats and len(vehicles) > 1:
            why.append(f"the roomiest, {seats(v)} seats")
        else:
            why.append(f"{seats(v)} seats")
        if price(v) == cheapest_price and len(vehicles) > 1:
            why.append("the lowest daily rate")
        extras = [name for name, present in (("automatic", str(v.get("transmission") or "").casefold() in ("automatic", "cvt")),
                                             ("GPS", v.get("gps_navigation")), ("diesel", str(v.get("fuel_type") or "").casefold() == "diesel"))
                  if present]
        if extras:
            why.append(", ".join(extras))
        return "; ".join(why)

    lead = (f"For {need} people, here's what you can rent now, closest fit first:" if need
            else f"Here {'are' if len(picks) > 1 else 'is'} {len(picks)} good {'options' if len(picks) > 1 else 'option'} you can rent now:")
    lines = [lead]
    lines += [f"- {label[id(v)]}: {money(price(v))} per day ({reasons(v)})" for v in picks]
    if len(vehicles) > len(picks):
        lines.append(f"{len(vehicles) - len(picks)} more {'is' if len(vehicles) - len(picks) == 1 else 'are'} available; ask to see all of them.")
    missing = [text for key, text in (("min_seats", "how many people"), ("max_daily_rate", "your daily budget"), ("rental_days", "how many days"))
               if not args.get(key)]
    if missing:
        lines.append(f"Tell me {_join(missing)}, and I'll narrow it down.")
    return lines


def _availability_lines(vehicles, data, role, purpose, label):
    period = data.get("period")
    if period:
        off_fleet = [v for v in vehicles if v.get("status") not in (None, "", "available")]
        free = [label[id(v)] for v in vehicles if v.get("available_for_period") and v not in off_fleet]
        taken = [label[id(v)] for v in vehicles if v.get("available_for_period") is False and v not in off_fleet]
        lines = [f"For {period['start_date']} to {period['end_date']} ({period['rental_days']} days, return date not counted):"]
        lines.append("Available: " + (_join(free) if free else "none of the matching vehicles") + ".")
        if taken:
            lines.append("Already booked then: " + _join(taken) + ".")
        return lines + _off_fleet_lines(off_fleet, label)
    if role == "admin" and purpose == "status":
        groups = {}
        for v in vehicles:
            groups.setdefault(v.get("status") or "unknown", []).append(label[id(v)])
        return [f"{status.title()}: {_join(names)}." for status, names in sorted(groups.items())]
    if role == "admin":
        available = [v for v in vehicles if v.get("status") == "available"]
        others = {}
        for v in vehicles:
            if v.get("status") != "available":
                others[v.get("status") or "unknown"] = others.get(v.get("status") or "unknown", 0) + 1
        total = max(int(data.get("total_matching") or 0), len(vehicles))  # every match, not only the rows returned
        unseen = total - len(vehicles)
        lines = [(f"{len(available)} of the {total} vehicles {'is' if len(available) == 1 else 'are'} available now: "
                  + _join(f"{label[id(v)]} ({money(v['daily_rental_price'])}/day)" for v in available) + ".") if available
                 else f"None of the {total} vehicles is available now."]
        if unseen:
            # Available cars are listed first, so every vehicle past the limit is one that can't be rented now.
            lines.append(f"The other {total - len(available)} can't be rented right now. Ask \"fleet status\" for the totals by status.")
        elif others:
            statuses = sorted(others)
            lines.append("The others: " + ", ".join(f"{count} {status}" for status, count in sorted(others.items()))
                         + f". Ask \"which cars are {statuses[0]}?\" for their names.")
        return lines
    return ["Available for rent right now: " + _join(f"{label[id(v)]} ({money(v['daily_rental_price'])}/day)" for v in vehicles) + "."]


def _specs(v):
    parts = [f"{v.get('year')}" if v.get("year") else None, v.get("category"), v.get("transmission"), v.get("fuel_type"),
             f"{v.get('seating_capacity')} seats" if v.get("seating_capacity") else None,
             "air conditioning" if v.get("air_conditioning") else "no air conditioning" if v.get("air_conditioning") is False else None,
             "GPS" if v.get("gps_navigation") else None, v.get("color")]
    features = [str(item) for item in (v.get("additional_features") or []) if isinstance(item, str)]
    text = ", ".join(part for part in parts if part)
    if features:
        text += "; features: " + ", ".join(features)
    return text + f"; {money(v['daily_rental_price'])} per day, minimum {v.get('minimum_rental_duration')} day(s)"


def _vocabulary_names(vocabulary):
    return [f"{entry['make']} {entry['model']}" for entry in vocabulary.get("models", []) if entry.get("model")][:15]


# --------------------------------------------------------------------------- bookings

def _own_bookings(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["bookings.mine"].get("data") or {}
    rows = data.get("reservations") or []
    args = plan["retrievals"][0].get("args") or {}
    upcoming = args.get("upcoming")
    purpose = plan.get("purpose") or "list"
    lines = []
    wanted = plan.get("vehicle_names") or []
    if wanted and rows:
        # "When do I pick up the Vios?": only the reservations of that vehicle.
        rows = [r for r in rows if _names_vehicle(r, wanted)]
        if not rows:
            return f"You have no reservation for the {_join(wanted)} on this account."

    def one(r):
        when = f"{r['start_date']} to {r['end_date']}"
        extra = (f", pickup in {r['days_until_pickup']} day(s)" if r.get("days_until_pickup") else
                 f", {r['days_remaining']} day(s) left" if r.get("is_active") else "")
        return (f"#{r['id']} {name(r)}: {when} ({r['rental_days']} day(s)), status {r['status']}, total "
                + (money(r["total_amount"]) if r.get("total_amount") is not None else "not recorded") + extra)

    if not rows and args.get("reservation_id"):
        lines.append(f"I couldn't find reservation #{args['reservation_id']} on your account.")
    elif not rows and upcoming and data.get("all_records_count"):
        lines.append("You have no active or upcoming reservations. "
                     f"Your account has {data['all_records_count']} past or closed reservation(s); ask to see all your reservations to list them.")
    elif not rows:
        lines.append("You don't have any reservations on this account. You can book from Vehicles (/users/vehicles).")
    elif purpose == "payment":
        # "Did my payment go through?": the open reservations first, else the latest one.
        focus = [r for r in rows if r.get("status") in ("pending", "confirmed") and not r.get("returned")] or rows[:1]
        if args.get("reservation_id"):
            focus = rows[:1]
        for r in focus[:3]:
            lines.append(f"Reservation #{r['id']} for the {name(r)} ({r['start_date']} to {r['end_date']}): "
                         + PAYMENT_MEANINGS.get(r.get("payment_status"), f"its status is {r.get('status')}, but I don't have its payment "
                                                "details here; My Reservations shows them."))
        if any(r.get("payment_status") in ("unpaid", "failed", "pending") for r in focus[:3]):
            lines.append("Pay from My Reservations (/users/reservations). A screenshot or receipt doesn't confirm a booking; "
                         "only the payment provider's verification does.")
    elif purpose in ("status", "details"):
        asked = plan.get("asked_statuses") or []
        matching = [r for r in rows if r.get("status") in asked]
        if matching and "rejected" in asked:
            lines.append("The system does not record a reason for a rejection; the rental team can tell you why.")
        # Lead with the reservation the question is most likely about: an active or upcoming one first.
        focus = matching or [r for r in rows if r.get("is_active") or r.get("days_until_pickup") is not None] or rows[:1]
        # "Is my booking confirmed?" when it is pending: "No, ..." and the actual status in the same sentence.
        answer_no = asked and not matching and not args.get("reservation_id")
        for r in focus[:3]:
            meaning = STATUS_MEANINGS.get(r.get("status"))
            lines.append((f"No, it isn't {_join(asked)}: your" if answer_no else "Your")
                         + f" reservation #{r['id']} for the {name(r)} ({r['start_date']} to {r['end_date']}) is {r['status']}"
                         + (f": {meaning}." if meaning else "."))
            answer_no = False
        if purpose == "details":
            lines += [f"- {one(r)}" for r in focus[:1]]
    elif purpose == "length":
        # "How many days is my booking?": the open reservation first, else the latest one.
        chosen = [r for r in rows if r.get("status") in ("pending", "confirmed")] or rows[:1]
        for r in chosen[:3]:
            days = r.get("rental_days")
            lines.append(f"Your reservation #{r['id']} for the {name(r)} is {days} day{'s' if days != 1 else ''}: "
                         f"{r['start_date']} to {r['end_date']} (the return date is not counted), status {r['status']}.")
    elif purpose == "total":
        asked = plan.get("asked_statuses") or []
        chosen = [r for r in rows if r.get("status") in asked] if asked else rows
        if not wanted and not asked:
            # "My reservation total": the open (pending or confirmed) reservations first.
            chosen = [r for r in rows if r.get("status") in ("pending", "confirmed")] or rows
        if not chosen:
            lines.append(f"You have no {_join(asked)} reservations.")
        for r in chosen[:5]:
            verb = "was" if r.get("status") in ("completed", "cancelled", "rejected") else "is"
            total = money(r["total_amount"]) if r.get("total_amount") is not None else "not recorded"
            lines.append(f"Your reservation #{r['id']} for the {name(r)} ({r['start_date']} to {r['end_date']}, {r['status']}) "
                         f"{verb} {total} in total.")
        if notes and chosen:
            notes.add(lines, "no_payments")
    elif purpose == "last":
        # Past or closed reservations: neither upcoming nor in progress. Completed ones are actual rentals.
        past = sorted((r for r in rows if r.get("days_until_pickup") is None and not r.get("is_active")),
                      key=lambda row: row["start_date"], reverse=True)
        completed = [r for r in past if r.get("status") == "completed"]
        if completed:
            r = completed[0]
            lines.append(f"Your most recent completed rental was the {name(r)}: reservation #{r['id']}, {r['start_date']} to {r['end_date']}.")
        elif past:
            r = past[0]
            lines.append(f"You have no completed rentals. Your most recent past reservation was #{r['id']} for the {name(r)} "
                         f"({r['start_date']} to {r['end_date']}), status {r['status']}.")
        else:
            lines.append("You have no past rentals on this account yet.")
        upcoming_rows = [r for r in rows if r.get("days_until_pickup") is not None or r.get("is_active")]
        if upcoming_rows:
            r = sorted(upcoming_rows, key=lambda row: row["start_date"])[0]
            lines.append(f"Your next one is #{r['id']} ({name(r)}), from {r['start_date']}, status {r['status']}.")
    elif purpose == "next":
        r = next((row for row in sorted(rows, key=lambda row: row["start_date"]) if row.get("days_until_pickup") is not None or row.get("is_active")), None)
        if r is None:
            lines.append("You have no upcoming pickup.")
        elif r.get("is_active"):
            lines.append(f"Your rental #{r['id']} ({name(r)}) started on {r['start_date']} and is in progress; the return is due on {r['end_date']}.")
        else:
            lines.append(f"Your next pickup is on {r['start_date']} (in {r['days_until_pickup']} day(s)): reservation #{r['id']}, "
                         f"{name(r)}, return on {r['end_date']}, status {r['status']}.")
    elif purpose == "days_left":
        active = [r for r in rows if r.get("is_active")]
        if active:
            r = active[0]
            lines.append(f"Your rental #{r['id']} ({name(r)}) has {r['days_remaining']} day(s) left; the return is due on {r['end_date']}.")
        else:
            nxt = sorted((r for r in rows if r.get("days_until_pickup") is not None), key=lambda row: row["start_date"])
            lines.append("You don't have a rental in progress right now."
                         + (f" Your next one, #{nxt[0]['id']} ({name(nxt[0])}), starts on {nxt[0]['start_date']}, in {nxt[0]['days_until_pickup']} day(s)."
                            if nxt else ""))
    elif purpose == "spent":
        totals = data.get("totals_by_status") or {}
        open_totals = {status: total for status, total in totals.items() if status in ("pending", "confirmed")}
        if plan.get("owe") and open_totals:
            lines.append("Your open reservations (pending or confirmed) total "
                         + "; ".join(f"{status} {money(total)}" for status, total in sorted(open_totals.items())) + ".")
        elif plan.get("owe"):
            lines.append("You have no pending or confirmed reservations, so no open reservation total is recorded.")
        elif totals:
            lines.append("Your recorded reservation totals: "
                         + "; ".join(f"{status} {money(total)}" for status, total in sorted(totals.items())) + ".")
        else:
            lines.append("There are no recorded totals on your reservations.")
        if notes:
            notes.add(lines, "no_payments")
    if plan.get("guide") and guide:
        found = guide()
        if found:
            lines.insert(0, found[0])
            plan.setdefault("_guide_sources", []).extend(found[1])
    if rows and purpose in ("list",) or (rows and plan.get("guide")):
        count = data.get("total_count", len(rows))
        lines.append(f"You have {count} {'upcoming ' if upcoming else ''}reservation{'s' if count != 1 else ''}:")
        lines += [f"- {one(r)}" for r in rows]
        if data.get("total_count", len(rows)) > len(rows):
            lines.append(f"Showing the latest {len(rows)} of {data['total_count']}.")
    if rows:
        lines.append("Full details are on My Reservations (/users/reservations).")
    return "\n".join(lines)


def _names_vehicle(row, wanted):
    """True if the reservation row is for one of the named vehicles ("Vios", "Honda City")."""
    car = name(row).casefold()
    return any(all(part in car for part in str(item).casefold().split()) for item in wanted)


def _reservation_summary(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["reservations.summary"].get("data") or {}
    rows = data.get("reservations") or []
    wanted = plan.get("vehicle_names") or []
    if wanted:
        # "Who booked the Fortuner?": that vehicle's reservations, with the customer's name.
        rows = [r for r in rows if _names_vehicle(r, wanted)]
        if not rows:
            return f"There are no reservations for the {_join(wanted)}."
        lines = [f"Reservations for the {_join(wanted)}:"]
        lines += [f"- #{r['id']} {r.get('customer_name') or 'customer'}: {r['start_date']} to {r['end_date']}, {r['status']}, "
                  + (money(r['total_amount']) if r.get("total_amount") is not None else "no total") for r in rows[:10]]
        return "\n".join(lines)
    counts = data.get("counts_by_status") or {}
    args = (plan.get("retrievals") or [{}])[0].get("args") or {}
    if plan.get("rank_customers"):
        # "Who has the most bookings?": customers ranked by how many reservations they made (cancelled ones included).
        tally = {}
        if data.get("by_customer") and not args:
            # The Back-End counts every reservation, not only the rows listed.
            tally = {item["customer_name"]: {"count": item["reservations"], "value": 0.0} for item in data["by_customer"]}
            rows_counted = data.get("rows_counted", len(rows))
        else:
            rows_counted = len(rows)
        for r in ([] if tally else rows):
            customer = r.get("customer_name") or "Unnamed customer"
            entry = tally.setdefault(customer, {"count": 0, "value": 0.0})
            entry["count"] += 1
            if r.get("status") in ("pending", "confirmed", "completed") and r.get("total_amount") is not None:
                entry["value"] += float(r["total_amount"])
        if not tally:
            return "There are no reservations to rank."
        ranked = sorted(tally.items(), key=lambda item: (-item[1]["count"], -item[1]["value"], item[0]))
        top_count = ranked[0][1]["count"]
        leaders = [customer for customer, entry in ranked if entry["count"] == top_count]
        window = f" from {args['from']} to {args.get('to') or args['from']}" if args.get("from") else ""
        lines = [f"{_join(leaders)} {'has' if len(leaders) == 1 else 'have'} the most reservations{window}: {top_count}."]
        # Counts only: summed peso amounts would not appear in the records, and the answer check refuses unrecorded amounts.
        lines += [f"- {customer}: {entry['count']} reservation(s)" for customer, entry in ranked[:10]]
        if data.get("all_records_count", 0) > rows_counted and not args:
            lines.append(f"Counted from the latest {rows_counted} of {data['all_records_count']} reservations.")
        return "\n".join(lines)
    if args.get("reservation_id"):
        if not rows:
            return f"No reservation #{args['reservation_id']} was found."
        r = rows[0]
        meaning = STATUS_MEANINGS.get(r.get("status"), "").replace("your dates", "those dates")
        return (f"Reservation #{r['id']} ({r.get('customer_name') or 'customer'}): {name(r)}, {r['start_date']} to {r['end_date']}, "
                f"total " + (money(r["total_amount"]) if r.get("total_amount") is not None else "not recorded")
                + f". Status {r['status']}" + (f": {meaning}." if meaning else "."))
    filters = []
    if args.get("statuses"):
        filters.append(" or ".join(args["statuses"]))
    if args.get("pickup_on"):
        filters.append(f"picked up on {args['pickup_on']}")
    if args.get("return_on"):
        filters.append(f"returned on {args['return_on']}")
    if args.get("from"):
        filters.append(f"overlapping {args['from']} to {args.get('to') or args['from']}")
    if args.get("customer_name"):
        filters.append(f"for customers matching \"{args['customer_name']}\"")
    if filters:
        what = " ".join(filters)
        count = data.get("total_count", len(rows))
        lines = [f"{count} reservation(s) {what}." if count else f"No reservations {what}."]
    else:
        lines = [f"Reservations in the system: {data.get('all_records_count', 0)}"
                 + (" (" + ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())) + ")" if counts else "") + "."]
    lines += [f"- #{r['id']} {r.get('customer_name') or 'customer'}: {name(r)}, {r['start_date']} to {r['end_date']}, "
              f"{r['status']}, " + (money(r['total_amount']) if r.get("total_amount") is not None else "no total") for r in rows[:10]]
    if len(rows) > 10:
        lines.append(f"Showing 10 of {len(rows)}; see Reservations in the admin pages for the rest.")
    return "\n".join(lines)


def _reassessment_due(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    """Monthly vehicle reassessments (administrators): overdue first, then due within 7 days, then never assessed."""
    data = by_tool["assessments.due"].get("data") or {}
    vehicles = data.get("vehicles") or []
    if not vehicles:
        return "No vehicles matched, so there are no reassessments to report."

    def label(v):
        return name(v) + (f" ({v['plate_number']})" if v.get("plate_number") else "")

    groups = (
        ("overdue", "Overdue", lambda v: f"due {v['due_on']}, {abs(v['days_until_due'] or 0)} day(s) late"),
        ("due_soon", "Due within 7 days", lambda v: f"due {v['due_on']}"),
        ("never_assessed", "Never assessed", lambda v: "no assessment recorded"),
    )
    lines = []
    for status, heading, detail in groups:
        rows = sorted((v for v in vehicles if v.get("status") == status), key=lambda v: (v.get("days_until_due") or 0, label(v)))
        if rows:
            lines.append(f"{heading}: {len(rows)}")
            lines += [f"- {label(v)}: {detail(v)}" for v in rows[:10]]
    current = sum(1 for v in vehicles if v.get("status") == "current")
    if not lines:
        return (f"All {current} vehicle(s) are up to date with the monthly reassessment."
                if current else "No reassessment status is recorded for these vehicles.")
    if current:
        lines.append(f"Up to date: {current} vehicle(s).")
    lines.append("Assess them from Predictive Maintenance (/admin/maintenance/predictions).")
    return "\n".join(lines)


def _maintenance_costs(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["maintenance.costs"].get("data") or {}
    window = f" from {data['from'] or 'the first record'} to {data['to'] or 'today'}" if data.get("from") or data.get("to") else ""
    if not data.get("records"):
        return f"No maintenance records with costs are stored{window}."
    if not float(data.get("total_cost") or 0) and data.get("records_without_cost") == data.get("records"):
        # Records exist but none has a cost: there is nothing to rank, so don't name a "highest" PHP 0.00.
        return (f"None of the {data['records']} maintenance record(s){window} has a cost entered, so costs can't be compared yet. "
                "Enter the cost when you record maintenance (Maintenance Records, /admin/maintenance/records) and I can rank vehicles by spend.")
    lines = [f"Total recorded maintenance cost{window}: {money(data['total_cost'])} across {data['records']} record(s)."]
    per_vehicle = [v for v in data.get("per_vehicle") or [] if float(v.get("total_cost") or 0) > 0]
    if len(per_vehicle) > 1:
        top = max(per_vehicle, key=lambda v: float(v["total_cost"]))
        lines.insert(0, f"The highest recorded maintenance cost is the {name(top)} at {money(top['total_cost'])}.")
    lines += [f"- {name(v)}" + (f" ({v['plate_number']})" if v.get("plate_number") else "")
              + f": {money(v['total_cost'])} ({v['records']} record(s))" for v in per_vehicle[:10]]
    if len(per_vehicle) > 10:
        lines.append(f"...and {len(per_vehicle) - 10} more vehicles.")
    if data.get("records_without_cost"):
        lines.append(f"{data['records_without_cost']} record(s) have no cost entered and are not included.")
    return "\n".join(lines)


def _vehicle_status(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    vehicles = (by_tool["vehicles.status"].get("data") or {}).get("vehicles") or []
    if not vehicles:
        asked = _join((plan.get("requested") or {}).get("models") or ["that vehicle"])
        return f"I couldn't find {asked} in the fleet."
    label = labels(vehicles)
    lines = []
    for v in vehicles:
        mileage = f"{v['current_mileage']:,} km" if v.get("current_mileage") is not None else "no mileage recorded"
        line = f"{label[id(v)]}" + (f" ({v['plate_number']})" if v.get("plate_number") else "") + f": status {v.get('status')}, {mileage}."
        service = v.get("service") or {}
        if service:
            line += f" Service: {STATUS_LABELS.get(service.get('service_status'), service.get('service_status'))}"
            if service.get("last_service_date"):
                line += f", last completed service {service['last_service_date']}"
            if service.get("estimated_due_date") and service.get("service_status") != "insufficient_history":
                line += f", date-based due {service['estimated_due_date']}"
            line += "."
        last = v.get("last_service")
        if last:
            line += (f" Latest log entry: #{last['maintenanceID']} {last['maintenanceType']} on {last['maintenanceDate']} ({last['maintenanceStatus']})"
                     + (f", cost {money(last['cost'])}" if last.get("cost") not in (None, "") else "") + ".")
        current, upcoming = v.get("current_reservation"), v.get("next_reservation")
        if current:
            line += f" Currently booked: #{current['id']} until {current['end_date']} ({current.get('customer_name') or 'customer'}, {current['status']})."
        if upcoming:
            line += f" Next booking: #{upcoming['id']} {upcoming['start_date']} to {upcoming['end_date']} ({upcoming.get('customer_name') or 'customer'}, {upcoming['status']})."
        if not current and not upcoming:
            line += " No current or upcoming booking."
        lines.append(line)
    return "\n".join(lines)


def _revenue(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["reservations.revenue"].get("data") or {}
    by_status = data.get("by_status") or {}
    window = f" for reservations starting {data['from'] or 'any time'} to {data['to'] or 'any time'}" if data.get("from") or data.get("to") else ""
    if not by_status:
        period = f" with pickup dates from {data['from'] or 'the start'} to {data['to'] or 'today'}" if window else ""
        return f"There are no reservations{period}, so no booked value is recorded."
    lines = []
    for status in plan.get("focus_statuses") or []:
        item = by_status.get(status)
        lines.append(f"{status.capitalize()} reservations{window}: {money(item['total'])} from {item['reservations']} reservation(s)."
                     if item else f"There are no {status} reservations{window}.")
    breakdown = plan.get("breakdown")
    if breakdown in ("vehicle", "customer") and data.get(f"by_{breakdown}"):
        # "Revenue per car", "top customers by revenue": booked value of pending, confirmed and completed reservations.
        rows = data[f"by_{breakdown}"]
        who = (lambda item: name(item)) if breakdown == "vehicle" else (lambda item: item["customer_name"])
        lines.append(f"{'The ' if breakdown == 'vehicle' else ''}{who(rows[0])} {'brings' if breakdown == 'vehicle' else 'has'} the highest booked value"
                     f"{window}: {money(rows[0]['booked'])} from {rows[0]['reservations']} reservation(s).")
        lines += [f"- {who(item)}: {money(item['booked'])} from {item['reservations']} reservation(s)" for item in rows]
        lines.append(f"Overall booked value{window}: {money(data.get('booked_total', 0))}.")
        return "\n".join(lines)
    lines.append(f"{'Overall b' if lines else 'B'}ooked value{window}: {money(data.get('booked_total', 0))} (pending, confirmed and completed reservations).")
    if data.get("paid_total"):
        lines.append(f"Of that, {money(data['paid_total'])} is paid through verified online payments.")
    lines += [f"- {status}: {money(item['total'])} from {item['reservations']} reservation(s)" for status, item in by_status.items()]
    if notes:
        notes.add(lines, "booked_value")
    else:
        lines.append(NOTES["booked_value"])
    return "\n".join(lines)


def _utilization(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["fleet.utilization"].get("data") or {}
    vehicles = data.get("vehicles") or []
    if not vehicles:
        return "There are no vehicles to report utilisation for."
    label = labels(vehicles)
    top = vehicles[0]
    if not any(v.get("booked_days") for v in vehicles):
        lines = [f"From {data['from']} to {data['to']} ({data['window_days']} days), no vehicle has any booked days, so utilisation is 0%."]
    else:
        lines = [f"From {data['from']} to {data['to']} ({data['window_days']} days), the most booked vehicle is the {label[id(top)]} "
                 f"at {top['utilization_percent']}% ({top['booked_days']} booked day(s))."]
    lines += [f"- {label[id(v)]}: {v['utilization_percent']}% ({v['booked_days']} day(s))" for v in vehicles]
    lines.append("Booked days count pending, confirmed and completed reservations.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- fleet and maintenance

def _fleet_status(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["fleet.status_counts"].get("data") or {}
    counts = data.get("counts") or {}
    total = data.get("total", 0)
    summary = (f"The fleet has {total} vehicle(s): "
               + (", ".join(f"{status} {count}" for status, count in sorted(counts.items())) if counts else "no status counts recorded") + ".")
    asked = plan.get("asked_status")
    if asked and total:
        # "Are they all rented?": yes or no first, then the totals.
        key = {"in maintenance": "maintenance", "booked": "rented"}.get(asked, asked)
        matching = counts.get(key, 0)
        lead = (f"Yes, all {total} are {asked}." if matching == total
                else f"No, none of the {total} vehicles is {asked}." if matching == 0
                else f"No, {matching} of the {total} vehicles {'is' if matching == 1 else 'are'} {asked}.")
        notes = " Vehicles marked unavailable can't be booked until an administrator changes their status." if counts.get("unavailable") else ""
        return f"{lead} {summary}{notes}"
    return summary


def _maintenance_due(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["maintenance.due"].get("data") or {}
    rows = data.get("vehicles") or []
    if not rows:
        return "No vehicles matched, so there is no service status to report."
    lines = []
    if plan.get("asked_overdue"):
        # "Why is the Hilux overdue?": say first whether it is.
        overdue = [r for r in rows if r.get("service_status") == "overdue"]
        others = [r for r in rows if r.get("service_status") != "overdue"]
        if not overdue:
            lines.append(("It isn't overdue: " if len(rows) == 1 else "No, none of them is overdue: ")
                         + "; ".join(f"{_plate_label(r)}: {STATUS_LABELS.get(r.get('service_status'), '').lower()}, {_service_reason(r)}" for r in rows[:4]) + ".")
            return "\n".join(lines)
        lines.append(f"{_join(_plate_label(r) for r in overdue)} {'is' if len(overdue) == 1 else 'are'} overdue because: "
                     + "; ".join(_service_reason(r) for r in overdue) + "."
                     + (f" {_join(_plate_label(r) for r in others)} {'is' if len(others) == 1 else 'are'} not." if others else ""))
        return "\n".join(lines)
    lines.append(f"Service schedule (every {BASELINE_SERVICE_KM:,} km or {BASELINE_SERVICE_DAYS} days since the last completed service, "
                 "whichever comes first, or a recorded next-service date):")
    order = ["overdue", "approaching", "insufficient_history", "not_due"]
    many = len(rows) > 8
    for status in order:
        group = [r for r in rows if r.get("service_status") == status]
        if not group:
            continue
        if status == "not_due" and many:
            # A long fleet list: the cars that need nothing are counted, not listed.
            lines.append(f"Not due: {len(group)} vehicle{'s' if len(group) != 1 else ''}.")
            continue
        lines.append(f"{STATUS_LABELS[status]}:")
        for r in group[:10]:
            type_text = f" ({r['last_service_type']})" if r.get("last_service_type") and status != "insufficient_history" else ""
            lines.append(f"- {_plate_label(r)}: {_service_reason(r)}{type_text}.")
        if len(group) > 10:
            lines.append(f"...and {len(group) - 10} more.")
    assessed = _rows(by_tool, "assessments.condition")
    if assessed:
        # The schedule is only half of it: what the latest assessments found.
        needing = sorted((v for v in assessed if _concerns(v) or _safety_findings(v)), key=lambda v: -_priority_score(v, None))
        critical = [v for v in needing if any(s["rating"] == "critical" for s in _concerns(v)) or
                    any(f["severity"] == "critical" for f in _safety_findings(v))]
        if needing:
            lines.append(f"By condition, though, {len(needing)} of {len(assessed)} assessed vehicles have systems needing work"
                         + (f", {len(critical)} with critical or safety items (worst: "
                            + ", ".join(_plate_label(v) for v in critical[:3]) + ")" if critical else "")
                         + ". Ask \"what should we service first\" for the ranked list.")
        else:
            lines.append(f"By condition, none of the {len(assessed)} assessed vehicles has a system rated poor or critical.")
    if notes:
        notes.add(lines, "schedule_estimate")
    else:
        lines.append(NOTES["schedule_estimate"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- condition-based maintenance (assessments)

RATING_ORDER = {"critical": 0, "poor": 1, "fair": 2, "good": 3, "excellent": 4}
BASELINE_SERVICE_KM = 5000
BASELINE_SERVICE_DAYS = 180
CONDITION_SOURCE_NOTE = ("These rankings come from the recorded assessments (ratings, findings and their trend) and the verified "
                         "fault history, not from a machine-learning forecast.")
MODEL_PENDING_NOTE = ("Model-based service-need predictions are unavailable (no service-need model has passed its acceptance check), "
                      "so this isn't a model forecast: it's the assessment ratings, findings, their trend and the verified fault history.")


def _plate_label(v):
    return name(v) + (f" ({v['plate_number']})" if v.get("plate_number") else "")


def _rows(by_tool, tool, key="vehicles"):
    item = by_tool.get(tool) or {}
    return ((item.get("data") or {}).get(key) or []) if item.get("status") == "ok" else []


CONDITION_UNAVAILABLE = ("The vehicle assessment data isn't available from the server right now, so this can't include the "
                         "assessment ratings and findings")


def _condition_missing(by_tool):
    """The assessment records were asked for but didn't come back: an older Back-End without the tool, or a database error.
    Answers then say the data is unavailable instead of claiming no assessments exist."""
    item = by_tool.get("assessments.condition")
    return item is not None and item.get("status") != "ok"


def _worsened(system):
    return system.get("previous_score") is not None and system.get("rating_score") is not None \
        and system["rating_score"] < system["previous_score"]


def _rating_text(system, safety=False):
    """ "critical (was fair; safety-critical)": the rating, its previous value and, when asked, the safety flag."""
    extra = []
    if system.get("previous_rating") and system["previous_rating"] != system["rating"]:
        extra.append(f"was {system['previous_rating']}")
    if safety and system.get("safety_critical"):
        extra.append("safety-critical")
    return system["rating"] + (f" ({'; '.join(extra)})" if extra else "")


def _concerns(v):
    """Systems needing attention, worst first: critical or poor, or fair and worse than at the previous assessment."""
    found = [s for s in v.get("systems") or [] if s.get("rating") in ("critical", "poor") or s.get("rating") == "fair" and _worsened(s)]
    return sorted(found, key=lambda s: (RATING_ORDER.get(s["rating"], 9), not s.get("safety_critical"), s.get("category") or ""))


def _safety_findings(v):
    return [f for f in v.get("findings") or [] if f.get("is_safety") and f.get("severity") in ("critical", "warning")]


def _due_by_id(by_tool):
    return {int(r["id"]): r for r in _rows(by_tool, "maintenance.due") if r.get("id") is not None}


def _service_reason(r):
    """Why a car is overdue or when it is due, in km and days, from the service-due row."""
    status = r.get("service_status")
    current, last_km = r.get("current_mileage"), r.get("last_service_odometer")
    since = current - last_km if current is not None and last_km is not None else None
    if status == "insufficient_history":
        return "no completed service with a date is recorded, so no schedule can be worked out"
    if since is not None and since < 0:
        # The car's mileage is below the odometer at its last service: one of the two entries is wrong.
        since = None
        mismatch = (f"its recorded mileage ({current:,} km) is lower than at its last service ({last_km:,} km), so one of those "
                    "odometer entries needs correcting")
    else:
        mismatch = None
    parts = []
    if r.get("last_service_date"):
        parts.append(f"last service {r['last_service_date']}" + (f" at {last_km:,} km" if last_km is not None else ""))
    if status == "overdue":
        if since is not None and since >= BASELINE_SERVICE_KM:
            parts.append(f"{since:,} km driven since then, past the {BASELINE_SERVICE_KM:,} km interval")
        elif r.get("next_scheduled_date"):
            parts.append(f"the scheduled date {r['next_scheduled_date']} has passed")
        else:
            parts.append(f"more than {BASELINE_SERVICE_DAYS} days have passed")
    else:
        due = []
        if last_km is not None:
            due.append(f"{last_km + BASELINE_SERVICE_KM:,} km")
        if r.get("estimated_due_date"):
            due.append(r["estimated_due_date"])
        if due:
            parts.append("next due at " + " or ".join(due) + ", whichever comes first")
        if since is not None:
            parts.append(f"{since:,} km driven since")
    if mismatch:
        parts.append(mismatch)
    return "; ".join(parts)


def _priority_score(v, due):
    concerns = _concerns(v)
    score = 100 * len([f for f in _safety_findings(v) if f["severity"] == "critical"]) + 30 * len(_safety_findings(v))
    score += sum(40 if s["rating"] == "critical" else 15 if s["rating"] == "poor" else 5 for s in concerns)
    score += 5 * sum(1 for s in concerns if _worsened(s))
    status = (due or {}).get("service_status")
    score += 25 if status == "overdue" else 10 if status == "approaching" else 0
    return score


def _first_action(v):
    finding = next((f for f in v.get("findings") or [] if f.get("severity") in ("critical", "warning")), None)
    return finding.get("recommended_action") if finding else None


def _model_line(fleet_risk, by_tool):
    """The registered model's results when one is accepted; otherwise one plain sentence instead of the long notice."""
    if fleet_risk is None or (by_tool.get("maintenance.model_inputs") or {}).get("status") != "ok":
        return None
    text = fleet_risk(_rows(by_tool, "maintenance.model_inputs"))
    return MODEL_PENDING_NOTE if not text or text.startswith("Model-based maintenance predictions are unavailable") else text


def _maintenance_condition(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    vehicles = _rows(by_tool, "assessments.condition")
    due = _due_by_id(by_tool)
    model = _model_line(fleet_risk, by_tool)
    if plan.get("purpose") == "vehicle":
        text = _vehicle_condition(plan, by_tool, vehicles, due)
        # A "will it break down?" question also says whether a model forecast exists.
        return text.replace("\n\n" + CONDITION_SOURCE_NOTE, "\n\n" + model) if model else text
    if not vehicles:
        lead = (CONDITION_UNAVAILABLE + "." if _condition_missing(by_tool) else
                "No vehicle assessments are recorded yet, so there's no condition data to rank. Record assessments under "
                "Predictive Maintenance (/admin/maintenance/predictions).")
        text = (lead + " The service schedule is below.\n\n" + _maintenance_due(plan, by_tool, role, vocabulary, guide, fleet_risk, notes)) \
            if due else lead
        return text + ("\n\n" + model if model else "")
    dates = sorted(v["assessed_on"] for v in vehicles if v.get("assessed_on"))
    focus = plan.get("focus") or "vehicles"
    lines = [f"From the latest assessments of {len(vehicles)} vehicle{'s' if len(vehicles) != 1 else ''}"
             + (f" ({dates[0]} to {dates[-1]})" if dates else "") + ":"]
    labels = {s["category_key"]: s["category"] for v in vehicles for s in v.get("systems") or []}

    # Vehicles at risk, worst first.
    ranked = sorted((v for v in vehicles if _concerns(v) or _safety_findings(v)), key=lambda v: -_priority_score(v, due.get(v["id"])))
    shown = 5 if focus == "vehicles" else 3
    at_risk = []
    if ranked:
        at_risk.append("Vehicles most at risk:" if focus == "vehicles" else "Vehicles to look at first:")
        for v in ranked[:shown]:
            systems = ", ".join(f"{s['category'].lower()} {_rating_text(s, safety=True)}" for s in _concerns(v)[:3])
            action = _first_action(v)
            at_risk.append(f"- {_plate_label(v)}: {systems or 'safety findings recorded'}." + (f" First: {action}" if action else ""))
        if len(ranked) > shown:
            at_risk.append(f"{len(ranked) - shown} more vehicle{'s have' if len(ranked) - shown != 1 else ' has'} items needing attention.")
    else:
        at_risk.append("No vehicle has a system rated poor or critical, or getting worse.")

    # Systems: how many cars have each one in poor or critical condition, and the verified failures per system.
    weak, critical = {}, {}
    for v in vehicles:
        for s in v.get("systems") or []:
            if s.get("rating") in ("critical", "poor"):
                weak.setdefault(s["category_key"], []).append(v)
            if s.get("rating") == "critical":
                critical.setdefault(s["category_key"], []).append(v)
    failures = {}
    for v in vehicles:
        for system, results in (v.get("outcomes") or {}).items():
            failures[system] = failures.get(system, 0) + int((results or {}).get("confirmed_failure") or 0)
    systems = []
    if weak:
        # The most vehicles needing work on a system first; more critical ones break a tie.
        top = sorted(weak.items(), key=lambda item: (-len(item[1]), -len(critical.get(item[0], [])), labels.get(item[0], item[0])))
        if focus == "systems":
            key, cars = top[0]
            systems.append(f"The system most likely to need work is {labels.get(key, key).lower()}: {len(cars)} of {len(vehicles)} vehicles "
                           f"have it poor or critical ({len(critical.get(key, []))} critical)"
                           + (f", and it has {failures[key]} verified failure{'s' if failures[key] != 1 else ''} on record" if failures.get(key) else "")
                           + ".")
            systems.append("Systems by how many vehicles have them poor or critical:")
            for key, cars in top[:5]:
                worst = sorted(cars, key=lambda v: next(RATING_ORDER[s["rating"]] for s in v["systems"] if s["category_key"] == key))[:3]
                systems.append(f"- {labels.get(key, key)}: {len(cars)} ({len(critical.get(key, []))} critical)"
                               + (f", {failures[key]} verified failure{'s' if failures[key] != 1 else ''}" if failures.get(key) else "")
                               + "; worst: " + ", ".join(v.get("plate_number") or name(v) for v in worst))
            most_failed = max(failures.items(), key=lambda item: item[1]) if failures else None
            if most_failed and most_failed[1] and most_failed[0] != key:
                systems.append(f"By history, {labels.get(most_failed[0], most_failed[0]).lower()} has failed most often "
                               f"({most_failed[1]} verified failures), so keep it on the inspection list even where it's rated well now.")
        else:
            systems.append("Weakest systems across the fleet: "
                           + ", ".join(f"{labels.get(k, k)} ({len(cars)} vehicle{'s' if len(cars) != 1 else ''} poor or critical)" for k, cars in top[:3]) + ".")

    # The trend since the previous assessment.
    worse = [(v, s) for v in vehicles for s in v.get("systems") or [] if _worsened(s)]
    worse.sort(key=lambda pair: (RATING_ORDER.get(pair[1]["rating"], 9), pair[1]["rating_score"] - pair[1]["previous_score"]))
    trend = []
    if focus == "trend":
        cars = {}
        for v, s in worse:
            cars.setdefault(v["id"], (v, []))[1].append(s)
        if not cars:
            trend.append("No system got worse since the previous assessment.")
        else:
            trend.append(f"{len(cars)} vehicle{'s' if len(cars) != 1 else ''} got worse since the previous assessment, worst first:")
            for v, drops in sorted(cars.values(), key=lambda item: min(RATING_ORDER[s["rating"]] for s in item[1]))[:6]:
                trend.append(f"- {_plate_label(v)} (assessed {v['assessed_on']}, before that {v.get('previous_assessed_on') or 'earlier'}): "
                             + ", ".join(f"{s['category'].lower()} {s['previous_rating']} to {s['rating']}" for s in drops) + ".")
            if len(cars) > 6:
                trend.append(f"...and {len(cars) - 6} more.")
            trend.append("A rating that drops between monthly assessments is the earliest warning the records give: inspect those systems "
                         "before they reach poor or critical.")
    elif worse:
        trend.append("Getting worse since the previous assessment: "
                     + "; ".join(f"{_plate_label(v)} {s['category'].lower()} {s['previous_rating']} to {s['rating']}" for v, s in worse[:3])
                     + (f" (and {len(worse) - 3} more)" if len(worse) > 3 else "") + ".")

    history = []
    if failures and focus != "systems" and any(failures.values()):
        system, count = max(failures.items(), key=lambda item: item[1])
        history.append(f"In the verified repair history, {labels.get(system, system).lower()} has the most confirmed failures ({count}).")

    order = {"systems": [systems, at_risk, trend], "trend": [trend, at_risk, systems]}.get(focus, [at_risk, systems, trend])
    for section in order + [history]:
        lines += section
    model = _model_line(fleet_risk, by_tool)
    lines.append(model or CONDITION_SOURCE_NOTE)
    return "\n".join(lines)


def _vehicle_condition(plan, by_tool, vehicles, due):
    records = _rows(by_tool, "maintenance.records", "legacy_service_records") or _rows(by_tool, "maintenance.records", "records")
    ids = {v["id"] for v in vehicles} | set(due)
    if not ids:
        return "No assessment or service record matched that vehicle."
    by_id = {v["id"]: v for v in vehicles}
    blocks = []
    for vehicle_id in sorted(ids, key=lambda i: -_priority_score(by_id.get(i, {}), due.get(i)))[:3]:
        v, d = by_id.get(vehicle_id), due.get(vehicle_id)
        label = _plate_label(v or d)
        lines = []
        if v is None and _condition_missing(by_tool):
            lines.append(f"{label}: {CONDITION_UNAVAILABLE[0].lower() + CONDITION_UNAVAILABLE[1:]}.")
        elif v is None:
            lines.append(f"{label}: no assessment is recorded, so its condition is unknown. Record one under Predictive Maintenance "
                         "(/admin/maintenance/predictions).")
        else:
            km = f" at {v['odometer_km']:,} km" if v.get("odometer_km") is not None else ""
            safety = f", {v['safety_warning_count']} safety warning{'s' if v['safety_warning_count'] != 1 else ''}" if v.get("safety_warning_count") else ""
            lines.append(f"{label}, assessed {v['assessed_on']}{km}: overall {str(v.get('overall_screening') or 'unknown').replace('_', ' ')}{safety}.")
            concerns = _concerns(v)
            findings = v.get("findings") or []
            if concerns:
                lines.append("Needs attention:")
                for s in concerns[:5]:
                    detail = next((f for f in findings if f.get("category_key") == s["category_key"] and f.get("severity") in ("critical", "warning")),
                                  None) or next((f for f in findings if f.get("category_key") == s["category_key"]), None)
                    text = f"- {s['category']}: {_rating_text(s, safety=True)}."
                    if detail and not detail["observation"].startswith("Assessor rated"):
                        text += f" {detail['observation']}"
                    if detail:
                        text += f" Next: {detail['recommended_action']}"
                    lines.append(text)
            else:
                lines.append("No system is rated poor or critical.")
            fine = [s for s in v.get("systems") or [] if s.get("rating") in ("good", "excellent")]
            if fine:
                lines.append("In good shape: " + ", ".join(f"{s['category']} ({s['rating']})" for s in fine) + ".")
            repeats = [(system, (results or {}).get("confirmed_failure")) for system, results in (v.get("outcomes") or {}).items()
                       if (results or {}).get("confirmed_failure")]
            if repeats:
                labels = {s["category_key"]: s["category"] for s in v.get("systems") or []}
                lines.append("Verified failures on record: " + ", ".join(f"{labels.get(k, k)} {n}" for k, n in sorted(repeats, key=lambda x: -x[1])) + ".")
        if d:
            lines.append(f"Service schedule: {STATUS_LABELS.get(d.get('service_status'), d.get('service_status')).lower()}; {_service_reason(d)}.")
        own = [r for r in records if int(r.get("vehicleID") or r.get("vehicle_id") or 0) == vehicle_id][:3]
        if own:
            lines.append("Recent work: " + "; ".join(f"{r.get('maintenanceDate') or r.get('date')} {r.get('maintenanceType') or r.get('type')}"
                                                     for r in own) + ".")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n\n" + CONDITION_SOURCE_NOTE


def _maintenance_priority(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    vehicles = _rows(by_tool, "assessments.condition")
    due = _due_by_id(by_tool)
    by_id = {v["id"]: v for v in vehicles}
    ids = set(by_id) | set(due)
    scored = sorted(((i, _priority_score(by_id.get(i, {}), due.get(i))) for i in ids), key=lambda pair: -pair[1])
    urgent = [(i, score) for i, score in scored if score > 0]
    if not urgent and _condition_missing(by_tool):
        return CONDITION_UNAVAILABLE + ", and no vehicle is overdue on the service schedule."
    if not urgent:
        return ("Nothing stands out: no vehicle has a poor or critical rating, a safety finding or an overdue service. "
                "Keep to the regular schedule and the monthly reassessments.")
    lines = ["Service priority, most urgent first:"]
    for rank, (vehicle_id, _) in enumerate(urgent[:5], 1):
        v, d = by_id.get(vehicle_id), due.get(vehicle_id)
        reasons = []
        safety = _safety_findings(v or {})
        if safety:
            reasons.append(f"safety: {safety[0]['observation'].rstrip('.').lower()}")
        concerns = _concerns(v or {})
        if concerns:
            reasons.append(", ".join(f"{s['category'].lower()} {_rating_text(s)}" for s in concerns[:2] if not safety or s["category_key"] != safety[0].get("category_key"))
                           or f"{concerns[0]['category'].lower()} {_rating_text(concerns[0])}")
        if d and d.get("service_status") in ("overdue", "approaching"):
            reasons.append(f"service {STATUS_LABELS[d['service_status']].lower()} ({_service_reason(d)})")
        action = _first_action(v or {})
        lines.append(f"{rank}. {_plate_label(v or d)}: " + "; ".join(r for r in reasons if r) + "." + (f" Do first: {action}" if action else ""))
    rest = len(ids) - len(urgent)
    if len(urgent) > 5:
        lines.append(f"{len(urgent) - 5} more vehicle{'s' if len(urgent) - 5 != 1 else ''} need attention after these.")
    if rest:
        lines.append(f"{rest} vehicle{'s have' if rest != 1 else ' has'} nothing urgent recorded.")
    lines.append("Order: safety findings first, then critical and poor ratings (and anything getting worse), then overdue service.")
    return "\n".join(lines)


RETIRE_RULE = ("Rule of thumb: retire a car when the next repairs would cost more than about half its resale value, when the same "
               "safety system keeps failing, or when downtime keeps it off the road more than it earns. The system doesn't store "
               "resale values, so get a repair quote and a valuation before deciding.")


def _maintenance_retire(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    vehicles = {v["id"]: v for v in _rows(by_tool, "assessments.condition")}
    due = _due_by_id(by_tool)
    costs_data = (by_tool.get("maintenance.costs") or {}).get("data") or {}
    spend = {int(r["vehicle_id"]): r for r in costs_data.get("per_vehicle") or [] if r.get("vehicle_id") is not None}
    named = bool((plan.get("requested") or {}).get("models") or (plan.get("requested") or {}).get("makes")
                 or any((r.get("args") or {}).get("vehicle_ids") for r in plan.get("retrievals") or []))
    ids = set(vehicles) | set(due) | set(spend)
    if not ids:
        return "No assessment, service or cost record matched that vehicle, so there's nothing to base a decision on.\n\n" + RETIRE_RULE
    all_spend = sorted(float(r.get("total_cost") or 0) for r in spend.values())
    median_spend = all_spend[len(all_spend) // 2] if all_spend else 0.0
    today = datetime.date.today()

    def signals(vehicle_id):
        v, d, c = vehicles.get(vehicle_id, {}), due.get(vehicle_id, {}), spend.get(vehicle_id)
        found, facts = [], []
        year = v.get("year")
        if year:
            age = today.year - int(year)
            facts.append(f"{age} year{'s' if age != 1 else ''} old ({year})")
            if age >= 10:
                found.append("older than 10 years")
        km = v.get("current_mileage") if v.get("current_mileage") is not None else d.get("current_mileage")
        if km is not None:
            facts.append(f"{km:,} km")
            if km >= 200000:
                found.append("over 200,000 km")
        if c:
            plural = "s" if c["records"] != 1 else ""
            if float(c.get("total_cost") or 0) > 0:
                facts.append(f"{money(c['total_cost'])} spent over {c['records']} maintenance record{plural}")
                if median_spend and float(c["total_cost"]) >= 1.5 * median_spend and len(all_spend) >= 3:
                    found.append("maintenance spend well above the fleet's typical car")
            else:
                facts.append(f"{c['records']} maintenance record{plural}, with no costs recorded")
        bad = [s for s in v.get("systems") or [] if s.get("rating") in ("critical", "poor")]
        if bad:
            facts.append(f"{len(bad)} system{'s' if len(bad) != 1 else ''} poor or critical (" + ", ".join(s["category"].lower() for s in bad[:4]) + ")")
            if sum(1 for s in bad if s["rating"] == "critical") >= 2:
                found.append("two or more systems rated critical")
            elif len(bad) >= 3:
                found.append(f"{len(bad)} systems in poor or critical condition at once")
        labels = {s["category_key"]: s["category"] for s in v.get("systems") or []}
        repeats = [labels.get(k, k).lower() for k, r in (v.get("outcomes") or {}).items() if int((r or {}).get("confirmed_failure") or 0) >= 2]
        if repeats:
            found.append("repeated verified failures in " + _join(repeats))
            if len(repeats) >= 3:
                found.append("problems keep coming back across several systems")
        if _safety_findings(v):
            facts.append("open safety findings")
        return found, facts

    def verdict(found):
        if len(found) >= 3:
            return "a retirement candidate"
        if len(found) == 2:
            return "worth reviewing: repair the open items only if the quote is well below its value"
        return "keep it running: nothing points to retirement yet" if not found else "keep it for now, but watch it"

    if named:
        blocks = []
        for vehicle_id in sorted(ids)[:3]:
            found, facts = signals(vehicle_id)
            label = _plate_label(vehicles.get(vehicle_id) or due.get(vehicle_id) or spend.get(vehicle_id) or {"id": vehicle_id})
            blocks.append(f"{label}: {verdict(found).capitalize()}.\n- Facts: " + ("; ".join(facts) if facts else "nothing recorded") + "."
                          + ("\n- Retirement signs: " + "; ".join(found) + "." if found else ""))
        return "\n\n".join(blocks) + "\n\n" + RETIRE_RULE
    candidates = sorted(((i,) + signals(i) for i in ids), key=lambda item: -len(item[1]))
    candidates = [c for c in candidates if c[1]]
    if not candidates:
        return "No vehicle shows signs that point to retirement (age, mileage, spend, critical ratings or repeated failures).\n\n" + RETIRE_RULE
    lines = ["Vehicles with the most signs pointing to retirement:"]
    for vehicle_id, found, facts in candidates[:4]:
        label = _plate_label(vehicles.get(vehicle_id) or due.get(vehicle_id) or spend.get(vehicle_id) or {"id": vehicle_id})
        lines.append(f"- {label}: {verdict(found)}. Signs: {'; '.join(found)}. Facts: {'; '.join(facts)}.")
    if len(candidates) > 4:
        lines.append(f"{len(candidates) - 4} more vehicle{'s show' if len(candidates) - 4 != 1 else ' shows'} one or two signs.")
    return "\n".join(lines) + "\n\n" + RETIRE_RULE


def _maintenance_records(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["maintenance.records"].get("data") or {}
    legacy = data.get("legacy_service_records") or []
    if not legacy:
        return "No maintenance records are stored for the vehicles you asked about."
    lines = ["Latest maintenance records:"]
    for r in legacy:
        cost = r.get("cost")
        lines.append(f"- #{r.get('maintenanceID')} {r.get('maintenanceDate')} {name(r)}: {r.get('maintenanceType')} ({r.get('maintenanceStatus')})"
                     + (f", cost {money(cost)}" if cost not in (None, "") else ", cost not recorded")
                     + (f", odometer {int(r['odometerReading']):,} km" if r.get("odometerReading") not in (None, "") else "")
                     + (f" - {r['description']}" if r.get("description") else ""))
    if notes:
        notes.add(lines, "service_log")
    else:
        lines.append(NOTES["service_log"])
    return "\n".join(lines)


def _maintenance_recurring(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    issues = (by_tool["maintenance.recurring"].get("data") or {}).get("issues") or []
    if not issues:
        return "No service type is recorded more than once for the same vehicle in the latest maintenance records."
    lines = ["Service types recorded more than once for the same vehicle:"]
    lines += [f"- {name(i)}: {i['maintenance_type']} x{i['count']} (latest {i['last_date']})" for i in issues]
    lines.append("Repeats can be routine (such as oil changes) or a recurring problem; check the record details.")
    return "\n".join(lines)


def _maintenance_inspections(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    inspections = ((by_tool.get("maintenance.inspections") or {}).get("data") or {}).get("inspections") or []
    if not inspections:
        return "No component inspections are stored for those vehicles."
    lines = ["Component inspections:"]
    lines += [f"- {i.get('inspected_on')} {name(i)}, {i.get('component') or i.get('component_key')}: condition {i.get('condition_rating')}"
              + (f", DTCs {i['dtc_codes']}" if i.get("dtc_codes") else ", no DTCs recorded")
              + (f", findings: {i['findings']}" if i.get("findings") else "")
              + (f", confirmed fault: {i['confirmed_fault']}" if i.get("confirmed_fault") else "")
              + (f", recommended: {i['recommended_action']}" if i.get("recommended_action") else "") for i in inspections]
    return "\n".join(lines)


def _maintenance_prediction(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    vehicles = ((by_tool.get("maintenance.model_inputs") or {}).get("data") or {}).get("vehicles") or []
    lines = []
    if fleet_risk:
        lines.append(fleet_risk(vehicles))
    if (by_tool.get("maintenance.due") or {}).get("status") == "ok":
        lines.append(_maintenance_due(plan, by_tool, role, vocabulary, guide, fleet_risk, notes))
    return "\n\n".join(line for line in lines if line)


def _fallback(by_tool):
    """Plain listing of retrieved vehicles, used if a composed answer fails validation."""
    for item in by_tool.values():
        for v in (item.get("data") or {}).get("vehicles") or []:
            if "daily_rental_price" in v:
                return "Vehicle records: " + "; ".join(f"{name(v)} {money(v['daily_rental_price'])} per day" for v in (item.get("data") or {}).get("vehicles"))
    return "I retrieved the records but couldn't summarise them reliably. Please check the relevant page in the app."
