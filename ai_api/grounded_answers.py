"""Answers built only from the records the Back-End retrieved for a plan.

Every figure in an answer comes from the retrieval results (prices, deposits and totals
are computed by the Back-End with the reservation pricing rule). validate_amounts()
rejects an answer that mentions a peso amount not present in the results, and the
caller then falls back to a plain listing of the records. When a record is missing or
a tool failed, the answer says exactly what could not be verified.
"""
import re


ADMIN_ONLY = ("That information is limited to administrators, so I can't show it for this account. "
              "I can help with vehicle listings, prices, availability and your own reservations.")
DATA_ERROR = ("I couldn't read {what} from the database just now, so I can't give a verified answer. "
              "Please try again in a moment.")
AMOUNT = re.compile(r"PHP\s?([\d,]+(?:\.\d{1,2})?)")
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
    "booked_value": "These are stored reservation totals; the system records no payments, so they are not money received.",
    "no_payments": "The system records reservation totals but no payments, so this is the booked amount, not money paid.",
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
    }.get(intent)
    if handler is None:
        return "I couldn't work out which records answer that. Could you rephrase it?", data
    notes = Notes(shown_notes)
    answer = handler(plan, by_tool, role, vocabulary or {}, guide, fleet_risk, notes)
    data["notes_shown"] = notes.added
    try:
        validate_amounts(answer, results)
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

    if plan.get("follow_up") and vehicles:
        lines.append(f"(About the {_join(sorted({name(v) for v in vehicles}))} from earlier.)")
    for unknown in requested.get("unknown_names") or []:
        lines.append(f"I couldn't find a {unknown} {scope}.")

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

    if purpose == "feature_check":
        lines += _feature_lines(plan.get("feature_check") or {}, vehicles, label)
    elif purpose == "compare":
        lines += _compare_lines(plan.get("compare") or {}, vehicles, label)
    elif purpose == "deposit":
        if len(vehicles) == 1:
            v = vehicles[0]
            lines.append(f"The {label[id(v)]} has a {money(v['security_deposit'])} deposit (daily rate {money(v['daily_rental_price'])})."
                         if v.get("security_deposit") else f"No security deposit is listed for the {label[id(v)]}.")
        else:
            lines.append("Security deposits from the vehicle records:")
            lines += [f"- {label[id(v)]}: {money(v['security_deposit'])} deposit (daily rate {money(v['daily_rental_price'])})"
                      if v.get("security_deposit") else f"- {label[id(v)]}: no security deposit is listed" for v in vehicles]
        notes.add(lines, "deposit_rule")
    elif purpose in {"quote", "availability"} and vehicles[0].get("quote"):
        if purpose == "availability" and data.get("period"):
            period = data["period"]
            free = [label[id(v)] for v in vehicles if v.get("available_for_period")]
            taken = [label[id(v)] for v in vehicles if v.get("available_for_period") is False]
            dates = f"{period['start_date']} to {period['end_date']}"
            if free:
                lines.append(f"Yes: {_join(free)} {'is' if len(free) == 1 else 'are'} free from {dates}.")
            if taken:
                lines.append(f"{_join(taken)} {'is' if len(taken) == 1 else 'are'} already booked for {dates}.")
        lines += _quote_lines(vehicles, data, label)
        notes.add(lines, "quote_rule")
    elif purpose == "availability" or purpose == "status":
        lines += _availability_lines(vehicles, data, role, purpose, label)
        if role != "admin" and not data.get("period"):
            notes.add(lines, "dates_hint")
    elif purpose == "details":
        if len(vehicles) == 1:
            lines.append(f"The {label[id(vehicles[0])]}: {_specs(vehicles[0])}.")
        else:
            heading = "Vehicles that seat at least {} people".format(args["min_seats"]) if args.get("min_seats") else None
            if heading or _feature_words(args):
                lines.append(f"{heading or _feature_words(args).capitalize() + ' vehicles'} {scope}:")
            lines += [f"- {label[id(v)]}: {_specs(v)}" for v in vehicles]
    else:
        lines += _price_lines(vehicles, args, label, role, scope)
        if purpose == "price" and len(vehicles) > 0:
            notes.add(lines, "price_rule")
    if data.get("truncated"):
        lines.append(f"Showing {len(vehicles)} of {data.get('total_matching')} matching vehicles.")
    return "\n".join(lines)


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
        lines.append(f"The {label[id(v)]} is {money(v['daily_rental_price'])} per day, with a {money(v['security_deposit'])} deposit.")
        return lines
    features = _feature_words(args)
    lines.append(f"{features.capitalize()} vehicles {scope}:" if features else ("Vehicles in the fleet:" if role == "admin" else "Daily rates:"))
    lines += [f"- {label[id(v)]} ({v.get('category') or 'vehicle'}, {v.get('seating_capacity')} seats): "
              f"{money(v['daily_rental_price'])} per day, {money(v['security_deposit'])} deposit"
              + (f", status {v['status']}" if role == "admin" and v.get("status") else "") for v in vehicles]
    return lines


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


def _feature_lines(check, vehicles, label):
    feature, value = check.get("feature"), check.get("value")
    lines = []
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
        if v.get("available_for_period") is False:
            text += "; it is already booked for those dates"
        elif v.get("available_for_period") is True:
            text += "; it is free for those dates"
        return text

    if len(vehicles) == 1:
        v = vehicles[0]
        quote = v["quote"]
        return [f"Renting the {label[id(v)]} for {days} day{plural}{when} costs {money(quote['base_total'])} "
                f"({money(quote['daily_rate'])} x {days} day{plural}), plus a {money(quote['security_deposit'])} "
                f"security deposit shown separately{extras(v, quote)}."]
    lines = [f"Estimate for {days} day{plural}{when}:"]
    for v in vehicles:
        quote = v["quote"]
        lines.append(f"- {label[id(v)]}: {money(quote['base_total'])} ({money(quote['daily_rate'])} x {days} day{plural}), "
                     f"plus a {money(quote['security_deposit'])} security deposit shown separately{extras(v, quote)}")
    return lines


def _availability_lines(vehicles, data, role, purpose, label):
    period = data.get("period")
    if period:
        free = [label[id(v)] for v in vehicles if v.get("available_for_period")]
        taken = [label[id(v)] for v in vehicles if v.get("available_for_period") is False]
        lines = [f"For {period['start_date']} to {period['end_date']} ({period['rental_days']} days, return date not counted):"]
        lines.append("Available: " + (_join(free) if free else "none of the matching vehicles") + ".")
        if taken:
            lines.append("Already booked or not rentable then: " + _join(taken) + ".")
        return lines
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
        lines = ["Available now: " + (_join(f"{label[id(v)]} ({money(v['daily_rental_price'])}/day)" for v in available) if available else "none") + "."]
        if others:
            lines.append("Not available now: " + ", ".join(f"{count} {status}" for status, count in sorted(others.items()))
                         + " (ask which cars are rented or in maintenance for their names).")
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
    elif purpose in ("status", "details"):
        # Lead with the reservation the question is most likely about: an active or upcoming one first.
        focus = [r for r in rows if r.get("is_active") or r.get("days_until_pickup") is not None] or rows[:1]
        for r in focus[:3]:
            lines.append(f"Your reservation #{r['id']} for the {name(r)} ({r['start_date']} to {r['end_date']}) is {r['status']}.")
        if purpose == "details":
            lines += [f"- {one(r)}" for r in focus[:1]]
    elif purpose == "next":
        r = next((row for row in sorted(rows, key=lambda row: row["start_date"]) if row.get("days_until_pickup") is not None or row.get("is_active")), None)
        if r is None:
            lines.append("You have no upcoming pickup.")
        elif r.get("is_active"):
            lines.append(f"Your rental #{r['id']} ({name(r)}) is in progress; the return is due on {r['end_date']}.")
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
        if totals:
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
        lines.append(("Your upcoming reservations:" if upcoming else "Your reservations:"))
        lines += [f"- {one(r)}" for r in rows]
        if data.get("total_count", len(rows)) > len(rows):
            lines.append(f"Showing the latest {len(rows)} of {data['total_count']}.")
    if rows:
        lines.append("Full details are on My Reservations (/users/reservations).")
    return "\n".join(lines)


def _reservation_summary(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["reservations.summary"].get("data") or {}
    rows = data.get("reservations") or []
    counts = data.get("counts_by_status") or {}
    args = (plan.get("retrievals") or [{}])[0].get("args") or {}
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


def _maintenance_costs(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["maintenance.costs"].get("data") or {}
    window = f" from {data['from'] or 'the first record'} to {data['to'] or 'today'}" if data.get("from") or data.get("to") else ""
    if not data.get("records"):
        return f"No maintenance records with costs are stored{window}."
    lines = [f"Total recorded maintenance cost{window}: {money(data['total_cost'])} across {data['records']} record(s)."]
    per_vehicle = data.get("per_vehicle") or []
    if len(per_vehicle) > 1:
        top = max(per_vehicle, key=lambda v: float(v["total_cost"]))
        lines.insert(0, f"The highest recorded maintenance cost is the {name(top)} at {money(top['total_cost'])}.")
    lines += [f"- {name(v)}" + (f" ({v['plate_number']})" if v.get("plate_number") else "")
              + f": {money(v['total_cost'])} ({v['records']} record(s))" for v in data.get("per_vehicle") or []]
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
    lines = [f"Booked value{window}: {money(data.get('booked_total', 0))} (pending, confirmed and completed reservations)."]
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
    return (f"The fleet has {data.get('total', 0)} vehicle(s): "
            + (", ".join(f"{status} {count}" for status, count in sorted(counts.items())) if counts else "no status counts recorded") + ".")


def _maintenance_due(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["maintenance.due"].get("data") or {}
    rows = data.get("vehicles") or []
    if not rows:
        return "No vehicles matched, so there is no service status to report."
    lines = ["Service status from recorded maintenance history (" + data.get("rule", "rule-based estimate") + "):"]
    order = ["overdue", "approaching", "insufficient_history", "not_due"]
    for status in order:
        group = [r for r in rows if r.get("service_status") == status]
        if not group:
            continue
        lines.append(f"{STATUS_LABELS[status]}:")
        for r in group:
            detail = []
            if r.get("last_service_date"):
                detail.append(f"last service {r['last_service_date']}" + (f" ({r['last_service_type']})" if r.get("last_service_type") else ""))
            if r.get("current_mileage") is not None:
                detail.append(f"{r['current_mileage']:,} km now")
            if r.get("estimated_due_date") and status != "insufficient_history":
                detail.append(f"date-based due {r['estimated_due_date']}")
            lines.append(f"- {name(r)}" + (f" ({r['plate_number']})" if r.get("plate_number") else "") + ": " + "; ".join(detail) + ".")
    if notes:
        notes.add(lines, "schedule_estimate")
    else:
        lines.append(NOTES["schedule_estimate"])
    return "\n".join(lines)


def _maintenance_records(plan, by_tool, role, vocabulary, guide, fleet_risk, notes=None):
    data = by_tool["maintenance.records"].get("data") or {}
    legacy = data.get("legacy_service_records") or []
    verified = data.get("verified_evidence_records") or []
    if not legacy and not verified:
        return "No maintenance records are stored for the vehicles you asked about."
    lines = ["Latest maintenance records:"]
    for r in legacy:
        cost = r.get("cost")
        lines.append(f"- #{r.get('maintenanceID')} {r.get('maintenanceDate')} {name(r)}: {r.get('maintenanceType')} ({r.get('maintenanceStatus')})"
                     + (f", cost {money(cost)}" if cost not in (None, "") else ", cost not recorded")
                     + (f", odometer {int(r['odometerReading']):,} km" if r.get("odometerReading") not in (None, "") else "")
                     + (f" - {r['description']}" if r.get("description") else ""))
    for r in verified:
        lines.append(f"- #{r.get('id')} {r.get('recorded_at')} {name(r)} (verified {r.get('record_type')}): {r.get('component')}; "
                     f"cause: {r.get('confirmed_cause') or 'not recorded'}; repair: {r.get('repair_action') or 'not recorded'}"
                     + (f"; DTCs {r['dtc_codes']}" if r.get("dtc_codes") else ""))
    if data.get("verified_evidence_table_available") is False:
        lines.append("The verified maintenance-evidence table is not installed yet, so only service-log entries are shown.")
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
    verified = ((by_tool.get("maintenance.records") or {}).get("data") or {}).get("verified_evidence_records") or []
    if not inspections and not verified:
        return "No component inspections or verified diagnostic records are stored for those vehicles."
    lines = []
    if inspections:
        lines.append("Component inspections:")
        lines += [f"- {i.get('inspected_on')} {name(i)}, {i.get('component') or i.get('component_key')}: condition {i.get('condition_rating')}"
                  + (f", DTCs {i['dtc_codes']}" if i.get("dtc_codes") else ", no DTCs recorded")
                  + (f", findings: {i['findings']}" if i.get("findings") else "")
                  + (f", confirmed fault: {i['confirmed_fault']}" if i.get("confirmed_fault") else "")
                  + (f", recommended: {i['recommended_action']}" if i.get("recommended_action") else "") for i in inspections]
    coded = [r for r in verified if r.get("dtc_codes")]
    if coded:
        lines.append("Verified repair records with trouble codes:")
        lines += [f"- #{r.get('id')} {r.get('recorded_at')} {name(r)}: {r['dtc_codes']}; component {r.get('component') or 'not recorded'}; "
                  f"cause: {r.get('confirmed_cause') or 'not recorded'}; work done: {r.get('repair_action') or 'not recorded'}"
                  + (f"; outcome: {r['repair_outcome']}" if r.get("repair_outcome") else "") for r in coded]
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
