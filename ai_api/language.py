"""Message normalisation before intent matching and retrieval.

Maps common misspellings and Tagalog/Taglish rental phrases to the English terms the
classifier, routing rules and guide keywords use. Only whole words in these lists are
replaced; everything else (vehicle names, DTC codes, dates, numbers, prices) is kept
as typed. Back-End/app/libraries/ChatIntentRouter.php keeps the matching list for
routing database questions.
"""
import re


_RULES = [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in (
    # Misspellings
    (r"\b(?:res[ae]?rv|reserb|resrve|reseve)(?:e|ed|ing)?\b", "reserve"),
    (r"\b(?:res[ae]?rv|resev|reserb)a(?:t|s|c|sh)i?on(s?)\b", r"reservation\1"),
    (r"\b(?:vehical|vehicel|vehicule|vechile|vehicl|vehilce|vihicle|veicle|vehcile)(s?)\b", r"vehicle\1"),
    (r"\b(?:bookng|boking|bokking|bookin)(s?)\b", r"booking\1"),
    (r"\b(?:availble|avilable|avaliable|availabe|avalable)\b", "available"),
    (r"\b(?:prise|pirce|prce)(s?)\b", r"price\1"),
    (r"\b(?:deposite|depost|deposti|diposit)(s?)\b", r"deposit\1"),
    (r"\b(?:lisence|liscense|licens|lisensya)\b", "license"),
    (r"\b(?:requirment|requirments|reqs|rekwayrments?)\b", "requirements"),
    (r"\b(?:cancle|cancell)\b", "cancel"),
    # Tagalog / Taglish
    (r"\b(?:magkano|mgkano|magkanu|hm)\b", "how much"),
    (r"\b(?:presyo|halaga)\b", "price"),
    (r"\bdeposito\b", "deposit"),
    (r"\b(?:sasakyan|kotse)\b", "car"),
    (r"\b(?:kada|bawat)\s+araw\b", "per day"),
    (r"\bisang\s+araw\b", "1 day"),
    (r"\b(\d+)\s*na\s+araw\b", r"\1 days"),
    (r"\baraw\b", "days"),
    (r"\b(?:isang\s+linggo|one\s+week|a\s+week|1\s+week)\b", "7 days"),
    (r"\b(?:magbayad|makabayad|pagbayad|bayaran|nagbayad|bayad)\b", "pay"),
    (r"\b(?:pwede|puwede|pede)\s+(?:ba|po)?\s*", "can i "),
    (r"\bpag\s+(?=\d)", "for "),
    (r"\b(?:paano|pano|papaano)\b", "how do i"),
    (r"\b(?:makapag|mag|magpa)[- ]?(book|reserve|rent)\b", r"\1"),
    (r"\b(?:magbook|magpabook)\b", "book"),
    (r"\b(?:magrenta|umupa|arkila|arkilahin|renta|rentahan)\b", "rent"),
    (r"\b(?:makita|makikita|tingnan|titingnan|pakita|ipakita|patingin)\b", "see"),
    # Car problems in Tagalog / Taglish
    (r"\b(?:kapag|pag|kung)\s+(?:nag-?preno|nagpepreno|nagbe-?brake|nagbrake)\b", "when braking"),
    (r"\b(?:nag-?preno|nagpepreno)\b", "braking"),
    (r"\b(?:ayaw|hindi)\s+(?:umandar|mag-?start|magstart|mag-?andar)\b", "won't start"),
    (r"\b(?:flat|butas|pumutok)\s+(?:ang\s+)?gulong\b", "flat tire"),
    (r"\b(?:sa\s+)?(?:mataas na takbo|mabilis na takbo|high speed)\b", "at high speed"),
    (r"\b(?:may\s+)?(?:tulo|tumulo|tumutulo|natutulo|tagas|tumatagas)\b", "leaking"),
    (r"\b(?:usok|umuusok|mausok|nag-?uusok)\b", "smoke"),
    (r"\b(?:ingay|maingay|umiingay)\b", "noise"),
    (r"\b(?:kumakalampag|kalampag|kumakalansing)\b", "rattling noise"),
    (r"\b(?:umuuga|nanginginig|yumuyugyog|uga)\b", "shaking"),
    (r"\b(?:kumakabig|kabig|humihila)\b", "pulls"),
    (r"\bpakaliwa\b", "to the left"),
    (r"\bpakanan\b", "to the right"),
    (r"\bmanibela\b", "steering wheel"),
    (r"\bpreno\b", "brakes"),
    (r"\bgulong\b", "tire"),
    (r"\bmakina\b", "engine"),
    (r"\bbaterya\b", "battery"),
    (r"\blangis\b", "oil"),
    (r"\btubig\b", "water"),
    (r"\bilalim\b", "under"),
    (r"\b(?:mainit|umiinit|nag-?iinit|nag-?ooverheat|init)\b", "overheating"),
    (r"\b(?:sira|nasira|sirang)\b", "broken"),
    (r"\b(?:ayusin|pagawa|ipagawa|ayos)\b", "fix"),
    (r"\bmga\s+", ""),
    (r"\b(bookings?|reservations?|rentals?|rental history)\s+(?:ko|namin)\b", r"my \1"),
    (r"\bkailangan\b", "need"),
    (r"\bano\b", "what"),
    (r"\bpara\s+(?=rent|book|reserve)", "to "),
    (r"\bpo\b", ""),
    (r"\b(?:ang|ng)\b", "the"),
)]


def normalize_message(message):
    text = message
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip() or message
