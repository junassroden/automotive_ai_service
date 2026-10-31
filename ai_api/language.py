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
    (r"\bilan\s+(?:ang\s+)?(?:mga\s+)?(?:sasakyan|kotse|cars?|vehicles?|units?)\b", "how many cars"),
    (r"\bilan\b", "how many"),
    # Text-speak
    (r"\bur\b", "your"),
    (r"\bu\b(?!-)", "you"),
    (r"\br\b(?=\s+(?:available|there|you|the|they|these|those|any)\b)", "are"),
    (r"\btemp\b", "temperature"),
    (r"\bkailan\b", "when"),
    (r"\b(?:kukunin|kunin|kukuha|makukuha)\b", "pick up"),
    (r"\b(?:ibabalik|isasauli|isauli|ibalik)\b", "return"),
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
    (r"\b(?:won'?t|wont|doesn'?t|does not|will not|can'?t|cannot|ayaw)\s+(?:turn|switch|mag)\s*on\b", "won't start"),
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


# Words the assistant routes on. A typed word one keyboard slip away from exactly one of them
# ("depoist", "aricon", "milaege", "novmeber") is read as that word; see correct_typos().
DOMAIN_WORDS = set("""
price prices pricing rate rates cost costs deposit deposits security total quote budget cheap cheapest expensive afford affordable
daily available availability rent rental rentals renting reserve reservation reservations booking bookings booked book
cancel cancellation cancelled refund refundable payment payments online credit debit gcash cash discount discounts promo promos
requirements requirement license licence documents driver drivers chauffeur delivery deliver insurance smoking pets
pending confirmed completed rejected status approved extend extension password account register signup schedule
vehicle vehicles sedan seater seaters seats seating capacity passengers people persons transmission automatic manual
navigation aircon conditioning color colour mileage odometer engine brakes braking brake squeaking squealing grinding
overheating overheat temperature coolant battery alternator exhaust smoke tires tire tyres puncture steering vibration
leaking leak warning flashing blinking misfire starter maintenance service inspection revenue utilization fleet
monday tuesday wednesday thursday friday saturday sunday january february march april june july august september october
november december tomorrow weekend
toyota honda mitsubishi suzuki nissan hyundai isuzu mazda subaru chevrolet volkswagen ford
vios fortuner hiace innova montero ertiga navara civic accent ranger everest hilux mirage sorento
magkano presyo sasakyan pickup return thanks salamat hello please goodbye rented conditioned airconditioned excuse number
magbayad bayad kotse nagastos gastos promo promos fleet odometer kilometers diesel petrol
brother sister friend friends husband allowed approval approved papers signup register question magandang charge charges gearbox
change changes changed changing chance charged charging charger
""".split())
# Ordinary words that are one slip away from a domain word but mean something else.
_NOT_TYPOS = {"break", "breaks", "broke", "dates", "carry", "where", "there", "these", "those", "other", "about",
              "house", "renter", "pricey", "lease", "leave", "later", "sellers", "covers", "brand", "ranter",
              "router", "statue", "colors", "colours", "models", "today", "renters", "member", "numbers"}


def _one_slip(word, target):
    """True if word is target with one swapped adjacent pair, or (for longer words) one letter added, missing or changed."""
    if word == target:
        return False
    if len(word) == len(target):
        diff = [i for i in range(len(word)) if word[i] != target[i]]
        if len(diff) == 2 and diff[1] == diff[0] + 1 and word[diff[0]] == target[diff[1]] and word[diff[1]] == target[diff[0]]:
            return True
        return len(diff) == 1 and len(word) >= 6
    if abs(len(word) - len(target)) == 1 and min(len(word), len(target)) >= 6:
        short, long_ = (word, target) if len(word) < len(target) else (target, word)
        return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))
    return False


def correct_typos(text):
    def fix(match):
        word = match.group(0)
        low = word.casefold()
        if len(low) < 4 or low in DOMAIN_WORDS or low in _NOT_TYPOS:
            return word
        candidates = [target for target in DOMAIN_WORDS if abs(len(target) - len(low)) <= 1 and _one_slip(low, target)]
        return candidates[0] if len(candidates) == 1 else word
    return re.sub(r"[A-Za-z]+", fix, text)


def normalize_message(message):
    text = correct_typos(message)
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip() or message
