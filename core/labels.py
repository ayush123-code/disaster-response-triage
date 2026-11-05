CATEGORIES = [
    "requests_or_urgent_needs",
    "injured_or_dead_people",
    "missing_or_found_people",
    "displaced_people_and_evacuations",
    "infrastructure_and_utility_damage",
    "caution_and_advice",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "sympathy_and_support",
    "not_humanitarian",
]

DISPLAY = {
    "requests_or_urgent_needs": "Request / urgent need",
    "injured_or_dead_people": "Injured or dead people",
    "missing_or_found_people": "Missing or found people",
    "displaced_people_and_evacuations": "Displaced / evacuations",
    "infrastructure_and_utility_damage": "Infrastructure damage",
    "caution_and_advice": "Caution and advice",
    "rescue_volunteering_or_donation_effort": "Volunteering / donations",
    "other_relevant_information": "Other relevant info",
    "sympathy_and_support": "Sympathy and prayers",
    "not_humanitarian": "Not humanitarian",
}

# Categories where a person may need help right now. Used both by the priority
# rule and as ground truth ("truly needs help") when evaluating rankings.
URGENT = {"requests_or_urgent_needs", "injured_or_dead_people", "missing_or_found_people"}

LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}
