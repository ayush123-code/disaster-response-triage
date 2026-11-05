from core.labels import DISPLAY, URGENT

LEVELS = ["Critical", "High", "Medium", "Low"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
DISTRESS_THRESHOLD = 0.6
MEDIUM = {"infrastructure_and_utility_damage", "caution_and_advice"}


def priority(category: str, cat_conf: float, p_urgent: float, distress: float, top_emotion: str):
    """Return (level, score, reason). Score orders posts within and across levels."""
    distressed = distress >= DISTRESS_THRESHOLD
    if category in URGENT:
        level = "Critical" if distressed else "High"
    elif category == "displaced_people_and_evacuations":
        level = "High" if distressed else "Medium"
    elif category in MEDIUM:
        level = "Medium"
    else:
        level = "Low"

    score = 0.75 * p_urgent + 0.25 * distress
    reason = (f"{DISPLAY[category]} ({cat_conf:.0%})"
              f" + {'high' if distressed else 'low'} distress {distress:.2f} ({top_emotion})"
              f" -> {level}")
    return level, score, reason
