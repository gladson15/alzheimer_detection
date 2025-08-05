import random

def estimate_age_from_stage(stage_label):
    age_ranges = {
        "No Impairment": (55, 65),
        "Very Mild Impairment": (65, 75),
        "Mild Impairment": (70, 80),
        "Moderate Impairment": (75, 90)
    }
    age_min, age_max = age_ranges.get(stage_label, (60, 80))
    return random.randint(age_min, age_max)