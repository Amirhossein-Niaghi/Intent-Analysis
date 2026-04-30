"""
Simple counterfactual generator (template-based).
Semantic label preserved, surface form altered.
"""

import random


def generate_counterfactual(text: str) -> str:
    rules = [
        lambda t: t.replace("please", ""),
        lambda t: t.replace("can you", "could you"),
        lambda t: t.replace("I want", "I'd like"),
    ]

    rule = random.choice(rules)
    return rule(text)
