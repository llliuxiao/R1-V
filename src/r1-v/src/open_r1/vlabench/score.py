import networkx as nx
import json
import matplotlib.pyplot as plt
from collections import Counter
import re

# predefined subtask patterns
SUBTASK_PATTERN = [
    ["pick", "place"],
    ["pick", "insert"],
    ["pick", "pour"],
    ["pick", "pull"],
    ["pick", "lift"],
    ["pick", "push", "pull"],
    ["pick", "open_door"],
    ["press"],
]
# PATTERN = r"<think>(.*)?</think>\s*<answer>(.*)?</answer>"
# PATTERN = r"```json(.*)?```"


def calculate_skill_and_entity_scores(sequence1, sequence2):
    skills1 = [skill["name"] for skill in sequence1]
    skills2 = [skill["name"] for skill in sequence2]

    entities1 = []
    entities2 = []

    for skill in sequence1:
        if skill["params"].get("target_entity_name") is not None:
            entities1.append(
                ("target_entity", skill["params"].get("target_entity_name"))
            )
        if skill["params"].get("target_container_name") is not None:
            entities1.append(
                ("target_container", skill["params"].get("target_container_name"))
            )

    for skill in sequence2:
        if skill["params"].get("target_entity_name") is not None:
            entities2.append(
                ("target_entity", skill["params"].get("target_entity_name"))
            )
        if skill["params"].get("target_container_name") is not None:
            entities2.append(
                ("target_container", skill["params"].get("target_container_name"))
            )

    skills1_counter = Counter(skills1)
    skills2_counter = Counter(skills2)
    skill_match_count = sum((skills1_counter & skills2_counter).values())

    entities1_counter = Counter(entities1)
    entities2_counter = Counter(entities2)
    entity_match_count = sum((entities1_counter & entities2_counter).values())

    total_skills = len(skills1)
    total_entities = len(entities1)
    skill_match_score = skill_match_count / total_skills if total_skills > 0 else 0
    entity_match_score = (
        entity_match_count / total_entities if total_entities > 0 else 0
    )

    return {
        "skill_match_score": skill_match_score,
        "entity_match_score": entity_match_score,
    }


def get_format_score(standard_skill_sequences, model_skill_sequences):
    model_skill_sequences = [
        completion[0]["content"] for completion in model_skill_sequences
    ]
    rewards = []
    for model_skill_sequence in model_skill_sequences:
        try:
            answer_content = model_skill_sequence.split("```json")[1].split("```")[0]
            json.loads(answer_content)
            rewards.append(1.0)
        except:
            rewards.append(0.0)
            continue
    return rewards


def get_accurancy_score(standard_skill_sequences, model_skill_sequences):
    model_skill_sequences = [
        completion[0]["content"] for completion in model_skill_sequences
    ]
    rewards = []
    assert len(standard_skill_sequences) == 1
    standard_skill_sequence = standard_skill_sequences[0]
    for model_skill_sequence in model_skill_sequences:
        try:
            answer_content = model_skill_sequence.split("```json")[1].split("```")[0]
            standard_skill_sequence = json.loads(standard_skill_sequence)[
                "skill_sequence"
            ]
            model_skill_sequence = json.loads(answer_content)
            skill_entity_scores = calculate_skill_and_entity_scores(
                standard_skill_sequence, model_skill_sequence
            )
        except:
            rewards.append(0.0)
            continue

        score_weight = {
            "skill_match_score": 0.5,
            "entity_match_score": 0.5,
        }
        total_score = sum(
            score_weight[key] * value for key, value in skill_entity_scores.items()
        )
        rewards.append(total_score)
    return rewards
