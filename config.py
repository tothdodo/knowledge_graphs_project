"""Shared constants"""

import re

RAW_CSV = "data/Players_Stats_Per_game.csv" # Raw 25-26 NBA season player stats
BASE_GRAPH_CSV = "data/base_graph.csv" # Facts from raw csv
DERIVED_CSV = "data/derived_triples.csv" # Datalog rules applied to base graph
KG_CSV = "data/final_knowledge_graph.csv" # Sum of base + derived triples
RULES_FILE = "rules/playstyle_rules.dl" # Datalog rules for deriving archetypes, tiers, position groups
ARTIFACTS_DIR = "artifacts" # Directory for the trained TransE model, relation ids, and metadata

# Filter out players with too few games or minutes
MIN_GAMES = 15
MIN_MINUTES = 10.0

# Predicates that describe the player playstyle
PLAYSTYLE_PREDICATES = [
    "has_Pos",
    "has_position_group",
    "has_archetype",
    "has_archetype_family",
    "has_scoring_tier",
    "has_playmaking_tier",
    "has_rebounding_tier",
    "has_three_point_tier",
    "has_defensive_tier",
    "has_role_tier",
]
TEAM_PREDICATE = "has_Team"

# Predicates used to explain a recommendation
EXPLAIN_PREDICATES = [p for p in PLAYSTYLE_PREDICATES if p != "has_Pos"]

_NUM = re.compile(r"^-?\d+(\.\d+)?$")

# CSV field num parser (if parsing is possiuble)
def parse_object(value):
    s = str(value)
    if _NUM.match(s):
        return float(s) if "." in s else int(s)
    return s
