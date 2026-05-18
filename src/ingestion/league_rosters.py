"""Team rosters per T20 league for identifying markets in mixed cricket archives.

A market belongs to a league iff BOTH runner names are in that league's set.
Sets include both current and historical names (rebrands, replacements).
"""
from __future__ import annotations

LEAGUE_TEAMS: dict[str, frozenset[str]] = {
    "ipl": frozenset({
        "Chennai Super Kings", "Mumbai Indians",
        "Royal Challengers Bangalore", "Royal Challengers Bengaluru",
        "Kolkata Knight Riders", "Delhi Capitals", "Delhi Daredevils",
        "Sunrisers Hyderabad", "Punjab Kings", "Kings XI Punjab",
        "Rajasthan Royals", "Lucknow Super Giants", "Gujarat Titans",
        "Rising Pune Supergiant", "Rising Pune Supergiants",
        "Gujarat Lions", "Pune Warriors", "Pune Warriors India",
        "Kochi Tuskers Kerala", "Deccan Chargers",
    }),
    "bbl": frozenset({
        "Adelaide Strikers", "Brisbane Heat", "Hobart Hurricanes",
        "Melbourne Renegades", "Melbourne Stars", "Perth Scorchers",
        "Sydney Sixers", "Sydney Thunder",
    }),
    "psl": frozenset({
        "Islamabad United", "Karachi Kings", "Lahore Qalandars",
        "Multan Sultans", "Peshawar Zalmi", "Quetta Gladiators",
    }),
    "cpl": frozenset({
        "Barbados Royals", "Barbados Tridents",
        "Guyana Amazon Warriors",
        "Jamaica Tallawahs",
        "Saint Kitts and Nevis Patriots", "St Kitts and Nevis Patriots",
        "Saint Lucia Kings", "St Lucia Kings", "Saint Lucia Zouks", "St Lucia Zouks",
        "Saint Lucia Stars", "St Lucia Stars",
        "Trinbago Knight Riders", "Trinidad and Tobago Red Steel",
        "Antigua and Barbuda Falcons",
    }),
    "ntb": frozenset({
        "Birmingham Bears", "Warwickshire",
        "Derbyshire", "Derbyshire Falcons",
        "Durham", "Durham Jets",
        "Essex", "Essex Eagles",
        "Glamorgan",
        "Gloucestershire",
        "Hampshire", "Hampshire Hawks",
        "Kent", "Kent Spitfires",
        "Lancashire", "Lancashire Lightning",
        "Leicestershire", "Leicestershire Foxes",
        "Middlesex",
        "Northamptonshire", "Northamptonshire Steelbacks",
        "Nottinghamshire", "Notts Outlaws", "Nottinghamshire Outlaws",
        "Somerset",
        "Surrey",
        "Sussex", "Sussex Sharks",
        "Worcestershire", "Worcestershire Rapids",
        "Yorkshire", "Yorkshire Vikings",
    }),
    "sa20": frozenset({
        "Durban Super Giants", "Durban's Super Giants",
        "Joburg Super Kings", "Johannesburg Super Kings",
        "MI Cape Town",
        "Paarl Royals",
        "Pretoria Capitals",
        "Sunrisers Eastern Cape",
    }),
    "ilt20": frozenset({
        "Abu Dhabi Knight Riders", "Desert Vipers", "Dubai Capitals",
        "Gulf Giants", "MI Emirates", "Sharjah Warriorz", "Sharjah Warriors",
    }),
    "mlc": frozenset({
        "Los Angeles Knight Riders", "LA Knight Riders",
        "MI New York", "San Francisco Unicorns", "Seattle Orcas",
        "Texas Super Kings", "Washington Freedom",
    }),
    "lpl": frozenset({
        "Colombo Stars", "Colombo Strikers",
        "Dambulla Aura", "Dambulla Giants", "Dambulla Thunders",
        "Galle Gladiators", "Galle Marvels", "Galle Titans",
        "Jaffna Kings", "Jaffna Stallions",
        "Kandy Falcons", "Kandy Tuskers", "B-Love Kandy",
    }),
}


def detect_league(runner_names: list[str]) -> str | None:
    """Return the league code if both runners belong to one league, else None."""
    if len(runner_names) != 2:
        return None
    names = set(runner_names)
    for league, teams in LEAGUE_TEAMS.items():
        if names.issubset(teams):
            return league
    return None
