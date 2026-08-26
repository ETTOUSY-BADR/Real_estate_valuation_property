"""French territorial identifiers used by the nationwide Phase 4 pipeline."""

from __future__ import annotations

import re


# Official region groupings are intentionally embedded so preprocessing does not
# depend on a mutable lookup service.  The mapping includes metropolitan France
# and the five overseas departments.  DVF currently has no open observations for
# 57, 67, 68 or 976; keeping them here makes that absence auditable.
REGION_DEPARTMENTS: dict[str, tuple[str, ...]] = {
    "11": ("75", "77", "78", "91", "92", "93", "94", "95"),
    "24": ("18", "28", "36", "37", "41", "45"),
    "27": ("21", "25", "39", "58", "70", "71", "89", "90"),
    "28": ("14", "27", "50", "61", "76"),
    "32": ("02", "59", "60", "62", "80"),
    "44": ("08", "10", "51", "52", "54", "55", "57", "67", "68", "88"),
    "52": ("44", "49", "53", "72", "85"),
    "53": ("22", "29", "35", "56"),
    "75": ("16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87"),
    "76": ("09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"),
    "84": ("01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74"),
    "93": ("04", "05", "06", "13", "83", "84"),
    "94": ("2A", "2B"),
    "01": ("971",),
    "02": ("972",),
    "03": ("973",),
    "04": ("974",),
    "06": ("976",),
}

DEPARTMENT_TO_REGION = {
    department: region
    for region, departments in REGION_DEPARTMENTS.items()
    for department in departments
}

REGION_NAMES = {
    "01": "Guadeloupe",
    "02": "Martinique",
    "03": "Guyane",
    "04": "La Réunion",
    "06": "Mayotte",
    "11": "Île-de-France",
    "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comté",
    "28": "Normandie",
    "32": "Hauts-de-France",
    "44": "Grand Est",
    "52": "Pays de la Loire",
    "53": "Bretagne",
    "75": "Nouvelle-Aquitaine",
    "76": "Occitanie",
    "84": "Auvergne-Rhône-Alpes",
    "93": "Provence-Alpes-Côte d'Azur",
    "94": "Corse",
}

KNOWN_DVF_GAPS = ("57", "67", "68", "976")


def normalize_department(value: object) -> str | None:
    """Return a canonical INSEE department code, including Corsica/overseas."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "<NA>"}:
        return None
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        text = text.split(".", 1)[0]
        if len(text) <= 2:
            return text.zfill(2)
        return text.zfill(3)
    if text in {"2A", "2B"}:
        return text
    return None


def department_from_commune(commune_code: object) -> str | None:
    """Infer the department prefix of a normalized INSEE commune code."""
    if commune_code is None:
        return None
    code = str(commune_code).strip().upper()
    if not code or code in {"NAN", "<NA>"}:
        return None
    if code.startswith(("97", "98")) and len(code) >= 3:
        return normalize_department(code[:3])
    if code.startswith(("2A", "2B")):
        return code[:2]
    return normalize_department(code[:2])


def region_from_department(department_code: object) -> str | None:
    """Map a department code to its official region code."""
    department = normalize_department(department_code)
    return DEPARTMENT_TO_REGION.get(department) if department else None
