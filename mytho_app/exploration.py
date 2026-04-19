from __future__ import annotations

import math
import re
from collections import defaultdict

from mytho_app.constants import ABSTRACT_FIELDS, TITLE_FIELDS
from mytho_app.parsing import clean_text, normalize_text

COORDINATE_TOKEN_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*°?\s*([NSEW])?", re.IGNORECASE)
ENTRY_ID_POPUP_RE = re.compile(r'data-entry-id="([^"]+)"')

RED_RGB = (215, 38, 61)
BLUE_RGB = (44, 123, 182)
ORIGINAL_STORY_FALLBACK_COORDINATES = (0.0, 170.0)
RELATED_STORY_FALLBACK_CENTER = (0.0, 170.0)
RELATED_STORY_FALLBACK_RING = (
    (0.0, 0.0),
    (0.9, 0.0),
    (-0.9, 0.0),
    (0.0, 0.9),
    (0.0, -0.9),
    (0.7, 0.7),
    (0.7, -0.7),
    (-0.7, 0.7),
    (-0.7, -0.7),
)


def _apply_direction(value: float, direction: str) -> float:
    if direction in {"S", "W"}:
        return -abs(value)
    if direction in {"N", "E"}:
        return abs(value)
    return value


def parse_space_coord(value: str) -> tuple[float, float] | None:
    text = clean_text(value).replace("−", "-")
    if not text:
        return None

    cleaned = (
        text.replace("≈", "")
        .replace("~", "")
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
    )
    cleaned = re.sub(r"(?<=\d),(?=\d)", ".", cleaned)
    matches = COORDINATE_TOKEN_RE.findall(cleaned)
    if len(matches) < 2:
        return None

    lat = _apply_direction(float(matches[0][0]), matches[0][1].upper())
    lon = _apply_direction(float(matches[1][0]), matches[1][1].upper())
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def similarity_to_color(score: float, minimum: float) -> str:
    if score >= 0.999:
        return f"#{RED_RGB[0]:02x}{RED_RGB[1]:02x}{RED_RGB[2]:02x}"

    span = max(1.0 - minimum, 1e-9)
    ratio = min(1.0, max(0.0, (score - minimum) / span))
    red = round(BLUE_RGB[0] + (RED_RGB[0] - BLUE_RGB[0]) * ratio)
    green = round(BLUE_RGB[1] + (RED_RGB[1] - BLUE_RGB[1]) * ratio)
    blue = round(BLUE_RGB[2] + (RED_RGB[2] - BLUE_RGB[2]) * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def entry_title(entry: dict) -> str:
    fields = entry.get("fields", {})
    for field_name in TITLE_FIELDS:
        title = clean_text(fields.get(field_name, ""))
        if title:
            return title
    return clean_text(entry.get("label", "")) or entry["entry_id"]


def english_story_title(entry: dict) -> str:
    fields = entry.get("fields", {})
    english_title = clean_text(fields.get("Story title (Eng)", ""))
    if english_title:
        return english_title
    return entry_title(entry)


def popup_html_for_entry(entry_id: str, lines: list[str]) -> str:
    hidden_id = f'<span data-entry-id="{entry_id}" style="display:none"></span>'
    return "<br>".join([hidden_id, *lines])


def extract_entry_id_from_popup(popup_html: str) -> str | None:
    match = ENTRY_ID_POPUP_RE.search(clean_text(popup_html))
    if not match:
        return None
    entry_id = clean_text(match.group(1))
    return entry_id or None


def entry_id_from_map_click(markers: list[dict], popup_html: str, clicked_point: dict | None) -> str | None:
    entry_id = extract_entry_id_from_popup(popup_html)
    if entry_id:
        return entry_id

    if not clicked_point:
        return None

    latitude = clicked_point.get("lat")
    longitude = clicked_point.get("lng")
    if latitude is None or longitude is None:
        return None

    for marker in markers:
        marker_latitude, marker_longitude = marker["coordinates"]
        if abs(marker_latitude - latitude) < 1e-6 and abs(marker_longitude - longitude) < 1e-6:
            return marker["entry_id"]
    return None


def primary_abstract(entry: dict) -> str:
    fields = entry.get("fields", {})
    preferred_fields = ["Abstract (Eng)", "Abstract (Fr)", "1-sentence summary"]
    for field_name in preferred_fields:
        value = clean_text(fields.get(field_name, ""))
        if value:
            return value
    for field_name in ABSTRACT_FIELDS:
        value = clean_text(fields.get(field_name, ""))
        if value:
            return value
    return ""


def _great_circle_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    arc = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371 * math.asin(min(1.0, math.sqrt(arc)))


def _marker_payload(
    *,
    entry: dict,
    coordinates: tuple[float, float],
    kind: str,
    similarity: float,
    matched_patterns: list[dict],
    color: str,
    has_location: bool = True,
) -> dict:
    return {
        "entry_id": entry["entry_id"],
        "entry": entry,
        "coordinates": coordinates,
        "kind": kind,
        "similarity": similarity,
        "matched_patterns": matched_patterns,
        "color": color,
        "title": entry_title(entry),
        "hover_title": english_story_title(entry),
        "abstract": primary_abstract(entry),
        "has_location": has_location,
    }


def _related_story_fallback_coordinates(index: int) -> tuple[float, float]:
    lat_offset, lon_offset = RELATED_STORY_FALLBACK_RING[index % len(RELATED_STORY_FALLBACK_RING)]
    return (
        RELATED_STORY_FALLBACK_CENTER[0] + lat_offset,
        RELATED_STORY_FALLBACK_CENTER[1] + lon_offset,
    )


def build_exploration_network(
    entries: list[dict],
    selected_pattern: str,
    related_patterns: list[dict],
    *,
    minimum_similarity: float,
) -> dict:
    selected_marker = normalize_text(selected_pattern)
    entries_by_pattern: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    entry_by_id = {entry["entry_id"]: entry for entry in entries}

    for entry in entries:
        for pattern in entry.get("patterns", []):
            entries_by_pattern[normalize_text(pattern)].append((pattern, entry))

    original_markers: list[dict] = []
    original_ids: set[str] = set()
    missing_original_coords = 0
    original_entries = entries_by_pattern.get(selected_marker, [])
    for pattern_text, entry in original_entries:
        coordinates = parse_space_coord(entry.get("fields", {}).get("space coord", ""))
        has_location = coordinates is not None
        if coordinates is None:
            missing_original_coords += 1
            coordinates = ORIGINAL_STORY_FALLBACK_COORDINATES
        original_ids.add(entry["entry_id"])
        original_markers.append(
            _marker_payload(
                entry=entry,
                coordinates=coordinates,
                kind="original",
                similarity=1.0,
                matched_patterns=[{"pattern": pattern_text, "score": 1.0}],
                color=similarity_to_color(1.0, minimum_similarity),
                has_location=has_location,
            )
        )

    related_by_entry_id: dict[str, dict] = {}
    missing_related_coords = 0
    seen_missing_related: set[str] = set()
    missing_related_index = 0
    for relation in related_patterns:
        pattern_text = clean_text(relation.get("text", ""))
        pattern_marker = normalize_text(pattern_text)
        if not pattern_marker or pattern_marker == selected_marker:
            continue
        score = float(relation.get("score", 0.0))
        for matched_pattern_text, entry in entries_by_pattern.get(pattern_marker, []):
            entry_id = entry["entry_id"]
            if entry_id in original_ids:
                continue
            coordinates = parse_space_coord(entry.get("fields", {}).get("space coord", ""))
            has_location = coordinates is not None
            if coordinates is None:
                if entry_id not in seen_missing_related:
                    missing_related_coords += 1
                    seen_missing_related.add(entry_id)
                    coordinates = _related_story_fallback_coordinates(missing_related_index)
                    missing_related_index += 1
                else:
                    marker = related_by_entry_id.get(entry_id)
                    if marker is not None:
                        marker["matched_patterns"].append({"pattern": matched_pattern_text, "score": score})
                        if score > marker["similarity"]:
                            marker["similarity"] = score
                            marker["color"] = similarity_to_color(score, minimum_similarity)
                    continue

            marker = related_by_entry_id.get(entry_id)
            if marker is None:
                marker = _marker_payload(
                    entry=entry,
                    coordinates=coordinates,
                    kind="related",
                    similarity=score,
                    matched_patterns=[],
                    color=similarity_to_color(score, minimum_similarity),
                    has_location=has_location,
                )
                related_by_entry_id[entry_id] = marker

            marker["matched_patterns"].append({"pattern": matched_pattern_text, "score": score})
            if score > marker["similarity"]:
                marker["similarity"] = score
                marker["color"] = similarity_to_color(score, minimum_similarity)

    related_markers = sorted(
        related_by_entry_id.values(),
        key=lambda item: (-item["similarity"], item["title"].lower(), item["entry_id"]),
    )
    for marker in related_markers:
        marker["matched_patterns"].sort(key=lambda item: (-item["score"], item["pattern"].lower()))

    connections: list[dict] = []
    if original_markers:
        for marker in related_markers:
            nearest = min(
                original_markers,
                key=lambda original: _great_circle_distance(original["coordinates"], marker["coordinates"]),
            )
            connections.append(
                {
                    "source_entry_id": nearest["entry_id"],
                    "target_entry_id": marker["entry_id"],
                    "source_coordinates": nearest["coordinates"],
                    "target_coordinates": marker["coordinates"],
                    "similarity": marker["similarity"],
                    "color": marker["color"],
                }
            )

    all_markers = original_markers + related_markers
    bounds = None
    if all_markers:
        latitudes = [marker["coordinates"][0] for marker in all_markers]
        longitudes = [marker["coordinates"][1] for marker in all_markers]
        bounds = [[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]]

    return {
        "entry_by_id": entry_by_id,
        "original_markers": sorted(original_markers, key=lambda item: (item["title"].lower(), item["entry_id"])),
        "related_markers": related_markers,
        "connections": connections,
        "bounds": bounds,
        "missing_original_coords": missing_original_coords,
        "missing_related_coords": missing_related_coords,
        "original_story_count": len(original_entries),
        "related_story_count": len(related_markers),
        "related_pattern_count": len(related_patterns),
    }
