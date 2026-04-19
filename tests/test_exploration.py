from __future__ import annotations

import unittest

from mytho_app.exploration import (
    build_exploration_network,
    english_story_title,
    entry_id_from_map_click,
    parse_space_coord,
    popup_html_for_entry,
    primary_abstract,
)


def make_entry(
    entry_id: str,
    *,
    patterns: list[str],
    coord: str,
    title: str,
    abstract: str = "",
    summary: str = "",
) -> dict:
    return {
        "entry_id": entry_id,
        "label": title,
        "patterns": patterns,
        "fields": {
            "Story title (Eng)": title,
            "space coord": coord,
            "Abstract (Eng)": abstract,
            "1-sentence summary": summary,
            "territory": "Test territory",
        },
    }


class ExplorationParsingTests(unittest.TestCase):
    def test_parse_space_coord_accepts_decimal_coordinates(self) -> None:
        self.assertEqual(parse_space_coord("-20.859062, 165.258667"), (-20.859062, 165.258667))

    def test_parse_space_coord_accepts_directional_coordinates(self) -> None:
        self.assertEqual(parse_space_coord("≈ 16.0° S, 168.4° E"), (-16.0, 168.4))

    def test_parse_space_coord_accepts_semicolon_coordinates(self) -> None:
        self.assertEqual(parse_space_coord("22.2994° ; 166.7483°"), (22.2994, 166.7483))

    def test_parse_space_coord_accepts_decimal_commas(self) -> None:
        self.assertEqual(parse_space_coord("-4,198,\n152,163"), (-4.198, 152.163))

    def test_primary_abstract_prefers_abstract_over_summary(self) -> None:
        entry = make_entry(
            "entry-1",
            patterns=["Selected pattern"],
            coord="-20.0, 165.0",
            title="Story",
            abstract="Long abstract",
            summary="Short summary",
        )
        self.assertEqual(primary_abstract(entry), "Long abstract")

    def test_english_story_title_prefers_english_title(self) -> None:
        entry = make_entry(
            "entry-1",
            patterns=["Selected pattern"],
            coord="-20.0, 165.0",
            title="English title",
        )
        entry["fields"]["Story title (French)"] = "Titre"
        self.assertEqual(english_story_title(entry), "English title")

    def test_entry_id_from_map_click_prefers_hidden_popup_id(self) -> None:
        popup_html = popup_html_for_entry("entry-7", ["<strong>Story</strong>"])
        self.assertEqual(entry_id_from_map_click([], popup_html, None), "entry-7")

    def test_entry_id_from_map_click_can_fall_back_to_coordinates(self) -> None:
        markers = [{"entry_id": "entry-2", "coordinates": (-20.0, 165.0)}]
        clicked_point = {"lat": -20.0, "lng": 165.0}
        self.assertEqual(entry_id_from_map_click(markers, "", clicked_point), "entry-2")


class ExplorationNetworkTests(unittest.TestCase):
    def test_build_exploration_network_aggregates_related_entries(self) -> None:
        entries = [
            make_entry("original-a", patterns=["Selected pattern"], coord="-20.0, 165.0", title="Original A"),
            make_entry("original-b", patterns=["Selected pattern"], coord="≈ 16.0° S, 168.4° E", title="Original B"),
            make_entry("original-missing", patterns=["Selected pattern"], coord="unknown", title="Original Missing"),
            make_entry("overlap", patterns=["Selected pattern", "Related one"], coord="-19.5, 165.5", title="Overlap"),
            make_entry("related-a", patterns=["Related one"], coord="(-21.0, 166.0)", title="Related A"),
            make_entry("related-b", patterns=["Related one", "Related two"], coord="22.2994° ; 166.7483°", title="Related B"),
            make_entry("related-missing", patterns=["Related two"], coord="missing", title="Related Missing"),
        ]

        network = build_exploration_network(
            entries,
            "Selected pattern",
            [
                {"text": "Related one", "score": 0.81},
                {"text": "Related two", "score": 0.65},
            ],
            minimum_similarity=0.60,
        )

        self.assertEqual(network["original_story_count"], 4)
        self.assertEqual(len(network["original_markers"]), 3)
        self.assertEqual(network["missing_original_coords"], 1)

        self.assertEqual(len(network["related_markers"]), 2)
        self.assertEqual(network["missing_related_coords"], 1)
        self.assertEqual(len(network["connections"]), 2)

        related_marker = next(marker for marker in network["related_markers"] if marker["entry_id"] == "related-b")
        self.assertEqual(related_marker["similarity"], 0.81)
        self.assertEqual([item["pattern"] for item in related_marker["matched_patterns"]], ["Related one", "Related two"])

        visible_ids = {marker["entry_id"] for marker in network["related_markers"]}
        self.assertNotIn("overlap", visible_ids)
        self.assertIsNotNone(network["bounds"])


if __name__ == "__main__":
    unittest.main()
