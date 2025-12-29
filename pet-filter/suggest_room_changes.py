from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import json

# Reuse your existing types:
# - CatEyeScores
# - DetectedObject
# - CATEGORY_NAMES
# - rate_cat_attractiveness_with_yolo

@dataclass
class Suggestion:
    title: str
    why_it_helps: str
    effort: str              # "tiny" | "small" | "medium"
    cost: str                # "free" | "low" | "medium"
    category: str            # "vertical" | "shelter" | "cozy" | "exploration" | "safety"
    steps: List[str]
    expected_score_lift: Dict[str, float]  # e.g. {"vertical_opportunity": 0.05, "overall": 0.03}


class CatRoomSuggestionEngine:
    """
    Generates tenant-friendly suggestions to improve cat-attractiveness scores.

    Strategy:
      1) Identify lowest scoring dimensions and relevant signals from debug (objects, nooks, clutter, warmth, etc.)
      2) Produce a set of candidates via deterministic rules (always available)
      3) Optionally, refine/expand with OpenAI for more natural, room-specific suggestions
    """

    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        max_suggestions: int = 6,
        use_openai: bool = True,
    ) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_suggestions = max_suggestions
        self.use_openai = use_openai and bool(self.openai_api_key)

    # ---------- public API ----------

    def suggest(
        self,
        scores: CatEyeScores,
        debug: Optional[Dict[str, Any]] = None,
    ) -> List[Suggestion]:
        debug = debug or {}
        base = self._rule_based_suggestions(scores, debug)

        # Keep only best N by predicted overall lift (rough heuristic)
        base_sorted = sorted(
            base,
            key=lambda s: (s.expected_score_lift.get("overall", 0.0), s.expected_score_lift.get(self._best_dim(scores), 0.0)),
            reverse=True,
        )
        base_sorted = base_sorted[: self.max_suggestions]

        if not self.use_openai:
            return base_sorted

        # Refine with OpenAI: rewrite + add room-specific nuance; keep constraints (tenant-friendly)
        try:
            refined = self._openai_refine(scores, debug, base_sorted)
            # If model returns junk or empty, fall back
            return refined or base_sorted
        except Exception:
            return base_sorted

    # ---------- deterministic logic ----------

    def _best_dim(self, scores: CatEyeScores) -> str:
        # lowest is what we want to improve
        dims = {
            "vertical_opportunity": scores.vertical_opportunity,
            "shelter_hiding": scores.shelter_hiding,
            "cozy_warmth": scores.cozy_warmth,
            "exploration_richness": scores.exploration_richness,
            "safety_low_threat": scores.safety_low_threat,
        }
        return min(dims.items(), key=lambda kv: kv[1])[0]

    def _rule_based_suggestions(self, scores: CatEyeScores, debug: Dict[str, Any]) -> List[Suggestion]:
        # Pull useful signals
        counts: Dict[str, int] = debug.get("counts", {}) or {}
        area_frac: Dict[str, float] = debug.get("area_frac", {}) or {}
        edge_density: float = float(debug.get("edge_density", 0.0) or 0.0)
        warmth: float = float(debug.get("warmth_mean_r_minus_b", 0.0) or 0.0)
        mean_brightness: float = float(debug.get("mean_brightness", 0.0) or 0.0)
        nook_count: int = int(debug.get("nook_count", 0) or 0)
        shelter_area: float = float(debug.get("shelter_area", 0.0) or 0.0)
        vertical_area: float = float(debug.get("vertical_area", 0.0) or 0.0)
        cozy_area: float = float(debug.get("cozy_area", 0.0) or 0.0)
        explore_area: float = float(debug.get("explore_area", 0.0) or 0.0)
        threat_count: int = int(debug.get("threat_count", 0) or 0)
        threat_area: float = float(debug.get("threat_area", 0.0) or 0.0)

        # Decide priority ordering
        dims = [
            ("vertical", scores.vertical_opportunity),
            ("shelter", scores.shelter_hiding),
            ("cozy", scores.cozy_warmth),
            ("exploration", scores.exploration_richness),
            ("safety", scores.safety_low_threat),
        ]
        dims_sorted = sorted(dims, key=lambda x: x[1])  # lowest first

        suggestions: List[Suggestion] = []

        # --- Vertical opportunity ---
        if scores.vertical_opportunity < 0.65 or vertical_area < 0.06:
            suggestions.append(Suggestion(
                title="Create one ‘cat highway’ without buying furniture",
                why_it_helps="Cats love vantage points and predictable routes. A simple path of climbable surfaces boosts vertical opportunity fast.",
                effort="small",
                cost="free",
                category="vertical",
                steps=[
                    "Clear a route along existing sturdy surfaces (chair → couch arm → table edge).",
                    "Move a chair next to the safest ‘step-up’ point (no wobble).",
                    "Keep the landing zones uncluttered so jumps feel safe.",
                ],
                expected_score_lift={"vertical_opportunity": 0.06, "overall": 0.03},
            ))

        if counts.get("book", 0) > 0 or counts.get("tv", 0) > 0 or counts.get("laptop", 0) > 0:
            suggestions.append(Suggestion(
                title="Turn existing shelves/tables into a lookout (safe + soft)",
                why_it_helps="A soft, non-slip perch on an existing elevated spot is a big win with minimal change.",
                effort="tiny",
                cost="low",
                category="vertical",
                steps=[
                    "Pick an elevated surface your cat already approaches (table corner / window-adjacent surface).",
                    "Add a folded towel or small non-slip mat (no adhesives needed).",
                    "Keep one side as a clear escape route (cats prefer 2 exits).",
                ],
                expected_score_lift={"vertical_opportunity": 0.04, "safety_low_threat": 0.02, "overall": 0.03},
            ))

        # --- Shelter/hiding ---
        if scores.shelter_hiding < 0.65 or (nook_count < 2 and shelter_area < 0.10):
            suggestions.append(Suggestion(
                title="Add a no-drill hiding nook using what you already have",
                why_it_helps="Safe ‘caves’ reduce stress and improve shelter/hiding score without buying big items.",
                effort="small",
                cost="free",
                category="shelter",
                steps=[
                    "Use a chair + blanket to create a small tent (leave 2 openings).",
                    "Or slide a cardboard box under a table/desk and cut an extra exit hole.",
                    "Place it away from the busiest walkway and loudest devices.",
                ],
                expected_score_lift={"shelter_hiding": 0.08, "safety_low_threat": 0.03, "overall": 0.05},
            ))

        # --- Cozy warmth ---
        if scores.cozy_warmth < 0.65 or (warmth < 0.03 and cozy_area < 0.12):
            suggestions.append(Suggestion(
                title="Make one ‘warm spot’ with a throw and consistent sunlight",
                why_it_helps="A dedicated cozy zone increases warmth/comfort and encourages resting behavior.",
                effort="tiny",
                cost="low",
                category="cozy",
                steps=[
                    "Pick the warmest spot (near a window or away from drafts).",
                    "Add a folded throw/blanket or small pet mat on an existing chair/couch corner.",
                    "Keep it consistent (cats love predictable spots).",
                ],
                expected_score_lift={"cozy_warmth": 0.07, "overall": 0.04},
            ))

        # --- Exploration richness ---
        if scores.exploration_richness < 0.65 or explore_area < 0.06:
            suggestions.append(Suggestion(
                title="Add 2–3 ‘micro-enrichment’ items (small + removable)",
                why_it_helps="A little novelty increases exploration without increasing clutter too much.",
                effort="tiny",
                cost="low",
                category="exploration",
                steps=[
                    "Add a small cat toy, paper bag (handles removed), or a crinkly ball in a corner zone.",
                    "Rotate items weekly (keep only 2–3 out at a time).",
                    "Place enrichment near the cat highway or cozy spot, not in the main walkway.",
                ],
                expected_score_lift={"exploration_richness": 0.06, "overall": 0.03},
            ))

        # --- Safety / low threat ---
        # Heuristic: high edge_density often means visual clutter
        if scores.safety_low_threat < 0.70 or edge_density > 0.18 or threat_count > 0 or threat_area > 0.02:
            suggestions.append(Suggestion(
                title="Reduce ‘visual clutter’ in one zone to feel safer",
                why_it_helps="Cats avoid tight, chaotic paths. A clear route improves safety and also helps vertical movement.",
                effort="small",
                cost="free",
                category="safety",
                steps=[
                    "Pick one 1m x 1m zone near the cat’s main path and clear small items from the floor.",
                    "Bundle cables and move dangling items off jump routes.",
                    "If there’s a mirror/TV reflection causing stress, angle it slightly or cover part with a cloth temporarily.",
                ],
                expected_score_lift={"safety_low_threat": 0.06, "vertical_opportunity": 0.02, "overall": 0.04},
            ))

        # Cap suggestions, but bias toward lowest dimensions first
        priority = {k: i for i, (k, _) in enumerate(dims_sorted)}
        suggestions = sorted(suggestions, key=lambda s: priority.get(s.category, 999))
        return suggestions[: max(self.max_suggestions, 6)]

    # ---------- OpenAI refinement ----------

    def _openai_refine(
        self,
        scores: CatEyeScores,
        debug: Dict[str, Any],
        base: List[Suggestion],
    ) -> List[Suggestion]:
        """
        Calls OpenAI to:
          - rewrite suggestions more naturally
          - keep them tenant-friendly
          - ensure they are tied to the lowest scoring dimensions
          - return strict JSON
        """
        # Lazy import so core pipeline works without OpenAI SDK
        from openai import OpenAI  # pip install openai

        client = OpenAI(api_key=self.openai_api_key)

        payload = {
            "scores": {
                "vertical_opportunity": scores.vertical_opportunity,
                "shelter_hiding": scores.shelter_hiding,
                "cozy_warmth": scores.cozy_warmth,
                "exploration_richness": scores.exploration_richness,
                "safety_low_threat": scores.safety_low_threat,
                "overall": scores.overall,
            },
            "signals": {
                # keep it compact; the model doesn’t need raw arrays
                "counts_top": dict(sorted((debug.get("counts") or {}).items(), key=lambda kv: kv[1], reverse=True)[:12]),
                "edge_density": float(debug.get("edge_density", 0.0) or 0.0),
                "warmth": float(debug.get("warmth_mean_r_minus_b", 0.0) or 0.0),
                "mean_brightness": float(debug.get("mean_brightness", 0.0) or 0.0),
                "nook_count": int(debug.get("nook_count", 0) or 0),
                "nook_area_sum": float(debug.get("nook_area_sum", 0.0) or 0.0),
                "cozy_area": float(debug.get("cozy_area", 0.0) or 0.0),
                "shelter_area": float(debug.get("shelter_area", 0.0) or 0.0),
                "vertical_area": float(debug.get("vertical_area", 0.0) or 0.0),
                "explore_area": float(debug.get("explore_area", 0.0) or 0.0),
                "threat_count": int(debug.get("threat_count", 0) or 0),
            },
            "base_suggestions": [s.__dict__ for s in base],
            "constraints": {
                "tenant_friendly": True,
                "avoid": [
                    "buying large furniture (sofa, big cat tree)",
                    "drilling holes or mounting shelves",
                    "permanent modifications",
                    "messy renovations",
                ],
                "prefer": [
                    "rearranging existing furniture",
                    "adding small items (small plant, small mat, blanket, cardboard box)",
                    "cheap removable accessories (non-slip mat, command hooks only if optional)",
                    "steps that take under 30 minutes",
                ],
                "output_limit": self.max_suggestions,
            },
        }

        system = (
            "You are an expert in cat-friendly interior tweaks. "
            "You must propose minimal, tenant-friendly changes to improve the given cat-attractiveness scores. "
            "Return STRICT JSON only that matches the provided schema."
        )

        schema = {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "why_it_helps": {"type": "string"},
                            "effort": {"type": "string", "enum": ["tiny", "small", "medium"]},
                            "cost": {"type": "string", "enum": ["free", "low", "medium"]},
                            "category": {"type": "string", "enum": ["vertical", "shelter", "cozy", "exploration", "safety"]},
                            "steps": {"type": "array", "items": {"type": "string"}},
                            "expected_score_lift": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                        },
                        "required": ["title", "why_it_helps", "effort", "cost", "category", "steps", "expected_score_lift"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["suggestions"],
            "additionalProperties": False,
        }

        resp = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Use the following data to produce suggestions:\n" + json.dumps(payload)},
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "cat_room_suggestions", "schema": schema}},
        )

        data = json.loads(resp.output_text)
        out: List[Suggestion] = []
        for item in data.get("suggestions", [])[: self.max_suggestions]:
            out.append(Suggestion(**item))
        return out