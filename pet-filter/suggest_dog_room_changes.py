from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import urllib.parse

# Reuse your existing types:
# - DogEyeScores (from rate_dog_attractiveness_with_yolo module)
# - DetectedObject (optional for future use)
# - rate_dog_attractiveness_with_yolo


# ---------- Data models ----------

@dataclass
class AmazonLink:
    """Option A: just a search link (no scraping, no PA-API)."""
    label: str
    query: str
    url: str


@dataclass
class Suggestion:
    title: str
    why_it_helps: str
    effort: str               # "tiny" | "small" | "medium"
    cost: str                 # "free" | "low" | "medium"
    category: str             # "floor" | "rest" | "sniff" | "water_food" | "safety"
    steps: List[str]
    expected_score_lift: Dict[str, float]
    amazon_links: List[AmazonLink] = field(default_factory=list)  # paid suggestions only


# ---------- Engine ----------

class DogRoomSuggestionEngine:
    """
    Generates tenant-friendly suggestions to improve dog-room attractiveness scores.

    Option A Amazon links for PAID suggestions:
      - No scraping
      - Uses Amazon search links from curated queries and/or OpenAI refinement.
    """

    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        max_suggestions: int = 6,
        use_openai: bool = True,
        amazon_domain: str = "www.amazon.com",
        amazon_tag: Optional[str] = None,
    ) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_suggestions = max_suggestions
        self.use_openai = use_openai and bool(self.openai_api_key)
        self.amazon_domain = amazon_domain
        self.amazon_tag = amazon_tag

    # ---------- public API ----------

    def suggest(
        self,
        scores: "DogEyeScores",
        debug: Optional[Dict[str, Any]] = None,
    ) -> List[Suggestion]:
        debug = debug or {}
        base = self._rule_based_suggestions(scores, debug)

        base_sorted = sorted(
            base,
            key=lambda s: (
                s.expected_score_lift.get("overall", 0.0),
                s.expected_score_lift.get(self._lowest_dim(scores), 0.0),
            ),
            reverse=True,
        )[: self.max_suggestions]

        base_sorted = self._attach_amazon_links(base_sorted)

        if not self.use_openai:
            return base_sorted

        try:
            refined = self._openai_refine(scores, debug, base_sorted)
            refined = self._attach_amazon_links(refined or base_sorted)
            return refined or base_sorted
        except Exception:
            return base_sorted

    def split_free_paid(self, suggestions: List[Suggestion]) -> Tuple[List[Suggestion], List[Suggestion]]:
        free = [s for s in suggestions if s.cost == "free"]
        paid = [s for s in suggestions if s.cost != "free"]
        return free, paid

    # ---------- helpers ----------

    def _lowest_dim(self, scores: "DogEyeScores") -> str:
        dims = {
            "floor_play_space": scores.floor_play_space,
            "rest_cozy": scores.rest_cozy,
            "sniff_enrichment": scores.sniff_enrichment,
            "water_food_ready": scores.water_food_ready,
            "safety_low_threat": scores.safety_low_threat,
        }
        return min(dims.items(), key=lambda kv: kv[1])[0]

    def _amazon_search_url(self, query: str) -> str:
        q = urllib.parse.quote_plus(query)
        url = f"https://{self.amazon_domain}/s?k={q}"
        if self.amazon_tag:
            url += f"&tag={urllib.parse.quote_plus(self.amazon_tag)}"
        return url

    def _attach_amazon_links(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        for s in suggestions:
            if s.cost == "free":
                s.amazon_links = []
                continue

            if s.amazon_links:
                fixed: List[AmazonLink] = []
                for link in s.amazon_links:
                    url = link.url or self._amazon_search_url(link.query or link.label)
                    fixed.append(AmazonLink(label=link.label, query=link.query, url=url))
                s.amazon_links = fixed
                continue

            defaults = self._default_paid_queries_for(s)
            s.amazon_links = [
                AmazonLink(label=label, query=query, url=self._amazon_search_url(query))
                for (label, query) in defaults
            ]
        return suggestions

    def _default_paid_queries_for(self, s: Suggestion) -> List[Tuple[str, str]]:
        """
        Small, tenant-friendly items only. Keep 2–3.
        """
        cat = s.category

        if cat == "floor":
            return [
                ("Non-slip rug pad", "non slip rug pad"),
                ("Dog puzzle feeder (small)", "dog puzzle feeder toy"),
                ("Toy basket", "small toy storage basket"),
            ]

        if cat == "rest":
            return [
                ("Washable dog bed mat", "washable dog bed mat"),
                ("Waterproof blanket cover", "waterproof pet blanket cover"),
                ("Cooling mat (optional)", "dog cooling mat"),
            ]

        if cat == "sniff":
            return [
                ("Snuffle mat", "snuffle mat for dogs"),
                ("Lick mat", "dog lick mat"),
                ("Treat pouch (for scent games)", "dog treat pouch"),
            ]

        if cat == "water_food":
            return [
                ("Non-spill water bowl", "non spill dog water bowl"),
                ("Silicone bowl mat", "silicone pet feeding mat"),
                ("Travel water bottle", "dog water bottle"),
            ]

        # safety
        return [
            ("Cable protector", "cord protector for pets"),
            ("Corner guards", "foam corner protectors"),
            ("Baby gate (pressure-mounted)", "pressure mounted pet gate"),
        ]

    # ---------- deterministic suggestions ----------

    def _rule_based_suggestions(self, scores: "DogEyeScores", debug: Dict[str, Any]) -> List[Suggestion]:
        counts: Dict[str, int] = debug.get("counts", {}) or {}
        edge_density: float = float(debug.get("edge_density", 0.0) or 0.0)
        mean_brightness: float = float(debug.get("mean_brightness", 0.0) or 0.0)
        warmth: float = float(debug.get("warmth_mean_r_minus_b", 0.0) or 0.0)

        obstacle_area: float = float(debug.get("obstacle_area", 0.0) or 0.0)
        rest_area: float = float(debug.get("rest_area", 0.0) or 0.0)
        sniff_area: float = float(debug.get("sniff_area", 0.0) or 0.0)
        water_food_area: float = float(debug.get("water_food_area", 0.0) or 0.0)

        threat_count: int = int(debug.get("threat_count", 0) or 0)
        threat_area: float = float(debug.get("threat_area", 0.0) or 0.0)

        # Order priorities by weakest score
        dims = [
            ("floor", scores.floor_play_space),
            ("rest", scores.rest_cozy),
            ("sniff", scores.sniff_enrichment),
            ("water_food", scores.water_food_ready),
            ("safety", scores.safety_low_threat),
        ]
        dims_sorted = sorted(dims, key=lambda x: x[1])
        priority = {k: i for i, (k, _) in enumerate(dims_sorted)}

        out: List[Suggestion] = []

        # --- Floor / play space ---
        # Dogs value clear floor routes (zoomies, toys, training).
        if scores.floor_play_space < 0.70 or edge_density > 0.18 or obstacle_area > 0.18:
            out.append(Suggestion(
                title="Make a clear ‘play lane’ (1–2 steps wide) using what you already have",
                why_it_helps="Dogs play and move on the floor; a clear lane reduces stress and boosts play space immediately.",
                effort="small",
                cost="free",
                category="floor",
                steps=[
                    "Pick one main route (door → couch area) and move small objects off the floor.",
                    "Push chairs fully under tables and tuck loose items into one basket/box.",
                    "Keep one corner open as a dedicated toy/play zone.",
                ],
                expected_score_lift={"floor_play_space": 0.08, "safety_low_threat": 0.03, "overall": 0.05},
            ))

        # Add an optional paid enhancement for traction + toy containment
        if scores.floor_play_space < 0.75:
            out.append(Suggestion(
                title="Improve traction and keep toys contained (small add-on)",
                why_it_helps="Less slipping and fewer scattered toys makes the room safer and more fun to play in.",
                effort="tiny",
                cost="low",
                category="floor",
                steps=[
                    "Add a non-slip pad under rugs/runner paths (no permanent install).",
                    "Use a small toy basket so toys don’t become hazards.",
                ],
                expected_score_lift={"floor_play_space": 0.05, "safety_low_threat": 0.03, "overall": 0.04},
            ))

        # --- Rest / cozy ---
        # Dogs often prefer a consistent “place” spot.
        if scores.rest_cozy < 0.70 or (warmth < 0.03 and rest_area < 0.12):
            out.append(Suggestion(
                title="Create one consistent ‘place’ spot (rest zone) near your routine",
                why_it_helps="A predictable resting spot reduces roaming and helps dogs settle quickly.",
                effort="tiny",
                cost="free",
                category="rest",
                steps=[
                    "Choose a spot near where you sit (dogs like proximity).",
                    "Put a folded blanket/towel on an existing mat or corner of the couch (if allowed).",
                    "Keep it consistent—don’t move it daily.",
                ],
                expected_score_lift={"rest_cozy": 0.07, "overall": 0.04},
            ))

        if scores.rest_cozy < 0.75:
            out.append(Suggestion(
                title="Upgrade the rest zone with an easy-clean mat (small add-on)",
                why_it_helps="A washable, non-slip mat makes resting more comfortable and keeps the area tidy.",
                effort="tiny",
                cost="low",
                category="rest",
                steps=[
                    "Add a washable dog mat/bed pad in the same ‘place’ location.",
                    "If your floor is slippery, put a thin non-slip pad underneath.",
                ],
                expected_score_lift={"rest_cozy": 0.06, "safety_low_threat": 0.02, "overall": 0.04},
            ))

        # --- Sniff / enrichment ---
        # Dogs need scent-based enrichment. Keep it structured (not chaotic clutter).
        if scores.sniff_enrichment < 0.70 or sniff_area < 0.06:
            out.append(Suggestion(
                title="Add a 5-minute daily sniff game (free, no new items)",
                why_it_helps="Sniffing is mentally tiring in a good way; it improves enrichment without adding clutter.",
                effort="tiny",
                cost="free",
                category="sniff",
                steps=[
                    "Hide 5–8 treats in easy spots at nose height (behind a chair leg, near a plant—safe only).",
                    "Increase difficulty slowly (different corners each day).",
                    "End when your dog succeeds so it stays fun.",
                ],
                expected_score_lift={"sniff_enrichment": 0.08, "overall": 0.05},
            ))

        out.append(Suggestion(
            title="Add one structured enrichment item (small add-on, rotate weekly)",
            why_it_helps="A snuffle mat / lick mat boosts enrichment without making the room messy.",
            effort="tiny",
            cost="low",
            category="sniff",
            steps=[
                "Pick ONE: snuffle mat, lick mat, or puzzle feeder (start easy).",
                "Use it in the same corner, then store it after use to reduce clutter.",
                "Rotate weekly so it stays novel.",
            ],
            expected_score_lift={"sniff_enrichment": 0.06, "overall": 0.04},
        ))

        # --- Water / food readiness ---
        # COCO proxies are weak; treat as gentle hints.
        if scores.water_food_ready < 0.60 or water_food_area < 0.04:
            out.append(Suggestion(
                title="Make a tidy, consistent water station",
                why_it_helps="Dogs drink more reliably when water is always in one easy-to-find place.",
                effort="tiny",
                cost="free",
                category="water_food",
                steps=[
                    "Pick a corner away from carpet and foot traffic.",
                    "Place the water bowl on a towel (or tray) to catch splashes.",
                    "Refill at the same times each day so it becomes a routine.",
                ],
                expected_score_lift={"water_food_ready": 0.07, "safety_low_threat": 0.02, "overall": 0.04},
            ))

            out.append(Suggestion(
                title="Reduce mess with a bowl mat or non-spill bowl (small add-on)",
                why_it_helps="Less slipping, less splash, and less cleanup—especially in rentals.",
                effort="tiny",
                cost="low",
                category="water_food",
                steps=[
                    "Add a silicone feeding mat under the bowls.",
                    "If spills happen a lot, use a non-spill water bowl.",
                ],
                expected_score_lift={"water_food_ready": 0.06, "safety_low_threat": 0.03, "overall": 0.04},
            ))

        # --- Safety / low threat ---
        if scores.safety_low_threat < 0.72 or edge_density > 0.18 or threat_count > 0 or threat_area > 0.02:
            out.append(Suggestion(
                title="Dog-proof one hotspot (cords + small items)",
                why_it_helps="Reducing tempting hazards (cords, small chewables) makes the room safer and calmer.",
                effort="small",
                cost="free",
                category="safety",
                steps=[
                    "Bundle cords and lift them off the floor where possible.",
                    "Move small chewable items (remotes, shoes) into a bin or onto a shelf.",
                    "Keep trash and food out of reach (close doors or use a lidded bin).",
                ],
                expected_score_lift={"safety_low_threat": 0.08, "overall": 0.05},
            ))

            out.append(Suggestion(
                title="Add a pressure-mounted gate for boundaries (small add-on)",
                why_it_helps="Boundaries reduce stress and prevent accidents—no drilling required.",
                effort="small",
                cost="medium",
                category="safety",
                steps=[
                    "Use a pressure-mounted gate to block one off-limits area.",
                    "Start with short sessions and reward calm behavior near the gate.",
                ],
                expected_score_lift={"safety_low_threat": 0.06, "overall": 0.04},
            ))

        out = sorted(out, key=lambda s: priority.get(s.category, 999))
        return out[: max(self.max_suggestions, 6)]

    # ---------- OpenAI refinement ----------

    def _openai_refine(
        self,
        scores: "DogEyeScores",
        debug: Dict[str, Any],
        base: List[Suggestion],
    ) -> List[Suggestion]:
        """
        OpenAI refinement: rewrite for clarity and optionally improve Amazon search queries for paid items.
        Output: strict JSON.
        """
        from openai import OpenAI  # pip install openai
        client = OpenAI(api_key=self.openai_api_key)

        payload = {
            "scores": {
                "floor_play_space": scores.floor_play_space,
                "rest_cozy": scores.rest_cozy,
                "sniff_enrichment": scores.sniff_enrichment,
                "water_food_ready": scores.water_food_ready,
                "safety_low_threat": scores.safety_low_threat,
                "overall": scores.overall,
            },
            "signals": {
                "counts_top": dict(sorted((debug.get("counts") or {}).items(), key=lambda kv: kv[1], reverse=True)[:12]),
                "edge_density": float(debug.get("edge_density", 0.0) or 0.0),
                "warmth": float(debug.get("warmth_mean_r_minus_b", 0.0) or 0.0),
                "mean_brightness": float(debug.get("mean_brightness", 0.0) or 0.0),
                "obstacle_area": float(debug.get("obstacle_area", 0.0) or 0.0),
                "rest_area": float(debug.get("rest_area", 0.0) or 0.0),
                "sniff_area": float(debug.get("sniff_area", 0.0) or 0.0),
                "water_food_area": float(debug.get("water_food_area", 0.0) or 0.0),
                "threat_count": int(debug.get("threat_count", 0) or 0),
            },
            "base_suggestions": [
                {
                    **s.__dict__,
                    "amazon_links": [{"label": a.label, "query": a.query, "url": a.url} for a in (s.amazon_links or [])],
                }
                for s in base
            ],
            "constraints": {
                "tenant_friendly": True,
                "avoid": [
                    "buying large furniture",
                    "drilling holes or permanent mounting",
                    "renovations",
                ],
                "prefer": [
                    "decluttering + rearranging existing items",
                    "small removable items (mats, blankets, gates, enrichment toys)",
                    "steps under 30 minutes",
                ],
                "amazon_links_policy": "Provide Amazon SEARCH queries only; do not invent prices, ratings, or specific brands.",
                "output_limit": self.max_suggestions,
            },
        }

        system = (
            "You are an expert in dog-friendly interior tweaks for renters. "
            "Propose minimal, practical changes to improve the given dog-room scores. "
            "If a suggestion is paid (cost != 'free'), include 1-3 Amazon SEARCH links (label+query). "
            "Return STRICT JSON only matching the schema."
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
                            "category": {"type": "string", "enum": ["floor", "rest", "sniff", "water_food", "safety"]},
                            "steps": {"type": "array", "items": {"type": "string"}},
                            "expected_score_lift": {"type": "object", "additionalProperties": {"type": "number"}},
                            "amazon_links": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "query": {"type": "string"},
                                    },
                                    "required": ["label", "query"],
                                    "additionalProperties": False,
                                },
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
            response_format={"type": "json_schema", "json_schema": {"name": "dog_room_suggestions", "schema": schema}},
        )

        data = json.loads(resp.output_text)

        out: List[Suggestion] = []
        for item in data.get("suggestions", [])[: self.max_suggestions]:
            raw_links = item.get("amazon_links") or []
            links: List[AmazonLink] = []
            for rl in raw_links[:3]:
                query = (rl.get("query") or "").strip()
                label = (rl.get("label") or query or "Amazon search").strip()
                if query:
                    links.append(AmazonLink(label=label, query=query, url=self._amazon_search_url(query)))
            item["amazon_links"] = links
            out.append(Suggestion(**item))

        return out


# ---------- Optional Streamlit renderer (Option A) ----------

def render_dog_suggestions_streamlit(suggestions: List[Suggestion]) -> None:
    import streamlit as st

    free = [s for s in suggestions if s.cost == "free"]
    paid = [s for s in suggestions if s.cost != "free"]

    st.subheader("Free changes (no purchases)")
    if not free:
        st.caption("No free changes found.")
    for s in free:
        with st.container(border=True):
            st.markdown(f"### {s.title}")
            st.write(s.why_it_helps)
            st.caption(f"Effort: {s.effort} • Category: {s.category}")
            st.markdown("**Steps**")
            for step in s.steps:
                st.write(f"• {step}")

    st.subheader("Paid add-ons (small items)")
    if not paid:
        st.caption("No paid add-ons suggested.")
    for s in paid:
        with st.container(border=True):
            st.markdown(f"### {s.title}")
            st.write(s.why_it_helps)
            st.caption(f"Effort: {s.effort} • Cost: {s.cost} • Category: {s.category}")

            st.markdown("**Steps**")
            for step in s.steps:
                st.write(f"• {step}")

            if s.amazon_links:
                st.markdown("**Amazon links**")
                for link in s.amazon_links[:3]:
                    st.link_button(link.label, link.url)
            else:
                st.caption("No links available for this item.")


# ---- Example usage ----
# scores, dbg = rate_dog_attractiveness_with_yolo("room.jpg", return_debug=True)
# engine = DogRoomSuggestionEngine(use_openai=True, amazon_tag=os.getenv("AMAZON_TAG"))
# suggestions = engine.suggest(scores, dbg)
# free, paid = engine.split_free_paid(suggestions)
# render_dog_suggestions_streamlit(free + paid)
