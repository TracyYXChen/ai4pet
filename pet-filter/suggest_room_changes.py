from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import urllib.parse

# Reuse your existing types:
# - CatEyeScores
# - DetectedObject
# - CATEGORY_NAMES
# - rate_cat_attractiveness_with_yolo


# ---------- Data models ----------

@dataclass
class AmazonLink:
    """Option A: just a search link (no scraping, no PA-API)."""
    label: str                 # e.g. "Non-slip mat"
    query: str                 # e.g. "non slip pet mat"
    url: str                   # computed from query


@dataclass
class Suggestion:
    title: str
    why_it_helps: str
    effort: str               # "tiny" | "small" | "medium"
    cost: str                 # "free" | "low" | "medium"
    category: str             # "vertical" | "shelter" | "cozy" | "exploration" | "safety"
    steps: List[str]
    expected_score_lift: Dict[str, float]
    amazon_links: List[AmazonLink] = field(default_factory=list)  # only for paid suggestions


# ---------- Engine ----------

class CatRoomSuggestionEngine:
    """
    Generates tenant-friendly suggestions to improve cat-attractiveness scores.

    Now includes Option A Amazon links for PAID suggestions:
      - We do not scrape.
      - We generate Amazon search links based on curated keywords and/or OpenAI refinement.
    """

    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        max_suggestions: int = 6,
        use_openai: bool = True,
        amazon_domain: str = "www.amazon.com",
        amazon_tag: Optional[str] = None,  # optional affiliate tag; appended as `tag=...`
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
        scores: CatEyeScores,
        debug: Optional[Dict[str, Any]] = None,
    ) -> List[Suggestion]:
        debug = debug or {}
        base = self._rule_based_suggestions(scores, debug)

        # Rank by predicted overall lift
        base_sorted = sorted(
            base,
            key=lambda s: (
                s.expected_score_lift.get("overall", 0.0),
                s.expected_score_lift.get(self._best_dim(scores), 0.0),
            ),
            reverse=True,
        )
        base_sorted = base_sorted[: self.max_suggestions]

        # Add Amazon links for paid suggestions (Option A)
        base_sorted = self._attach_amazon_links(base_sorted)

        if not self.use_openai:
            return base_sorted

        # Refine with OpenAI: rewrite + optionally adjust Amazon link queries for paid items
        try:
            refined = self._openai_refine(scores, debug, base_sorted)
            # Ensure links exist for paid items even if the model omitted them
            refined = self._attach_amazon_links(refined or base_sorted)
            return refined or base_sorted
        except Exception:
            return base_sorted

    def split_free_paid(self, suggestions: List[Suggestion]) -> Tuple[List[Suggestion], List[Suggestion]]:
        free = [s for s in suggestions if s.cost == "free"]
        paid = [s for s in suggestions if s.cost != "free"]
        return free, paid

    # ---------- helpers ----------

    def _best_dim(self, scores: CatEyeScores) -> str:
        dims = {
            "vertical_opportunity": scores.vertical_opportunity,
            "shelter_hiding": scores.shelter_hiding,
            "cozy_warmth": scores.cozy_warmth,
            "exploration_richness": scores.exploration_richness,
            "safety_low_threat": scores.safety_low_threat,
        }
        return min(dims.items(), key=lambda kv: kv[1])[0]

    def _amazon_search_url(self, query: str) -> str:
        q = urllib.parse.quote_plus(query)
        # Basic search URL
        url = f"https://{self.amazon_domain}/s?k={q}"
        # Optional affiliate tag
        if self.amazon_tag:
            url += f"&tag={urllib.parse.quote_plus(self.amazon_tag)}"
        return url

    def _attach_amazon_links(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """
        For Option A, we attach Amazon search links to paid suggestions using a curated mapping.
        If OpenAI already provided amazon_links, we keep them and only fill missing ones.
        """
        for s in suggestions:
            if s.cost == "free":
                s.amazon_links = []
                continue

            # Already has links? keep them, but ensure URLs exist
            if s.amazon_links:
                fixed: List[AmazonLink] = []
                for link in s.amazon_links:
                    url = link.url or self._amazon_search_url(link.query or link.label)
                    fixed.append(AmazonLink(label=link.label, query=link.query, url=url))
                s.amazon_links = fixed
                continue

            # Otherwise attach defaults based on category/title
            defaults = self._default_paid_queries_for(s)
            s.amazon_links = [
                AmazonLink(label=label, query=query, url=self._amazon_search_url(query))
                for (label, query) in defaults
            ]
        return suggestions

    def _default_paid_queries_for(self, s: Suggestion) -> List[Tuple[str, str]]:
        """
        Tenant-friendly, small-item defaults only.
        Keep it short (2–3 links).
        """
        title = (s.title or "").lower()
        cat = s.category

        if cat == "vertical":
            return [
                ("Non-slip mat", "non slip pet mat"),
                ("Suction-cup window perch", "cat window perch suction cups"),
                ("Cat steps (foam)", "foam pet stairs small"),
            ]

        if cat == "shelter":
            return [
                ("Foldable cat tunnel", "foldable cat tunnel"),
                ("Covered cat bed", "covered cat bed cave"),
                ("Cardboard cat house", "cardboard cat house hideout"),
            ]

        if cat == "cozy":
            return [
                ("Soft throw blanket", "soft throw blanket"),
                ("Washable pet mat", "washable pet mat"),
                ("Self-warming cat pad", "self warming cat pad"),
            ]

        if cat == "exploration":
            return [
                ("Wand toy", "cat wand toy"),
                ("Crinkle balls", "cat crinkle balls"),
                ("Catnip toys", "catnip toys variety pack"),
            ]

        # safety / default
        return [
            ("Cable organizer", "cable management sleeves"),
            ("Non-slip rug pad", "non slip rug pad"),
            ("Feliway diffuser", "feliway diffuser calming cats"),
        ]

    # ---------- deterministic suggestions ----------

    def _rule_based_suggestions(self, scores: CatEyeScores, debug: Dict[str, Any]) -> List[Suggestion]:
        counts: Dict[str, int] = debug.get("counts", {}) or {}
        edge_density: float = float(debug.get("edge_density", 0.0) or 0.0)
        warmth: float = float(debug.get("warmth_mean_r_minus_b", 0.0) or 0.0)
        nook_count: int = int(debug.get("nook_count", 0) or 0)
        shelter_area: float = float(debug.get("shelter_area", 0.0) or 0.0)
        vertical_area: float = float(debug.get("vertical_area", 0.0) or 0.0)
        cozy_area: float = float(debug.get("cozy_area", 0.0) or 0.0)
        explore_area: float = float(debug.get("explore_area", 0.0) or 0.0)
        threat_count: int = int(debug.get("threat_count", 0) or 0)
        threat_area: float = float(debug.get("threat_area", 0.0) or 0.0)

        dims = [
            ("vertical", scores.vertical_opportunity),
            ("shelter", scores.shelter_hiding),
            ("cozy", scores.cozy_warmth),
            ("exploration", scores.exploration_richness),
            ("safety", scores.safety_low_threat),
        ]
        dims_sorted = sorted(dims, key=lambda x: x[1])
        priority = {k: i for i, (k, _) in enumerate(dims_sorted)}

        suggestions: List[Suggestion] = []

        # Vertical
        if scores.vertical_opportunity < 0.65 or vertical_area < 0.06:
            suggestions.append(Suggestion(
                title="Create one ‘cat highway’ using existing furniture",
                why_it_helps="A simple step-up route boosts vertical opportunity without buying anything big.",
                effort="small",
                cost="free",
                category="vertical",
                steps=[
                    "Clear a route along sturdy surfaces (chair → couch arm → table edge).",
                    "Move a chair next to the safest step-up point (no wobble).",
                    "Keep landing zones uncluttered so jumps feel safe.",
                ],
                expected_score_lift={"vertical_opportunity": 0.06, "overall": 0.03},
            ))

        if counts.get("book", 0) > 0 or counts.get("tv", 0) > 0 or counts.get("laptop", 0) > 0:
            suggestions.append(Suggestion(
                title="Make a safe lookout perch (soft + non-slip)",
                why_it_helps="Cats love elevated perches; adding grip and softness makes it feel secure.",
                effort="tiny",
                cost="low",
                category="vertical",
                steps=[
                    "Pick an elevated surface your cat already approaches.",
                    "Add a folded towel or small non-slip mat (no adhesives needed).",
                    "Keep one side as a clear escape route (cats prefer 2 exits).",
                ],
                expected_score_lift={"vertical_opportunity": 0.04, "safety_low_threat": 0.02, "overall": 0.03},
            ))

        # Shelter
        if scores.shelter_hiding < 0.65 or (nook_count < 2 and shelter_area < 0.10):
            suggestions.append(Suggestion(
                title="Create a no-drill hiding nook (box or blanket tent)",
                why_it_helps="A simple ‘cave’ reduces stress and improves hiding/shelter comfort.",
                effort="small",
                cost="free",
                category="shelter",
                steps=[
                    "Use a chair + blanket to make a small tent (leave 2 openings).",
                    "Or slide a cardboard box under a table/desk and add an extra exit hole.",
                    "Place it away from the busiest walkway and loud devices.",
                ],
                expected_score_lift={"shelter_hiding": 0.08, "safety_low_threat": 0.03, "overall": 0.05},
            ))

        # Cozy
        if scores.cozy_warmth < 0.65 or (warmth < 0.03 and cozy_area < 0.12):
            suggestions.append(Suggestion(
                title="Set up one consistent ‘warm spot’",
                why_it_helps="A predictable cozy zone increases resting behavior and perceived comfort.",
                effort="tiny",
                cost="low",
                category="cozy",
                steps=[
                    "Pick the warmest spot (near window sun or away from drafts).",
                    "Add a folded throw/blanket or small pet mat on an existing surface.",
                    "Keep it consistent—avoid moving it daily.",
                ],
                expected_score_lift={"cozy_warmth": 0.07, "overall": 0.04},
            ))

        # Exploration
        if scores.exploration_richness < 0.65 or explore_area < 0.06:
            suggestions.append(Suggestion(
                title="Add 2–3 micro-enrichment items (rotate weekly)",
                why_it_helps="Small novelty boosts curiosity without turning the room into clutter.",
                effort="tiny",
                cost="low",
                category="exploration",
                steps=[
                    "Put out 1 wand toy OR 2 small toys (not a pile).",
                    "Rotate weekly (store the rest).",
                    "Place near the cat highway or cozy spot, not in the main walkway.",
                ],
                expected_score_lift={"exploration_richness": 0.06, "overall": 0.03},
            ))

        # Safety
        if scores.safety_low_threat < 0.70 or edge_density > 0.18 or threat_count > 0 or threat_area > 0.02:
            suggestions.append(Suggestion(
                title="Clear one ‘safe route’ by reducing visual clutter",
                why_it_helps="Clear routes make the space feel predictable and less threatening.",
                effort="small",
                cost="free",
                category="safety",
                steps=[
                    "Clear small items from the main cat path (especially near jump points).",
                    "Bundle cables and remove dangling objects from jump routes.",
                    "If reflections seem stressful, angle the mirror/TV slightly or cover part temporarily.",
                ],
                expected_score_lift={"safety_low_threat": 0.06, "vertical_opportunity": 0.02, "overall": 0.04},
            ))

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
          - optionally provide better Amazon search queries for paid items (Option A)
          - return strict JSON
        """
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
            "base_suggestions": [
                {
                    **s.__dict__,
                    "amazon_links": [
                        {"label": a.label, "query": a.query, "url": a.url}
                        for a in (s.amazon_links or [])
                    ],
                }
                for s in base
            ],
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
                    "cheap removable accessories (non-slip mat, suction-cup window perch)",
                    "steps that take under 30 minutes",
                ],
                "amazon_links_policy": "Provide Amazon SEARCH queries only; do not invent prices, ratings, or specific brands.",
                "output_limit": self.max_suggestions,
            },
        }

        system = (
            "You are an expert in cat-friendly interior tweaks. "
            "You must propose minimal, tenant-friendly changes to improve the given cat-attractiveness scores. "
            "If a suggestion is paid (cost != 'free'), include 1-3 Amazon SEARCH links (label+query). "
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
            response_format={"type": "json_schema", "json_schema": {"name": "cat_room_suggestions", "schema": schema}},
        )

        data = json.loads(resp.output_text)

        out: List[Suggestion] = []
        for item in data.get("suggestions", [])[: self.max_suggestions]:
            # Convert amazon_links (label+query) -> AmazonLink (with computed url)
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
def render_suggestions_streamlit(suggestions: List[Suggestion]) -> None:
    """
    Call this from your Streamlit app:
      free, paid = engine.split_free_paid(suggestions)
      render_suggestions_streamlit(free + paid)  # or render separately
    """
    import streamlit as st

    free = [s for s in suggestions if s.cost == "free"]
    paid = [s for s in suggestions if s.cost != "free"]

    st.subheader("Free changes (no purchases)")
    if not free:
        st.caption("No free changes found (this is rare).")
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
# scores, dbg = rate_cat_attractiveness_with_yolo("room.jpg", return_debug=True)
# engine = CatRoomSuggestionEngine(use_openai=True, amazon_tag=os.getenv("AMAZON_TAG"))
# suggestions = engine.suggest(scores, dbg)
# free, paid = engine.split_free_paid(suggestions)
# render_suggestions_streamlit(free + paid)
