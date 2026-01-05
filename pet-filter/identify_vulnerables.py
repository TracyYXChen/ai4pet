from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict, Any
import base64
import io
import json

from PIL import Image

RiskType = Literal["breakable", "scratchable", "chewable", "toxic", "unstable"]
Severity = Literal["low", "medium", "high"]

@dataclass
class VulnerableObject:
    label: str
    risk_type: RiskType
    severity: Severity
    bbox: Optional[Tuple[float, float, float, float]]  # Optional normalized (x1,y1,x2,y2) in [0..1]
    reasons: List[str]
    quick_fixes_free: List[str]
    quick_fixes_paid: List[str]


class PetVulnerabilityAnalyzer:
    """
    OpenAI-only room hazard detector for pets (dogs/cats).
    - No YOLO.
    - Model returns structured JSON: vulnerable objects + fixes.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_objects: int = 15,
    ) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_objects = max_objects

    @staticmethod
    def _image_to_data_url(img: Image.Image, *, max_side: int = 1024, jpeg_quality: int = 85) -> str:
        """
        Downscale to keep latency/cost reasonable, encode as data URL.
        """
        img = img.convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def analyze(
        self,
        img: Image.Image,
        *,
        pet_type: Literal["dog", "cat", "both"] = "both",
        renter_mode: bool = True,
    ) -> Tuple[str, Dict[str, List[VulnerableObject]], str]:
        """
        Analyze image for vulnerabilities. Returns (description, structured_vulnerabilities, raw_response).
        structured_vulnerabilities is a dict with keys: breakable, scratchable, chewable, toxic, unstable
        raw_response is the raw JSON string from the API for debugging
        """
        data_url = self._image_to_data_url(img)

        # Step 1: Generate image description
        descr_resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a detailed room observer. Describe the room in the image comprehensively, including all visible objects, furniture, decorations, and potential items that could interact with pets."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Provide a detailed description of this room image. List all visible objects, furniture, decorations, plants, electronics, and any items that could be relevant for pet safety."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=1000,
        )
        image_description = descr_resp.choices[0].message.content

        # Step 2: Analyze for vulnerabilities with more aggressive prompt
        system = (
            "You are a thorough pet-safety home inspector. "
            "Your job is to identify ALL potential hazards and vulnerable objects in the room, organized by risk type. "
            "Be PROACTIVE and find at least 5-10 items across different risk categories. Even if something seems minor, include it. "
            "Analyze based on the provided image description - you do NOT need to provide bounding boxes. "
            "Return ONLY JSON with objects grouped by risk type. "
            "When in doubt, include the item with appropriate severity (prefer 'low' or 'medium' if uncertain)."
        )

        pet_specific_notes = {
            "cat": "Cats climb, scratch, and knock things over. Look for items on shelves, tables, and high surfaces. Consider scratching surfaces and items that could fall.",
            "dog": "Dogs chew, knock things over, and may ingest items. Look for items at ground level, cables, small objects, and anything chewable.",
            "both": "Consider both cat and dog behaviors: climbing, scratching, chewing, knocking things over, and ingesting items.",
        }

        user_payload = {
            "pet_type": pet_type,
            "renter_mode": renter_mode,
            "max_objects": self.max_objects,
            "image_description": image_description,
            "instructions": [
                f"Pet type: {pet_type}. {pet_specific_notes.get(pet_type, pet_specific_notes['both'])}",
                "IMPORTANT: Find at least 5-10 vulnerable items across different risk categories. Be thorough and include even minor risks.",
                "",
                "Organize findings by risk type (return JSON with these exact keys):",
                "- breakable: Fragile items (glass, ceramic, vases, picture frames, decorative items, electronics on edges)",
                "- scratchable: Items that can be scratched (upholstery, curtains, rugs, furniture, walls, door frames)",
                "- chewable: Items that can be chewed (cables, wires, small objects, plants, toys, shoes, books, furniture corners)",
                "- toxic: Poisonous items (toxic plants, cleaning chemicals, medications, certain foods, essential oils)",
                "- unstable: Items that could fall or tip over (items on edges, wobbly furniture, items on high shelves, leaning objects, top-heavy items)",
                "",
                "For each item, provide: specific label, severity (low/medium/high), 2-3 reasons why it's a concern, and practical fixes.",
                "Prefer minimal renter-friendly fixes: rearrange, move higher, add small mats/bins/covers; avoid drilling or big furniture.",
                "Do not invent brand names, prices, or claims.",
            ],
        }

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Based on this room description and image, identify ALL vulnerable objects and hazards:\n" + json.dumps(user_payload, indent=2)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
        )

        raw_response = resp.choices[0].message.content
        print("=" * 80)
        print("RAW API RESPONSE:")
        print("=" * 80)
        print(raw_response)
        print("=" * 80)
        
        data = json.loads(raw_response)
        
        # Debug: Print parsed data structure
        print("\nPARSED DATA STRUCTURE:")
        print(f"Top-level keys: {list(data.keys())}")
        for key in data.keys():
            if isinstance(data[key], list):
                print(f"  {key}: list with {len(data[key])} items")
                if len(data[key]) > 0:
                    print(f"    First item keys: {list(data[key][0].keys())}")
                    print(f"    First item: {data[key][0]}")
            else:
                print(f"  {key}: {type(data[key])} = {data[key]}")
        print("=" * 80)
        
        # Convert structured JSON to Dict[str, List[VulnerableObject]]
        structured_vulns: Dict[str, List[VulnerableObject]] = {
            "breakable": [],
            "scratchable": [],
            "chewable": [],
            "toxic": [],
            "unstable": [],
        }
        
        # Process each risk type category
        for risk_type in structured_vulns.keys():
            items = data.get(risk_type, [])
            if not isinstance(items, list):
                print(f"Warning: {risk_type} is not a list, got {type(items)}: {items}")
                continue
                
            for item in items[: self.max_objects]:
                if not isinstance(item, dict):
                    print(f"Warning: item in {risk_type} is not a dict, got {type(item)}: {item}")
                    continue
                    
                # Debug: Print item structure
                print(f"\nProcessing item in {risk_type}:")
                print(f"  Item keys: {list(item.keys())}")
                print(f"  Item: {item}")
                
                # Bbox is optional - set to None since we're not asking for it
                bbox = None

                # Use .get() with defaults to handle missing fields gracefully
                try:
                    structured_vulns[risk_type].append(
                        VulnerableObject(
                            label=item.get("label", "Unknown"),
                            risk_type=risk_type,  # type: ignore - Set from the key, guaranteed to be valid RiskType
                            severity=item.get("severity", "low"),
                            bbox=bbox,
                            reasons=item.get("reasons", []),
                            quick_fixes_free=item.get("quick_fixes_free", []),
                            quick_fixes_paid=item.get("quick_fixes_paid", []),
                        )
                    )
                except Exception as e:
                    print(f"Error creating VulnerableObject: {e}")
                    print(f"  Item was: {item}")
                    raise
        
        return image_description, structured_vulns, raw_response
