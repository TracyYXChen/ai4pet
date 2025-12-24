from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("Install OpenCV: pip install opencv-python") from e

# YOLO hook (Ultralytics)
# pip install ultralytics
try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("Install YOLO: pip install ultralytics") from e


@dataclass
class CatEyeScores:
    vertical_opportunity: float
    shelter_hiding: float
    cozy_warmth: float
    exploration_richness: float
    safety_low_threat: float
    overall: float


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp01((x - lo) / (hi - lo))


def _resize_for_speed(img_bgr: np.ndarray, target: int = 960) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = target / max(h, w)
    if scale < 1.0:
        return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr


# --- Cat-relevant mapping from COCO classes to "affordances" ---
# COCO names YOLOv8 uses: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
# We'll keep it flexible: anything unknown just doesn't contribute.

COZY_CLASSES = {
    "couch", "bed", "chair", "dining table", "potted plant", "tv", "laptop", "keyboard", "book",
}
SHELTER_CLASSES = {
    "chair", "couch", "bed", "dining table", "bench", "suitcase", "backpack", "handbag",
}
VERTICAL_CLASSES = {
    "book", "tv", "laptop", "refrigerator", "microwave", "oven", "sink",
    # COCO doesn't include "shelf" or "window" unfortunately, but these proxies help a bit.
}
EXPLORATION_CLASSES = {
    "potted plant", "vase", "book", "sports ball", "bottle", "cup", "fork", "knife", "spoon",
    "remote", "mouse", "cell phone", "teddy bear", "clock",
}
THREAT_CLASSES = {
    "person", "dog", "cat",  # yes cat can be "social complexity" if multiple cats
}

# Category names for display
CATEGORY_NAMES = {
    "vertical": "Vertical",
    "shelter": "Shelter", 
    "cozy": "Cozy",
    "exploration": "Explore",
    "threat": "Threat",
}


def _get_object_category(class_name: str) -> Optional[str]:
    """Get the cat-relevant category for a detected object."""
    if class_name in VERTICAL_CLASSES:
        return "vertical"
    if class_name in SHELTER_CLASSES:
        return "shelter"
    if class_name in COZY_CLASSES:
        return "cozy"
    if class_name in EXPLORATION_CLASSES:
        return "exploration"
    if class_name in THREAT_CLASSES:
        return "threat"
    return None


@dataclass
class DetectedObject:
    """A detected object with bounding box and category information."""
    class_name: str
    category: Optional[str]  # One of: vertical, shelter, cozy, exploration, threat, or None
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float


def _detect_objects(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 100,
) -> Dict[str, Any]:
    """
    Detect objects in the image using the provided model.
    
    Returns:
      {
        "counts": {class_name: count},
        "area_frac": {class_name: sum(box_area)/image_area},
        "detections": List[DetectedObject] - all detected objects with boxes and categories,
        "raw": model result object (optional use)
      }
    """
    # Ultralytics expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(img_rgb, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]

    names = res.names  # id -> name
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)

    counts: Dict[str, int] = {}
    area_frac: Dict[str, float] = {}
    detections: list[DetectedObject] = []

    if res.boxes is None or len(res.boxes) == 0:
        return {"counts": counts, "area_frac": area_frac, "detections": detections, "raw": res}

    # xyxy in pixels
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cid, c in zip(xyxy, cls_ids, confs):
        name = names.get(int(cid), str(int(cid)))
        counts[name] = counts.get(name, 0) + 1
        box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_frac[name] = area_frac.get(name, 0.0) + float(box_area / img_area)
        
        # Create detection object with category
        category = _get_object_category(name)
        detections.append(DetectedObject(
            class_name=name,
            category=category,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=float(c),
        ))

    return {"counts": counts, "area_frac": area_frac, "detections": detections, "raw": res}


def rate_cat_attractiveness_with_yolo(
    image_path: str,
    yolo_model: Optional[YOLO] = None,
    yolo_weights: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    return_debug: bool = False,
) -> Tuple[CatEyeScores, Dict[str, Any] | None]:
    """
    Combines:
      - low-level heuristics (edges/entropy/warmth/dark nooks)
      - YOLO detections mapped to cat-relevant affordances
    """

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_bgr = _resize_for_speed(img_bgr, target=960)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Low-level features
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    edge_energy = float(np.mean(mag))
    vertical_ratio = float(np.mean(np.abs(gx) / (np.abs(gy) + 1e-6)))

    edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 140)
    edge_density = float(np.mean(edges > 0))

    hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [32], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-9)
    entropy = float(-(p * np.log2(p + 1e-9)).sum())

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    warmth = float(np.mean(rgb[..., 0] - rgb[..., 2]))
    mean_brightness = float(np.mean(gray))

    # Dark nook proxy
    dark = (gray < 0.25).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    area = gray.shape[0] * gray.shape[1]
    nook_areas = []
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        frac = a / area
        if 0.01 <= frac <= 0.20:
            nook_areas.append(frac)
    nook_count = len(nook_areas)
    nook_area_sum = float(sum(nook_areas))

    # Object detection
    model = yolo_model if yolo_model is not None else YOLO(yolo_weights)
    det = _detect_objects(model, img_bgr, conf=conf, iou=iou)
    counts = det["counts"]
    area_frac = det["area_frac"]

    # Aggregate affordance signals from detections
    def sum_area(class_set: set[str]) -> float:
        return float(sum(area_frac.get(k, 0.0) for k in class_set))

    def sum_count(class_set: set[str]) -> float:
        return float(sum(counts.get(k, 0) for k in class_set))

    cozy_area = sum_area(COZY_CLASSES)
    shelter_area = sum_area(SHELTER_CLASSES)
    vertical_area = sum_area(VERTICAL_CLASSES)
    explore_area = sum_area(EXPLORATION_CLASSES)
    threat_area = sum_area(THREAT_CLASSES)
    threat_count = sum_count(THREAT_CLASSES)

    # --- Dimension scoring (blend heuristic + detection) ---

    # 1) Vertical opportunity: edges + detected tall-ish objects
    vertical_from_edges = 0.6 * _normalize(vertical_ratio, 0.9, 2.0) + 0.4 * _normalize(edge_energy, 0.02, 0.10)
    vertical_from_det = _normalize(vertical_area, 0.00, 0.25)
    vertical_opportunity = _clamp01(0.65 * vertical_from_edges + 0.35 * vertical_from_det)

    # 2) Shelter/hiding: dark nooks + furniture presence
    shelter_from_nooks = 0.55 * _normalize(nook_count, 0, 6) + 0.45 * _normalize(nook_area_sum, 0.0, 0.30)
    shelter_from_det = _normalize(shelter_area, 0.00, 0.35)
    shelter_hiding = _clamp01(0.60 * shelter_from_nooks + 0.40 * shelter_from_det)

    # 3) Cozy warmth: warm tone + cozy objects + softness
    softness = 1.0 - _normalize(edge_energy, 0.02, 0.14)
    cozy_from_color = 0.55 * _normalize(warmth, 0.00, 0.18) + 0.45 * _clamp01(softness)
    cozy_from_det = _normalize(cozy_area, 0.00, 0.40)
    cozy_warmth = _clamp01(0.60 * cozy_from_color + 0.40 * cozy_from_det)

    # 4) Exploration richness: moderate complexity + interesting objects
    richness_edges = 1.0 - abs(_normalize(edge_density, 0.02, 0.20) - 0.55) * 2.0
    richness_ent = 1.0 - abs(_normalize(entropy, 2.0, 5.0) - 0.60) * 2.0
    richness_heur = _clamp01(0.55 * _clamp01(richness_edges) + 0.45 * _clamp01(richness_ent))
    richness_det = _normalize(explore_area, 0.00, 0.25)
    exploration_richness = _clamp01(0.65 * richness_heur + 0.35 * richness_det)

    # 5) Safety / low threat: brightness comfort + avoid clutter + reduce threats
    brightness_ok = 1.0 - abs(_normalize(mean_brightness, 0.20, 0.85) - 0.55) * 2.0
    clutter_penalty = _normalize(edge_density, 0.10, 0.30)
    threat_penalty = _clamp01(0.7 * _normalize(threat_area, 0.00, 0.30) + 0.3 * _normalize(threat_count, 0, 6))
    safety_low_threat = _clamp01(0.70 * _clamp01(brightness_ok) + 0.15 * (1.0 - clutter_penalty) + 0.15 * (1.0 - threat_penalty))

    # Overall
    weights = {
        "vertical_opportunity": 0.22,
        "shelter_hiding": 0.22,
        "cozy_warmth": 0.18,
        "exploration_richness": 0.18,
        "safety_low_threat": 0.20,
    }
    overall = _clamp01(
        weights["vertical_opportunity"] * vertical_opportunity
        + weights["shelter_hiding"] * shelter_hiding
        + weights["cozy_warmth"] * cozy_warmth
        + weights["exploration_richness"] * exploration_richness
        + weights["safety_low_threat"] * safety_low_threat
    )

    scores = CatEyeScores(
        vertical_opportunity=vertical_opportunity,
        shelter_hiding=shelter_hiding,
        cozy_warmth=cozy_warmth,
        exploration_richness=exploration_richness,
        safety_low_threat=safety_low_threat,
        overall=overall,
    )

    debug = None
    if return_debug:
        debug = {
            "counts": counts,
            "area_frac": area_frac,
            "detections": det.get("detections", []),  # List of DetectedObject
            "warmth_mean_r_minus_b": warmth,
            "mean_brightness": mean_brightness,
            "edge_energy": edge_energy,
            "vertical_ratio": vertical_ratio,
            "edge_density": edge_density,
            "entropy": entropy,
            "nook_count": nook_count,
            "nook_area_sum": nook_area_sum,
            "cozy_area": cozy_area,
            "shelter_area": shelter_area,
            "vertical_area": vertical_area,
            "explore_area": explore_area,
            "threat_area": threat_area,
            "threat_count": threat_count,
        }

    return scores, debug


# Example usage:
# scores, dbg = rate_cat_attractiveness_with_yolo(
#     "room.jpg",
#     yolo_weights="yolov8n.pt",  # or yolov8s.pt for better accuracy
#     conf=0.25,
#     iou=0.45,
#     return_debug=True
# )
# print(scores)
# print(dbg["counts"])
