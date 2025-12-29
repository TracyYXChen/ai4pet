from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("Install OpenCV: pip install opencv-python") from e

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("Install YOLO: pip install ultralytics") from e


# -------------------- Scores --------------------

@dataclass
class DogEyeScores:
    floor_play_space: float
    rest_cozy: float
    sniff_enrichment: float
    water_food_ready: float
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


# -------------------- Dog-relevant COCO mappings --------------------
# COCO names YOLOv8 uses: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
# We'll use proxies since COCO lacks "dog bed", "crate", "food bowl" explicitly.

REST_CLASSES = {
    "couch", "bed", "chair", "bench", "dining table",
}

SNIFF_CLASSES = {
    # interesting objects that often correlate with “stuff to investigate”
    "potted plant", "vase", "book", "teddy bear", "sports ball",
    "backpack", "handbag", "suitcase", "umbrella",
    "bottle", "cup", "remote", "cell phone", "clock",
}

WATER_FOOD_PROXY_CLASSES = {
    # COCO doesn't have "bowl"; use kitchen/drink proxies.
    "bottle", "cup", "sink", "refrigerator", "microwave", "oven",
}

# For safety, treat these as "social/chaos" sources in an indoor context
THREAT_CLASSES = {
    "person", "dog", "cat",
}

# For play space, these often occupy floor or create obstacles
FLOOR_OBSTACLE_CLASSES = {
    "chair", "bench", "dining table", "couch", "bed", "suitcase", "backpack",
}

CATEGORY_NAMES = {
    "floor": "Floor/Play Space",
    "rest": "Rest/Cozy",
    "sniff": "Sniff/Enrichment",
    "water_food": "Water/Food",
    "threat": "Threat",
}


def _get_object_category(class_name: str) -> Optional[str]:
    if class_name in FLOOR_OBSTACLE_CLASSES:
        return "floor"
    if class_name in REST_CLASSES:
        return "rest"
    if class_name in SNIFF_CLASSES:
        return "sniff"
    if class_name in WATER_FOOD_PROXY_CLASSES:
        return "water_food"
    if class_name in THREAT_CLASSES:
        return "threat"
    return None


@dataclass
class DetectedObject:
    class_name: str
    category: Optional[str]  # floor, rest, sniff, water_food, threat, or None
    bbox: Tuple[int, int, int, int]
    confidence: float


def _detect_objects(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 100,
) -> Dict[str, Any]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(img_rgb, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]

    names = res.names
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)

    counts: Dict[str, int] = {}
    area_frac: Dict[str, float] = {}
    detections: list[DetectedObject] = []

    if res.boxes is None or len(res.boxes) == 0:
        return {"counts": counts, "area_frac": area_frac, "detections": detections, "raw": res}

    xyxy = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cid, c in zip(xyxy, cls_ids, confs):
        name = names.get(int(cid), str(int(cid)))
        counts[name] = counts.get(name, 0) + 1
        box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_frac[name] = area_frac.get(name, 0.0) + float(box_area / img_area)

        detections.append(DetectedObject(
            class_name=name,
            category=_get_object_category(name),
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=float(c),
        ))

    return {"counts": counts, "area_frac": area_frac, "detections": detections, "raw": res}


def rate_dog_attractiveness_with_yolo(
    image_path: str,
    yolo_model: Optional[YOLO] = None,
    yolo_weights: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    return_debug: bool = False,
) -> Tuple[DogEyeScores, Dict[str, Any] | None]:
    """
    Dog-focused room attractiveness.

    Combines:
      - low-level heuristics (edges/entropy/warmth/brightness)
      - YOLO detections mapped to dog-relevant affordances (rest, sniff, obstacles, etc.)
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

    edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 140)
    edge_density = float(np.mean(edges > 0))

    hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [32], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-9)
    entropy = float(-(p * np.log2(p + 1e-9)).sum())

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    warmth = float(np.mean(rgb[..., 0] - rgb[..., 2]))
    mean_brightness = float(np.mean(gray))

    # Object detection
    model = yolo_model if yolo_model is not None else YOLO(yolo_weights)
    det = _detect_objects(model, img_bgr, conf=conf, iou=iou)
    counts = det["counts"]
    area_frac = det["area_frac"]

    def sum_area(class_set: set[str]) -> float:
        return float(sum(area_frac.get(k, 0.0) for k in class_set))

    def sum_count(class_set: set[str]) -> float:
        return float(sum(counts.get(k, 0) for k in class_set))

    rest_area = sum_area(REST_CLASSES)
    sniff_area = sum_area(SNIFF_CLASSES)
    water_food_area = sum_area(WATER_FOOD_PROXY_CLASSES)
    obstacle_area = sum_area(FLOOR_OBSTACLE_CLASSES)

    threat_area = sum_area(THREAT_CLASSES)
    threat_count = sum_count(THREAT_CLASSES)

    # -------------------- Dimension scoring --------------------

    # 1) Floor / play space: prefer low clutter + fewer obstacles
    # Use edge_density as clutter proxy; obstacle_area as furniture-on-floor proxy.
    clutter_penalty = _normalize(edge_density, 0.10, 0.30)
    obstacle_penalty = _normalize(obstacle_area, 0.05, 0.35)
    floor_play_space = _clamp01(1.0 - (0.60 * clutter_penalty + 0.40 * obstacle_penalty))

    # 2) Rest cozy: warmth + softness + presence of rest surfaces
    softness = 1.0 - _normalize(edge_energy, 0.02, 0.14)
    cozy_from_color = 0.55 * _normalize(warmth, 0.00, 0.18) + 0.45 * _clamp01(softness)
    cozy_from_det = _normalize(rest_area, 0.00, 0.40)
    rest_cozy = _clamp01(0.60 * cozy_from_color + 0.40 * cozy_from_det)

    # 3) Sniff enrichment: moderate complexity + interesting objects
    # Dogs like novelty, but too much chaos hurts. Aim for mid edge_density and mid entropy.
    richness_edges = 1.0 - abs(_normalize(edge_density, 0.02, 0.22) - 0.55) * 2.0
    richness_ent = 1.0 - abs(_normalize(entropy, 2.0, 5.0) - 0.60) * 2.0
    richness_heur = _clamp01(0.55 * _clamp01(richness_edges) + 0.45 * _clamp01(richness_ent))
    sniff_from_det = _normalize(sniff_area, 0.00, 0.28)
    sniff_enrichment = _clamp01(0.60 * richness_heur + 0.40 * sniff_from_det)

    # 4) Water/food readiness: proxies only (bottle/cup/sink/kitchen items)
    # This isn't perfect, but it gives a gentle signal.
    water_food_ready = _clamp01(
        0.60 * _normalize(water_food_area, 0.00, 0.18) +
        0.40 * _normalize(counts.get("bottle", 0) + counts.get("cup", 0), 0, 4)
    )

    # 5) Safety / low threat: brightness comfort + low clutter + fewer threats
    brightness_ok = 1.0 - abs(_normalize(mean_brightness, 0.20, 0.85) - 0.55) * 2.0
    threat_penalty = _clamp01(0.7 * _normalize(threat_area, 0.00, 0.30) + 0.3 * _normalize(threat_count, 0, 6))
    safety_low_threat = _clamp01(
        0.60 * _clamp01(brightness_ok) +
        0.25 * (1.0 - clutter_penalty) +
        0.15 * (1.0 - threat_penalty)
    )

    # Overall weights (dogs: floor space and safety are big)
    weights = {
        "floor_play_space": 0.24,
        "rest_cozy": 0.20,
        "sniff_enrichment": 0.18,
        "water_food_ready": 0.12,
        "safety_low_threat": 0.26,
    }
    overall = _clamp01(
        weights["floor_play_space"] * floor_play_space +
        weights["rest_cozy"] * rest_cozy +
        weights["sniff_enrichment"] * sniff_enrichment +
        weights["water_food_ready"] * water_food_ready +
        weights["safety_low_threat"] * safety_low_threat
    )

    scores = DogEyeScores(
        floor_play_space=floor_play_space,
        rest_cozy=rest_cozy,
        sniff_enrichment=sniff_enrichment,
        water_food_ready=water_food_ready,
        safety_low_threat=safety_low_threat,
        overall=overall,
    )

    dbg = None
    if return_debug:
        dbg = {
            "counts": counts,
            "area_frac": area_frac,
            "detections": det.get("detections", []),
            "warmth_mean_r_minus_b": warmth,
            "mean_brightness": mean_brightness,
            "edge_energy": edge_energy,
            "edge_density": edge_density,
            "entropy": entropy,
            "rest_area": rest_area,
            "sniff_area": sniff_area,
            "water_food_area": water_food_area,
            "obstacle_area": obstacle_area,
            "threat_area": threat_area,
            "threat_count": threat_count,
        }

    return scores, dbg


# Example usage:
# scores, dbg = rate_dog_attractiveness_with_yolo(
#     "room.jpg",
#     yolo_weights="yolov8n.pt",
#     conf=0.25,
#     iou=0.45,
#     return_debug=True
# )
# print(scores)
# print(dbg["counts"])
