import streamlit as st
import numpy as np
import cv2
import io
import yaml
from PIL import Image

from transformers import pipeline
from rate_cat_attractiveness import (
    rate_cat_attractiveness_with_yolo, 
    CatEyeScores, 
    DetectedObject,
    CATEGORY_NAMES,
)
from suggest_room_changes import CatRoomSuggestionEngine, Suggestion
from ultralytics import YOLO
from typing import Tuple, Dict, Any, Optional, List
import tempfile
import os

# 1. Load the AI Depth Model (Cached for speed)
@st.cache_resource
def load_depth_model():
    # 'depth-anything' is excellent, but 'vincent-cl/monodepth2-visdrone-v2' 
    # or the standard DPT are lighter. Let's use a standard robust one:
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    return pipe


# 2. Load the Object Detection Model (Cached for speed)
@st.cache_resource
def load_detector_model(weights: str = "yolov8n.pt"):
    """Load and cache the object detection model for cat attractiveness scoring."""
    return YOLO(weights)


# 3. Load config for API keys
@st.cache_resource
def load_config():
    """Load configuration from config.yaml file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# Category colors for bounding box visualization (BGR format for OpenCV)
CATEGORY_COLORS = {
    "vertical": (255, 165, 0),    # Orange
    "shelter": (147, 112, 219),   # Purple
    "cozy": (255, 192, 203),      # Pink
    "exploration": (50, 205, 50), # Lime green
    "threat": (0, 0, 255),        # Red
    None: (128, 128, 128),        # Gray for uncategorized
}


def draw_detections_on_image(
    img_rgb: np.ndarray,
    detections: List[DetectedObject],
) -> np.ndarray:
    """
    Draw bounding boxes with object names and category labels on the image.
    
    Args:
        img_rgb: RGB image as numpy array
        detections: List of DetectedObject with bbox and category info
        
    Returns:
        RGB image with annotations drawn
    """
    annotated = img_rgb.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        category = det.category
        
        # Get color for this category (convert BGR to RGB for display)
        color_bgr = CATEGORY_COLORS.get(category, CATEGORY_COLORS[None])
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 2)
        
        # Prepare label text
        if category:
            category_display = CATEGORY_NAMES.get(category, category.title())
            label = f"{det.class_name} [{category_display}]"
        else:
            label = det.class_name
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        label_y = max(y1 - 5, text_h + 5)
        cv2.rectangle(
            annotated, 
            (x1, label_y - text_h - 5), 
            (x1 + text_w + 4, label_y + 2), 
            color_rgb, 
            -1  # Filled
        )
        
        # Draw text (white on colored background)
        cv2.putText(
            annotated, 
            label, 
            (x1 + 2, label_y - 2), 
            font, 
            font_scale, 
            (255, 255, 255),  # White text
            thickness,
            cv2.LINE_AA
        )
    
    return annotated


def score_cat_attractiveness(
    image_input: str | np.ndarray,
    detector_model = None,
    detector_weights: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    return_debug: bool = False,
) -> Tuple[CatEyeScores, Optional[Dict[str, Any]]]:
    """
    Score how attractive a space is to a cat.
    
    Args:
        image_input: Either a file path (str) or a BGR numpy array
        detector_model: Optional pre-loaded detection model (for efficiency)
        detector_weights: Model weights file if model not provided
        conf: Detection confidence threshold
        iou: Detection IOU threshold
        return_debug: Whether to return debug information
        
    Returns:
        Tuple of (CatEyeScores, debug_dict or None)
        
        CatEyeScores contains:
        - vertical_opportunity: Score for climbing/perching spots (0-1)
        - shelter_hiding: Score for hiding spots and shelter (0-1)
        - cozy_warmth: Score for warm, cozy areas (0-1)
        - exploration_richness: Score for interesting objects to explore (0-1)
        - safety_low_threat: Score for safety/low threat level (0-1)
        - overall: Weighted overall attractiveness score (0-1)
        
        debug_dict (if return_debug=True) contains:
        - detections: List[DetectedObject] with bounding boxes and categories
        - counts: Dict of object counts
        - Various other analysis metrics
    """
    # If input is a numpy array, save to temp file since the underlying function expects a path
    if isinstance(image_input, np.ndarray):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image_input)
        try:
            scores, debug = rate_cat_attractiveness_with_yolo(
                image_path=tmp_path,
                yolo_model=detector_model,
                yolo_weights=detector_weights,
                conf=conf,
                iou=iou,
                return_debug=return_debug,
            )
        finally:
            os.unlink(tmp_path)  # Clean up temp file
    else:
        # Input is a file path
        scores, debug = rate_cat_attractiveness_with_yolo(
            image_path=image_input,
            yolo_model=detector_model,
            yolo_weights=detector_weights,
            conf=conf,
            iou=iou,
            return_debug=return_debug,
        )
    
    return scores, debug

# 2. The Geometric Perspective Warp (Approach 1: Keystone + Depth)
def apply_pet_perspective_warp(image_rgb, depth_map, mode="Cat", strength=1.0):
    h, w, _ = image_rgb.shape
    
    # 1. FIELD OF VIEW (FOV) SQUEEZE
    # Dogs have ~240°, Cats ~200°. We simulate this with Barrel Distortion.
    # Higher K = More "Fish-eye" / Wider peripheral view
    if mode == "Dog":
        k1 = -0.15 * strength  # Stronger barrel distortion for 240°
        zoom = 1.1             # Compensate for black borders
    else:
        k1 = -0.05 * strength  # Subtle barrel for 200°
        zoom = 1.02

    # Prepare the map for radial distortion
    dist_coeff = np.array([k1, 0, 0, 0], dtype=np.float32)
    cam_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
    
    # Generate the map for the fish-eye effect
    map_x, map_y = cv2.initUndistortRectifyMap(cam_matrix, dist_coeff, None, cam_matrix, (w, h), cv2.CV_32FC1)
    
    # Apply the FOV squeeze
    distorted_img = cv2.remap(image_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 2. KEYSTONE (The "Looking Up" shape)
    tilt = 0.4 * strength
    src_pts = np.float32([
        [0, 0], [w, 0], 
        [w * tilt, h], [w * (1-tilt), h]
    ])
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    keystone_image = cv2.warpPerspective(distorted_img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 3. DEPTH DISPLACEMENT (Parallax)
    depth = cv2.resize(np.array(depth_map), (w, h)).astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    shift_y = depth * (h * 0.1 * strength)
    
    map_y_new = (map_y - shift_y).astype(np.float32)
    map_x_new = map_x.astype(np.float32)
    
    final_image = cv2.remap(keystone_image, map_x_new, map_y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return cv2.resize(final_image, (w, h))

# pet_inerest,a Spectral Residual approach—a standard computer vision technique to find "surprising" or "novel" parts of an image that grab attention.
def analyze_pet_interest(img_bgr, mode="Cat"):
    """
    Analyzes visual interest using a robust fallback if cv2.saliency fails.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Try to use the specialized module
    if hasattr(cv2, 'saliency'):
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(gray)
            saliency_map = (saliency_map * 255).astype("uint8")
        except:
            # Fallback to manual saliency calculation
            saliency_map = cv2.GaussianBlur(gray, (5, 5), 0)
            saliency_map = cv2.absdiff(gray, saliency_map)
    else:
        # MANUAL SALIENCY: Detects high-contrast areas/edges
        # This simulates interest based on visual complexity
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        saliency_map = cv2.convertScaleAbs(cv2.addWeighted(cv2.absdiff(grad_x, 0), 0.5, cv2.absdiff(grad_y, 0), 0.5, 0))

    # Apply Species-Specific Weighting
    h, w = saliency_map.shape
    y_indices, _ = np.indices((h, w))
    
    # Weighting logic (Cats look up, Dogs look down)
    if mode == "Cat":
        weight_mask = np.clip(1.3 - (y_indices / h), 0.5, 1.5)
    else:
        weight_mask = np.clip(0.5 + (y_indices / h), 0.5, 1.5)
        
    weighted_saliency = (saliency_map * weight_mask).astype(np.uint8)
    heatmap = cv2.applyColorMap(weighted_saliency, cv2.COLORMAP_JET)
    _, _, _, max_loc = cv2.minMaxLoc(weighted_saliency)
    
    return heatmap, max_loc

st.set_page_config(page_title="Pet Vision Filters", layout="wide")
st.title("🐾 Pet Vision Camera Filters (Approx.)")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Vision mode", ["Cat", "Dog"], index=0)

    st.header("Tuning")
    blur_sigma = st.slider("Blur (acuity loss) sigma", 0.0, 6.0, 2.0, 0.1)

    # Color controls
    if mode == "Cat":
        default_desat = 0.50  # Cats see very washed out colors
    else:
        default_desat = 0.20  # Dogs see slightly more saturation
        
    desat = st.slider("Desaturation", 0.0, 0.8, default_desat, 0.01)

    # Low-light + grain controls
    lowlight_strength = st.slider("Low-light lift strength", 0.0, 1.5, 0.8, 0.05)
    noise_base = st.slider("Noise base amount", 0.0, 0.10, 0.02, 0.005)
    noise_extra = st.slider("Extra noise in dark scenes", 0.0, 0.20, 0.06, 0.005)

    # Motion controls
    enable_motion = st.checkbox("Enable motion pop", value=True)
    motion_gain = st.slider("Motion gain", 0.0, 10.0, 5.0, 0.5)
    motion_boost = st.slider("Motion boost amount", 0.0, 0.20, 0.05, 0.01)

    # Mode-specific controls
    if mode == "Dog":
        dog_strength = st.slider("Dog dichromacy strength", 0.0, 1.0, 0.85, 0.01)
    else:
        dog_strength = 0.85  # Default value for Cat mode (not used but needed for params)

st.session_state["vision_params"] = dict(
    mode=mode,
    blur_sigma=blur_sigma,
    desat=desat,
    lowlight_strength=lowlight_strength,
    noise_base=noise_base,
    noise_extra=noise_extra,
    enable_motion=enable_motion,
    motion_gain=motion_gain,
    motion_boost=motion_boost,
    dog_strength=dog_strength,
)

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def cat_color_transform(rgb: np.ndarray) -> np.ndarray:
    """
    Simulates Cat Vision (Protanopia-ish).
    Scientifically accurate: Cats cannot see Red. 
    Red objects appear dark (low luminance), not purple.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    # 1. Cats have very few cones for Red. Red light looks very dark.
    # We mix Red heavily into Green to simulate 'Yellow' but darken it.
    rg_mix = 0.1 * r + 0.9 * g 
    
    # 2. Output:
    # Red & Green channels become the mix (Yellowish/Grey)
    # Blue channel stays Blue
    r2 = rg_mix
    g2 = rg_mix
    b2 = b

    return np.stack([r2, g2, b2], axis=-1)

def dog_color_transform(rgb: np.ndarray, strength: float) -> np.ndarray:
    """
    Simulates Dog Vision (Deuteranopia-ish).
    Scientifically accurate: Dogs see Red and Green as the SAME color (Yellow).
    Red objects are bright yellow, not dark.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # 1. Dogs mix Red and Green equally to create 'Yellow'.
    # Unlike cats, Red objects retain their brightness.
    yellow_channel = 0.5 * r + 0.5 * g
    
    # 2. Dogs see: Yellow (R+G) and Blue.
    r_d = yellow_channel
    g_d = yellow_channel
    b_d = b 
    
    mapped = np.stack([r_d, g_d, b_d], axis=-1)

    # Blend with original to control strength
    return clamp01((1.0 - strength) * rgb + strength * mapped)

def apply_pet_filter_to_image(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply pet vision filter to a static image (no motion detection).
    Returns filtered BGR image.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 1) Color model
    if params["mode"] == "Cat":
        rgb2 = cat_color_transform(rgb)
    else:
        rgb2 = dog_color_transform(rgb, float(params["dog_strength"]))

    # Luma (for desat + low-light adaptation)
    luma = 0.2126 * rgb2[..., 0] + 0.7152 * rgb2[..., 1] + 0.0722 * rgb2[..., 2]

    # 2) Desaturation
    d = float(params["desat"])
    rgb2 = rgb2 * (1.0 - d) + luma[..., None] * d

    # 3) Reduced acuity blur
    sigma = float(params["blur_sigma"])
    if sigma > 0.0:
        rgb2 = cv2.GaussianBlur(rgb2, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 4) Low-light lift + adaptive noise
    avg_luma = float(luma.mean())
    lift = np.clip((0.55 - avg_luma) / 0.55, 0.0, 1.0) * float(params["lowlight_strength"])
    gamma = 1.0 - 0.35 * lift
    rgb2 = np.power(clamp01(rgb2), gamma)

    noise_amt = float(params["noise_base"]) + float(params["noise_extra"]) * lift
    if noise_amt > 0.0:
        noise = (np.random.rand(*luma.shape).astype(np.float32) - 0.5) * noise_amt
        rgb2 = clamp01(rgb2 + noise[..., None])

    # Note: Motion emphasis is skipped for static images

    out_rgb = (rgb2 * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr

st.subheader("Upload an Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Upload an image to apply the pet vision filter"
)

if uploaded_file is not None:
    # Load image (PIL = RGB)
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Get current params
    params = st.session_state.get("vision_params", {})

    # --- PROCESSING ---
    
    # 1. Biological Vision Simulation (Retinal only)
    # Apply color/blur filter to original BGR image
    bio_bgr = apply_pet_filter_to_image(img_bgr, params)
    bio_rgb = cv2.cvtColor(bio_bgr, cv2.COLOR_BGR2RGB)

    # Prepare for Geometric/Combined
    # We need the depth model for Approach 1
    with st.spinner("🤖 AI is analyzing depth and perspective..."):
        try:
            # Calculate Depth
            depth_pipe = load_depth_model()
            depth_result = depth_pipe(pil_image)
            depth_map = depth_result["depth"]

            # 2. Geometric Perspective (Physical Viewpoint)
            # NEW LOGIC: Determine height based on mode
            if params["mode"] == "Cat":
                warp_strength = 0.8  # Short (Strong angle)
            else:
                warp_strength = 0.5  # Tall (Mild angle)

            # Updated function call (make sure you renamed the function definition too!)
            geo_rgb = apply_pet_perspective_warp(img_rgb, depth_map, mode=params["mode"], strength=warp_strength)

            # 3. Complete Simulation (Combined)
            # Take the warped image (geo_rgb) and apply the biological filter
            geo_bgr = cv2.cvtColor(geo_rgb, cv2.COLOR_RGB2BGR)
            combined_bgr = apply_pet_filter_to_image(geo_bgr, params)
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

            # --- NEW: 4. Interest Analysis ---
            interest_heatmap_bgr, peak_loc = analyze_pet_interest(combined_bgr, mode=params["mode"])
            interest_heatmap_rgb = cv2.cvtColor(interest_heatmap_bgr, cv2.COLOR_BGR2RGB)

            # Create an overlay (Heatmap on top of filtered image)
            overlay_img = cv2.addWeighted(combined_rgb, 0.6, interest_heatmap_rgb, 0.4, 0)
            # Draw a target circle on the peak interest point
            cv2.circle(overlay_img, peak_loc, 20, (255, 255, 255), 3)

        except Exception as e:
            st.error(f"Error loading AI model: {e}")
            st.stop()
    
    # --- DISPLAY RESULTS ---
    st.write("### 1. The Main Comparison: Human vs. Cat")
    st.caption("See the difference between your reality and the cat's full experience.")

    # ROW 1: The Main Comparison (Original vs Complete)
    top_col1, top_col2 = st.columns(2)
    
    with top_col1:
        st.subheader("Human Reality")
        st.image(pil_image, use_container_width=True)
        st.info("**Human Body + Human Eyes**\n\nStandard standing height (approx. 1.7m) with trichromatic (3-color) sharp vision.")

    # Dynamic Text Variables
    current_mode = params["mode"]
    if current_mode == "Cat":
        body_desc = "Standing height approx. 20cm. Objects loom over you significantly."
    else:
        body_desc = "Standing height approx. 50cm. The angle is lower than human, but higher than a cat."

    with top_col2:
        st.subheader(f"{current_mode} Reality")
        st.image(combined_rgb, use_container_width=True)
        st.success(f"**{current_mode} Body + {current_mode} Eyes**\n\n{body_desc}")

    st.write("---")
    st.write("### 2. Why is it so different?")
    st.write("We break down the transformation into two key factors: Biology and Physics.")

    # ROW 2: The Breakdown (Bio vs Geo)
    bot_col1, bot_col2 = st.columns(2)

    with bot_col1:
        st.markdown("#### Factor A: Biology (The Eyes)")
        st.image(bio_rgb, use_container_width=True)
        st.warning(
            "**Retinal Processing Only**\n\n"
            "Even if a cat stood as tall as a human, the world would look like this. "
            "Cats are dichromatic (Red-Green colorblind) and have lower visual acuity (blurrier) "
            "to prioritize motion detection over detail."
        )
        
    with bot_col2:
        st.markdown("#### Factor B: Physics (The Body)")
        st.image(geo_rgb, use_container_width=True)
        st.warning(
            "**Physical Perspective Only**\n\n"
            f"If a human crawled at {current_mode} height, the world would look like this. "
            "Notice the 'Keystone Effect': because you are looking UP, vertical lines "
            "converge inward, making them feel taller and more imposing."
        )
    # ROW 3: Interest Analysis
    st.write("---")
    st.write("### 3. Visual Attention Analysis")
    st.caption(f"Where is a {current_mode} most likely to look?")

    col_int1, col_int2 = st.columns(2)

    with col_int1:
        st.image(overlay_img, use_container_width=True)
        st.info("**Attention Heatmap**\n\nBright red zones indicate high visual 'weight' based on contrast, shape, and height.")

    with col_int2:
        # Crop a small square around the peak interest point
        x, y = peak_loc
        # Ensure crop stays within image bounds
        x1, x2 = max(0, x-75), min(img_rgb.shape[1], x+75)
        y1, y2 = max(0, y-75), min(img_rgb.shape[0], y+75)
        crop = img_rgb[y1:y2, x1:x2]

        st.image(crop, caption="Primary Object of Interest (Human View)")
        st.warning(f"**Analysis:** Based on {current_mode} biology, this object is the most likely to grab their attention first.")

    # ROW 4: Cat Attractiveness Scoring (only for Cat mode)
    if current_mode == "Cat":
        st.write("---")
        st.write("### 4. 🐱 Cat Attractiveness Score")
        st.caption("How appealing is this space to a cat? AI analyzes objects and visual features.")
        
        with st.spinner("🔍 Analyzing space attractiveness for cats..."):
            try:
                detector_model = load_detector_model()
                cat_scores, debug_info = score_cat_attractiveness(
                    img_bgr, 
                    detector_model=detector_model,
                    return_debug=True
                )
                
                # Display overall score prominently
                overall_pct = int(cat_scores.overall * 100)
                if overall_pct >= 70:
                    score_color = "🟢"
                    score_msg = "Excellent! A cat would love this space."
                elif overall_pct >= 50:
                    score_color = "🟡"
                    score_msg = "Good. This space has nice cat-friendly features."
                elif overall_pct >= 30:
                    score_color = "🟠"
                    score_msg = "Okay. Some improvements could make it more cat-friendly."
                else:
                    score_color = "🔴"
                    score_msg = "Low. This space may not be very appealing to cats."
                
                st.metric(
                    label=f"{score_color} Overall Cat Attractiveness",
                    value=f"{overall_pct}%",
                    help=score_msg
                )
                st.caption(score_msg)
                
                # Display individual dimension scores
                score_cols = st.columns(5)
                
                dimensions = [
                    ("🧗 Vertical", cat_scores.vertical_opportunity, "Climbing & perching opportunities"),
                    ("🏠 Shelter", cat_scores.shelter_hiding, "Hiding spots & enclosed spaces"),
                    ("☀️ Cozy", cat_scores.cozy_warmth, "Warm & comfortable areas"),
                    ("🎯 Explore", cat_scores.exploration_richness, "Interesting objects to investigate"),
                    ("🛡️ Safety", cat_scores.safety_low_threat, "Low threat level & security"),
                ]
                
                for col, (label, score, tooltip) in zip(score_cols, dimensions):
                    with col:
                        pct = int(score * 100)
                        st.metric(label=label, value=f"{pct}%", help=tooltip)
                
                # Show annotated image with bounding boxes and detected objects
                detections = debug_info.get("detections", []) if debug_info else []
                
                if detections:
                    st.write("#### Detected Objects")
                    
                    # Draw bounding boxes on original image
                    annotated_img = draw_detections_on_image(img_rgb, detections)
                    st.image(annotated_img, use_container_width=True)
                    
                    # Legend for categories
                    st.caption("**Category Legend:** 🧗 Vertical (Orange) | 🏠 Shelter (Purple) | ☀️ Cozy (Pink) | 🎯 Explore (Green) | ⚠️ Threat (Red) | Uncategorized (Gray)")
                    
                    # Show object summary in expander
                    with st.expander("🔎 Detection Details"):
                        # Group by category
                        by_category: Dict[str, List[str]] = {}
                        for det in detections:
                            cat_key = det.category or "uncategorized"
                            if cat_key not in by_category:
                                by_category[cat_key] = []
                            by_category[cat_key].append(det.class_name)
                        
                        for cat_key, objects in by_category.items():
                            cat_display = CATEGORY_NAMES.get(cat_key, cat_key.title())
                            obj_counts = {}
                            for obj in objects:
                                obj_counts[obj] = obj_counts.get(obj, 0) + 1
                            obj_str = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                            st.write(f"**{cat_display}:** {obj_str}")
                else:
                    st.info("No objects detected in this image.")
                
                # ROW 5: Room Improvement Suggestions
                st.write("---")
                st.write("### 5. 💡 Room Improvement Suggestions")
                st.caption("Tenant-friendly tips to make this space more cat-attractive.")
                
                with st.spinner("🤔 Generating personalized suggestions..."):
                    try:
                        config = load_config()
                        openai_api_key = config.get("openai_api_key")
                        
                        suggestion_engine = CatRoomSuggestionEngine(
                            openai_api_key=openai_api_key,
                            max_suggestions=5,
                            use_openai=bool(openai_api_key),
                        )
                        
                        suggestions = suggestion_engine.suggest(cat_scores, debug_info)
                        
                        if suggestions:
                            for i, sug in enumerate(suggestions, 1):
                                # Category emoji mapping
                                cat_emoji = {
                                    "vertical": "🧗",
                                    "shelter": "🏠",
                                    "cozy": "☀️",
                                    "exploration": "🎯",
                                    "safety": "🛡️",
                                }.get(sug.category, "💡")
                                
                                # Effort/cost badges
                                effort_badge = {"tiny": "⚡", "small": "🔧", "medium": "🔨"}.get(sug.effort, "")
                                cost_badge = {"free": "🆓", "low": "💵", "medium": "💰"}.get(sug.cost, "")
                                
                                with st.expander(f"{cat_emoji} **{sug.title}** {effort_badge}{cost_badge}", expanded=(i == 1)):
                                    st.markdown(f"**Why it helps:** {sug.why_it_helps}")
                                    st.markdown("**Steps:**")
                                    for step_num, step in enumerate(sug.steps, 1):
                                        st.markdown(f"{step_num}. {step}")
                                    
                                    # Show expected improvements
                                    if sug.expected_score_lift:
                                        lift_parts = []
                                        for dim, lift in sug.expected_score_lift.items():
                                            dim_display = dim.replace("_", " ").title()
                                            lift_parts.append(f"{dim_display}: +{int(lift * 100)}%")
                                        st.caption(f"📈 Expected improvement: {', '.join(lift_parts)}")
                                    
                                    st.caption(f"Effort: {sug.effort.title()} | Cost: {sug.cost.title()}")
                        else:
                            st.success("🎉 This space already looks great for cats! No major improvements needed.")
                            
                    except Exception as e:
                        st.warning(f"Could not generate suggestions: {e}")
                            
            except Exception as e:
                st.error(f"Error analyzing cat attractiveness: {e}")

    # --- DOWNLOADS ---
    st.write("---")
    st.markdown("##### Download Results")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)

    # Helper to create download buttons
    def create_download(image_array):
        pil_img = Image.fromarray(image_array)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    with d_col1:
        st.download_button("⬇️ Human (Orig)", create_download(img_rgb), "human_view.png", "image/png")
    with d_col2:
        st.download_button("⬇️ Factor A (Bio)", create_download(bio_rgb), "bio_view.png", "image/png")
    with d_col3:
        st.download_button("⬇️ Factor B (Geo)", create_download(geo_rgb), "geo_view.png", "image/png")
    with d_col4:
        st.download_button("⬇️ Cat (Complete)", create_download(combined_rgb), "cat_complete.png", "image/png")

else:
    st.info("👆 Upload an image above to see the 3-stage simulation!")