import streamlit as st
import numpy as np
import cv2
import io
import yaml
import os
from PIL import Image

from transformers import pipeline
# Cat modules
from rate_cat_attractiveness import (
    rate_cat_attractiveness_with_yolo, 
    CatEyeScores, 
    DetectedObject as CatDetectedObject,
    CATEGORY_NAMES as CAT_CATEGORY_NAMES,
)
from suggest_cat_room_changes import CatRoomSuggestionEngine, Suggestion as CatSuggestion, AmazonLink

# Dog modules
from rate_dog_attractiveness import (
    rate_dog_attractiveness_with_yolo,
    DogEyeScores,
    DetectedObject as DogDetectedObject,
    CATEGORY_NAMES as DOG_CATEGORY_NAMES,
)
from suggest_dog_room_changes import DogRoomSuggestionEngine, Suggestion as DogSuggestion
from identify_vulnerables import PetVulnerabilityAnalyzer, VulnerableObject
from ultralytics import YOLO
from typing import Tuple, Dict, Any, Optional, List
import tempfile
import os

# 1. Load the AI Depth Model (Cached for speed)
@st.cache_resource
def load_depth_model():
    # Adding device=0 or device_map="auto" usually triggers the need for accelerate.
    # On Streamlit Cloud (CPU), the safest way is explicitly setting device="cpu".
    pipe = pipeline(
        task="depth-estimation", 
        model="LiheYoung/depth-anything-small-hf", 
        device="cpu"
    )
    return pipe


# 2. Load the Object Detection Model (Cached for speed)
@st.cache_resource
def load_detector_model():
    # Load model and immediately put it in evaluation mode to save memory
    model = YOLO("yolov8n.pt")
    return model



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


# 4. Generate suggestion diagram using DALL-E (cached)
# Increment DIAGRAM_PROMPT_VERSION when you change the prompt to bust the cache
DIAGRAM_PROMPT_VERSION = 2

def generate_suggestion_diagram(steps: tuple, openai_api_key: str, _version: int = DIAGRAM_PROMPT_VERSION) -> tuple[Optional[str], Optional[str]]:
    """
    Generate a single illustrated diagram for an entire suggestion using DALL-E.
    Returns a tuple of (image_url, error_message). If successful, error is None.
    """
    try:
        from openai import OpenAI
        
        if not openai_api_key:
            return None, "No OpenAI API key configured"
        
        client = OpenAI(api_key=openai_api_key)
        
        # Combine steps into a summary
        steps_summary = " → ".join([f"{s}" for s in steps])
        
        # Create a prompt for a single comprehensive diagram
        prompt = (
            f"You are a skilled illustrator and your task is to help users follow steps to improve the attractiveness of a room for cats. Create a detailed diagram based on the following steps: {steps_summary}. The diagram should be a single image that shows the entire suggestion."
        )
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        return response.data[0].url, None
    except Exception as e:
        return None, str(e)


# Category colors for bounding box visualization (BGR format for OpenCV)
CATEGORY_COLORS = {
    "vertical": (255, 165, 0),    # Orange
    "shelter": (147, 112, 219),   # Purple
    "cozy": (255, 192, 203),      # Pink
    "exploration": (50, 205, 50), # Lime green
    "threat": (0, 0, 255),        # Red
    None: (128, 128, 128),        # Gray for uncategorized
}


def scale_bboxes_to_original_size(
    detections: List[Any],
    original_h: int,
    original_w: int,
    resized_h: int,
    resized_w: int,
) -> List[Any]:
    """
    Scale bounding box coordinates from resized image back to original image size.
    
    Args:
        detections: List of DetectedObject with bbox coordinates in resized image space
        original_h: Original image height
        original_w: Original image width
        resized_h: Resized image height (used for detection)
        resized_w: Resized image width (used for detection)
        
    Returns:
        List of DetectedObject with bbox coordinates scaled to original image size
    """
    from dataclasses import replace
    
    scale_x = original_w / resized_w
    scale_y = original_h / resized_h
    
    scaled_detections = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        scaled_bbox = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )
        # Create a new DetectedObject with scaled bbox using dataclasses.replace
        scaled_det = replace(det, bbox=scaled_bbox)
        scaled_detections.append(scaled_det)
    
    return scaled_detections


def draw_detections_on_image(
    img_rgb: np.ndarray,
    detections: List[Any],
    category_names: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes with object names and category labels on the image.
    
    Args:
        img_rgb: RGB image as numpy array
        detections: List of DetectedObject with bbox and category info
        category_names: Dict mapping category keys to display names
        
    Returns:
        RGB image with annotations drawn
    """
    annotated = img_rgb.copy()
    category_names = category_names or {}
    
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
            category_display = category_names.get(category, category.title())
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


def score_dog_attractiveness(
    image_input: str | np.ndarray,
    detector_model = None,
    detector_weights: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    return_debug: bool = False,
) -> Tuple[DogEyeScores, Optional[Dict[str, Any]]]:
    """
    Score how attractive a space is to a dog.
    
    Args:
        image_input: Either a file path (str) or a BGR numpy array
        detector_model: Optional pre-loaded detection model (for efficiency)
        detector_weights: Model weights file if model not provided
        conf: Detection confidence threshold
        iou: Detection IOU threshold
        return_debug: Whether to return debug information
        
    Returns:
        Tuple of (DogEyeScores, debug_dict or None)
        
        DogEyeScores contains:
        - floor_play_space: Score for open floor/play space (0-1)
        - rest_cozy: Score for resting/cozy spots (0-1)
        - sniff_enrichment: Score for sniffing/enrichment opportunities (0-1)
        - water_food_ready: Score for water/food accessibility (0-1)
        - safety_low_threat: Score for safety/low threat level (0-1)
        - overall: Weighted overall attractiveness score (0-1)
    """
    # If input is a numpy array, save to temp file since the underlying function expects a path
    if isinstance(image_input, np.ndarray):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image_input)
        try:
            scores, debug = rate_dog_attractiveness_with_yolo(
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
        scores, debug = rate_dog_attractiveness_with_yolo(
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
    Analyzes visual interest with species-specific biases:
    - Cat: Bias toward the upper half (looking up) and center-vertical (depth).
    - Dog: Bias toward a wide horizontal band (left/right).
    - Human: Bias toward the golden ratio intersections.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Base Saliency (Edges/Contrast)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    saliency_map = cv2.convertScaleAbs(cv2.addWeighted(cv2.absdiff(grad_x, 0), 0.5, cv2.absdiff(grad_y, 0), 0.5, 0))

    # 2. Create Biological Bias Masks
    y_idx, x_idx = np.indices((h, w))
    weight_mask = np.ones((h, w), dtype=np.float32)

    if mode == "Cat":
        # Bias: Top of image (looking up) + center-vertical (depth)
        vertical_bias = np.clip(1.5 - (y_idx / (h * 0.5)), 0.5, 2.0)
        depth_bias = np.exp(-((x_idx - w/2)**2) / (2 * (w/4)**2))
        weight_mask = vertical_bias * depth_bias

    elif mode == "Dog":
        # Bias: Middle horizontal band (scanning left and right)
        horizontal_band = np.exp(-((y_idx - h/1.8)**2) / (2 * (h/4)**2))
        weight_mask = horizontal_band * 1.5

    else: # Human (Standard/Golden Ratio)
        # Bias: Golden Ratio point (~0.618)
        phi = 0.618
        golden_y, golden_x = int(h * (1 - phi)), int(w * phi)
        weight_mask = np.exp(-((y_idx - golden_y)**2 + (x_idx - golden_x)**2) / (2 * (w/6)**2))
        weight_mask = np.clip(weight_mask * 2.0, 0.8, 2.0)

    # 3. Combine Saliency with Biological Bias
    weighted_saliency = (saliency_map.astype(np.float32) * weight_mask)
    weighted_saliency = cv2.normalize(weighted_saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
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

# --- INITIALIZE SESSION STATE ---
if 'selected_sample' not in st.session_state:
    st.session_state['selected_sample'] = None

st.markdown("### 🖼️ Step 1: Choose or Upload an Image")

# --- CLICKABLE GALLERY ---
st.write("##### Quick Select Samples")
# FIXED: Changed .png to match your files in the samples/ folder
samples = {
    "Sample 1": "samples/sample1.png",
    "Sample 2": "samples/sample2.png",
}

cols = st.columns(4)
for i, (name, path) in enumerate(samples.items()):
    with cols[i]:
        if os.path.exists(path):
            st.image(path, use_container_width=True)
            if st.button(f"Select {i+1}", key=f"btn_{i}"):
                st.session_state['selected_sample'] = path
                # Use a dummy key to reset the file uploader if a sample is picked
                st.session_state['uploader_key'] = np.random.randint(1, 1000) 
        else:
            # Displays if file extension or path is wrong
            st.error(f"Path error: {path}")

st.write("---")

# --- UPLOAD OPTION ---
st.write("##### Or Upload Your Own")
uploaded_file = st.file_uploader(
    "Drag and drop file here", 
    type=["jpg", "jpeg", "png"],
    key=st.session_state.get('uploader_key', 'user_upload')
)

# --- LOGIC TO DETERMINE WHICH IMAGE TO USE ---
input_image = None

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.session_state['selected_sample'] = None 
elif st.session_state['selected_sample'] is not None:
    input_image = Image.open(st.session_state['selected_sample'])

# --- MAIN PROCESSING BLOCK ---
# Wrap EVERYTHING in this if-statement to prevent NameErrors
if input_image:
    st.info("✅ Image selected successfully!")
    
    # 1. Prepare images for OpenCV
    pil_image = input_image.convert("RGB")
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 2. Get current params
    params = st.session_state.get("vision_params", {})

    # --- SIMULATION STEPS ---
    
    # Biological Vision Simulation
    # Now img_bgr is guaranteed to exist because we are inside the 'if input_image' block
    bio_bgr = apply_pet_filter_to_image(img_bgr, params)
    bio_rgb = cv2.cvtColor(bio_bgr, cv2.COLOR_BGR2RGB)

    # --- IMPORTANT: Indent all your remaining code (Depth, Warp, Display) ---
    # From here, indent all your remaining lines until the end of the file 
    # so they only run when 'input_image' is present.

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
            
            # --- 4. Interest Analysis ---
            # 1. Find where the PET is looking (Cat or Dog bias) for the Heatmap
            interest_heatmap_bgr, pet_peak_loc = analyze_pet_interest(img_bgr, mode=params["mode"])

            # 2. Find where a HUMAN is looking (Golden Ratio bias) for the comparison
            _, human_peak_loc = analyze_pet_interest(img_bgr, mode="Human")

            # Prepare heatmap for display
            interest_heatmap_rgb = cv2.cvtColor(interest_heatmap_bgr, cv2.COLOR_BGR2RGB)
            
            # Create the overlay (Heatmap on top of filtered image)
            overlay_img = cv2.addWeighted(combined_rgb, 0.6, interest_heatmap_rgb, 0.4, 0)
            
            # Draw a target circle on the PET's specific peak interest point
            cv2.circle(overlay_img, pet_peak_loc, 20, (255, 255, 255), 3)

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
        st.image(pil_image, width="stretch")
        st.info("**Human Body + Human Eyes**\n\nStandard standing height (approx. 1.7m) with trichromatic (3-color) sharp vision.")

    # Dynamic Text Variables
    current_mode = params["mode"]
    if current_mode == "Cat":
        body_desc = "Standing height approx. 20cm. Objects loom over you significantly."
    else:
        body_desc = "Standing height approx. 50cm. The angle is lower than human, but higher than a cat."

    with top_col2:
        st.subheader(f"{current_mode} Reality")
        st.image(combined_rgb, width="stretch")
        st.success(f"**{current_mode} Body + {current_mode} Eyes**\n\n{body_desc}")

    st.write("---")
    st.write("### 2. Why is it so different?")
    st.write("We break down the transformation into two key factors: Biology and Physics.")

    # ROW 2: The Breakdown (Bio vs Geo)
    bot_col1, bot_col2 = st.columns(2)

    with bot_col1:
        st.markdown("#### Factor A: Biology (The Eyes)")
        st.image(bio_rgb, width="stretch")
        st.warning(
            "**Retinal Processing Only**\n\n"
            "Even if a cat stood as tall as a human, the world would look like this. "
            "Cats are dichromatic (Red-Green colorblind) and have lower visual acuity (blurrier) "
            "to prioritize motion detection over detail."
        )
        
    with bot_col2:
        st.markdown("#### Factor B: Physics (The Body)")
        st.image(geo_rgb, width="stretch")
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
        st.image(overlay_img, width="stretch")
        st.info("**Attention Heatmap**\n\nBright red zones indicate high visual 'weight' based on contrast, shape, and height.")
    
    
    with col_int2:
        # HUMAN CROP (using Golden Ratio point)
        hx, hy = human_peak_loc
        hx1, hx2 = max(0, hx-75), min(img_rgb.shape[1], hx+75)
        hy1, hy2 = max(0, hy-75), min(img_rgb.shape[0], hy+75)
        human_crop = img_rgb[hy1:hy2, hx1:hx2]

        # PET CROP (using Biological Bias point)
        px, py = pet_peak_loc
        px1, px2 = max(0, px-75), min(img_rgb.shape[1], px+75)
        py1, py2 = max(0, py-75), min(img_rgb.shape[0], py+75)

        pet_crop_sharp = img_rgb[py1:py2, px1:px2]
        

        # 3. Display side-by-side comparison
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.image(human_crop, caption="Primary Object (Human Focus)", use_container_width=True)
        with sub_col2:
            st.image(pet_crop_sharp, caption=f"Primary Object ({current_mode} Focus - Sharp View)", use_container_width=True)

        st.warning(f"**Analysis:** This clear segment shows exactly what is at the {current_mode.lower()}'s peak interest point. This object is the most likely to grab a {current_mode.lower()}'s attention first.")

    # ROW 4: Pet Attractiveness Scoring and Improvement
    st.write("---")
    pet_emoji = "🐱" if current_mode == "Cat" else "🐕"
    st.write(f"### 4. {pet_emoji} Room Attractiveness and Improvement")
    st.caption(f"Analyze how appealing this space is to a {current_mode.lower()} and get personalized improvement suggestions.")
    
    with st.spinner(f"🔍 Analyzing space attractiveness for {current_mode.lower()}s..."):
        try:
            detector_model = load_detector_model()
            
            # Use appropriate scoring function based on mode
            if current_mode == "Cat":
                pet_scores, debug_info = score_cat_attractiveness(
                    img_bgr, 
                    detector_model=detector_model,
                    return_debug=True
                )
                category_names = CAT_CATEGORY_NAMES
            else:  # Dog mode
                pet_scores, debug_info = score_dog_attractiveness(
                    img_bgr, 
                    detector_model=detector_model,
                    return_debug=True
                )
                category_names = DOG_CATEGORY_NAMES
            
            # Display overall score prominently
            overall_pct = int(pet_scores.overall * 100)
            if overall_pct >= 70:
                score_color = "🟢"
                score_msg = f"Excellent! A {current_mode.lower()} would love this space."
            elif overall_pct >= 50:
                score_color = "🟡"
                score_msg = f"Good. This space has nice {current_mode.lower()}-friendly features."
            elif overall_pct >= 30:
                score_color = "🟠"
                score_msg = f"Okay. Some improvements could make it more {current_mode.lower()}-friendly."
            else:
                score_color = "🔴"
                score_msg = f"Low. This space may not be very appealing to {current_mode.lower()}s."
            
            st.metric(
                label=f"{score_color} Overall {current_mode} Attractiveness",
                value=f"{overall_pct}%",
                help=score_msg
            )
            st.caption(score_msg)
            
            # Helper function to generate explanations for cat scores
            def explain_cat_score(dimension: str, score: float, debug_info: Dict[str, Any], detections: List[Any]) -> str:
                """Generate explanation string for a cat attractiveness score dimension."""
                if dimension == "vertical_opportunity":
                    vertical_detections = [d for d in detections if d.category == "vertical"]
                    vertical_area = debug_info.get("vertical_area", 0)
                    vertical_ratio = debug_info.get("vertical_ratio", 0)
                    edge_energy = debug_info.get("edge_energy", 0)
                    
                    parts = []
                    if vertical_detections:
                        obj_names = [d.class_name for d in vertical_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(vertical_area * 100)
                        parts.append(f"we detected {obj_list}, which occupies {area_pct}% of the image area")
                    
                    # Visual features contribute 65% of the score
                    if vertical_ratio > 1.2 or edge_energy > 0.05:
                        parts.append("the scene has strong vertical edge patterns and lines")
                    
                    if parts:
                        return f"This measures climbing & perching opportunities. Cats love vertical structures. The score combines detected objects (35%) and visual features (65%). In this image, {', and '.join(parts)}."
                    else:
                        return "This measures climbing & perching opportunities. Cats love vertical structures. The score combines detected objects (35%) and visual features (65%). We didn't detect vertical objects, and the scene has minimal vertical edge patterns."
                
                elif dimension == "shelter_hiding":
                    shelter_detections = [d for d in detections if d.category == "shelter"]
                    shelter_area = debug_info.get("shelter_area", 0)
                    nook_count = debug_info.get("nook_count", 0)
                    nook_area = debug_info.get("nook_area_sum", 0)
                    
                    parts = []
                    if shelter_detections:
                        obj_names = [d.class_name for d in shelter_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(shelter_area * 100)
                        parts.append(f"we detected {obj_list}, which occupies {area_pct}% of the image area")
                    
                    if nook_count > 0:
                        nook_area_pct = int(nook_area * 100)
                        parts.append(f"we found {nook_count} dark hiding nooks occupying {nook_area_pct}% of the image area")
                    
                    if parts:
                        return f"This measures hiding spots & enclosed spaces. Cats need shelter and hiding spots. The score combines dark nooks (60%) and detected objects (40%). In this image, {', and '.join(parts)}."
                    else:
                        return "This measures hiding spots & enclosed spaces. Cats need shelter and hiding spots. The score combines dark nooks (60%) and detected objects (40%). We didn't detect many shelter objects or dark hiding nooks in this image."
                
                elif dimension == "cozy_warmth":
                    cozy_detections = [d for d in detections if d.category == "cozy"]
                    cozy_area = debug_info.get("cozy_area", 0)
                    warmth = debug_info.get("warmth_mean_r_minus_b", 0)
                    edge_energy = debug_info.get("edge_energy", 0)
                    
                    parts = []
                    if cozy_detections:
                        obj_names = [d.class_name for d in cozy_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(cozy_area * 100)
                        parts.append(f"we detected {obj_list}, which occupies {area_pct}% of the image area")
                    
                    # Visual features: warmth and softness (low edge energy = soft)
                    visual_parts = []
                    if warmth > 0.05:
                        visual_parts.append("warm color tones")
                    elif warmth < -0.05:
                        visual_parts.append("cool color tones")
                    else:
                        visual_parts.append("neutral color tones")
                    
                    if edge_energy < 0.08:
                        visual_parts.append("soft textures (low edge energy)")
                    
                    if visual_parts:
                        parts.append(f"the scene has {', and '.join(visual_parts)}")
                    
                    if parts:
                        return f"This measures warm & comfortable areas. Cats seek cozy, warm areas. The score combines color/softness (60%) and detected objects (40%). In this image, {', and '.join(parts)}."
                    else:
                        return "This measures warm & comfortable areas. Cats seek cozy, warm areas. The score combines color/softness (60%) and detected objects (40%). We didn't detect many cozy objects, and the scene has neutral tones."
                
                elif dimension == "exploration_richness":
                    explore_detections = [d for d in detections if d.category == "exploration"]
                    explore_area = debug_info.get("explore_area", 0)
                    edge_density = debug_info.get("edge_density", 0)
                    entropy = debug_info.get("entropy", 0)
                    
                    parts = []
                    if explore_detections:
                        obj_names = [d.class_name for d in explore_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(explore_area * 100)
                        parts.append(f"we detected {obj_list}, which occupies {area_pct}% of the image area")
                    
                    # Visual complexity: moderate is best (not too simple, not too chaotic)
                    # Optimal edge_density is around 0.11 (middle of 0.02-0.20 range)
                    # Optimal entropy is around 3.5 (middle of 2.0-5.0 range)
                    visual_parts = []
                    if 0.08 <= edge_density <= 0.14 and 2.5 <= entropy <= 4.5:
                        visual_parts.append("moderate visual complexity (good for exploration)")
                    elif edge_density < 0.08 or entropy < 2.5:
                        visual_parts.append("low visual complexity (too simple)")
                    else:
                        visual_parts.append("high visual complexity (may be overwhelming)")
                    
                    if visual_parts:
                        parts.append(f"the scene has {visual_parts[0]}")
                    
                    if parts:
                        return f"This measures interesting objects to investigate. Cats love to explore interesting objects. The score combines visual complexity (65%) and detected objects (35%). In this image, {', and '.join(parts)}."
                    else:
                        return "This measures interesting objects to investigate. Cats love to explore interesting objects. The score combines visual complexity (65%) and detected objects (35%). We didn't detect exploration items, and the scene has low visual complexity."
                
                elif dimension == "safety_low_threat":
                    threat_detections = [d for d in detections if d.category == "threat"]
                    threat_count = debug_info.get("threat_count", 0)
                    threat_area = debug_info.get("threat_area", 0)
                    brightness = debug_info.get("mean_brightness", 0.5)
                    edge_density = debug_info.get("edge_density", 0)
                    
                    parts = []
                    if threat_detections:
                        obj_names = [d.class_name for d in threat_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(threat_area * 100)
                        parts.append(f"we detected {obj_list} (potential social complexity), which occupies {area_pct}% of the image area")
                    else:
                        parts.append("no threats detected")
                    
                    # Visual features: brightness (70%) + clutter (15%) + threats (15%)
                    visual_parts = []
                    if 0.35 <= brightness <= 0.75:
                        visual_parts.append("comfortable brightness level")
                    elif brightness < 0.35:
                        visual_parts.append("somewhat dim lighting")
                    else:
                        visual_parts.append("very bright lighting")
                    
                    if edge_density < 0.15:
                        visual_parts.append("low clutter")
                    elif edge_density > 0.25:
                        visual_parts.append("high clutter")
                    
                    if visual_parts:
                        parts.append(f"the scene has {', and '.join(visual_parts)}")
                    
                    return f"This measures low threat level & security. Cats need safe, low-threat environments. The score combines brightness (70%), clutter (15%), and threats (15%). In this image, {', and '.join(parts)}."
                
                return ""
            
            # Helper function to generate explanations for dog scores
            def explain_dog_score(dimension: str, score: float, debug_info: Dict[str, Any], detections: List[Any]) -> str:
                """Generate explanation string for a dog attractiveness score dimension."""
                if dimension == "floor_play_space":
                    obstacle_detections = [d for d in detections if d.category == "floor"]
                    obstacle_area = debug_info.get("obstacle_area", 0)
                    edge_density = debug_info.get("edge_density", 0)
                    
                    parts = []
                    if obstacle_detections:
                        obj_names = [d.class_name for d in obstacle_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(obstacle_area * 100)
                        parts.append(f"we detected {obj_list} as floor obstacles, which occupies {area_pct}% of the image area")
                    else:
                        parts.append("no major obstacles detected")
                    
                    # Visual features: clutter (edge_density)
                    if edge_density < 0.15:
                        parts.append("the floor is relatively clear (low clutter)")
                    elif edge_density > 0.25:
                        parts.append("the floor is cluttered (high edge density)")
                    else:
                        parts.append("the floor has moderate clutter")
                    
                    return f"This measures open floor space for play. Dogs need open floor space for play. The score combines clutter (60%) and obstacles (40%). In this image, {', and '.join(parts)}."
                
                elif dimension == "rest_cozy":
                    rest_detections = [d for d in detections if d.category == "rest"]
                    rest_area = debug_info.get("rest_area", 0)
                    warmth = debug_info.get("warmth_mean_r_minus_b", 0)
                    edge_energy = debug_info.get("edge_energy", 0)
                    
                    parts = []
                    if rest_detections:
                        obj_names = [d.class_name for d in rest_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(rest_area * 100)
                        parts.append(f"we detected {obj_list}, which occupies {area_pct}% of the image area")
                    
                    # Visual features: warmth and softness
                    visual_parts = []
                    if warmth > 0.05:
                        visual_parts.append("warm color tones")
                    elif warmth < -0.05:
                        visual_parts.append("cool color tones")
                    else:
                        visual_parts.append("neutral color tones")
                    
                    if edge_energy < 0.08:
                        visual_parts.append("soft textures (low edge energy)")
                    
                    if visual_parts:
                        parts.append(f"the scene has {', and '.join(visual_parts)}")
                    
                    if parts:
                        return f"This measures comfortable resting spots. Dogs need comfortable resting spots. The score combines color/softness (60%) and detected objects (40%). In this image, {', and '.join(parts)}."
                    else:
                        return "This measures comfortable resting spots. Dogs need comfortable resting spots. The score combines color/softness (60%) and detected objects (40%). We didn't detect many rest surfaces, and the scene has neutral tones."
                
                elif dimension == "sniff_enrichment":
                    sniff_detections = [d for d in detections if d.category == "sniff"]
                    sniff_area = debug_info.get("sniff_area", 0)
                    edge_density = debug_info.get("edge_density", 0)
                    entropy = debug_info.get("entropy", 0)
                    
                    parts = []
                    if sniff_detections:
                        obj_names = [d.class_name for d in sniff_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(sniff_area * 100)
                        parts.append(f"we detected {obj_list}, which occupies {area_pct}% of the image area")
                    
                    # Visual complexity: moderate is best
                    visual_parts = []
                    if 0.08 <= edge_density <= 0.18 and 2.5 <= entropy <= 4.5:
                        visual_parts.append("moderate visual complexity (good for exploration)")
                    elif edge_density < 0.08 or entropy < 2.5:
                        visual_parts.append("low visual complexity (too simple)")
                    else:
                        visual_parts.append("high visual complexity (may be overwhelming)")
                    
                    if visual_parts:
                        parts.append(f"the scene has {visual_parts[0]}")
                    
                    if parts:
                        return f"This measures sniffing & enrichment opportunities. Dogs love to sniff and explore interesting objects. The score combines visual complexity (60%) and detected objects (40%). In this image, {', and '.join(parts)}."
                    else:
                        return "This measures sniffing & enrichment opportunities. Dogs love to sniff and explore interesting objects. The score combines visual complexity (60%) and detected objects (40%). We didn't detect enrichment items, and the scene has low visual complexity."
                
                elif dimension == "water_food_ready":
                    water_food_detections = [d for d in detections if d.category == "water_food"]
                    water_food_area = debug_info.get("water_food_area", 0)
                    counts = debug_info.get("counts", {})
                    bottle_count = counts.get("bottle", 0)
                    cup_count = counts.get("cup", 0)
                    
                    parts = []
                    if water_food_detections:
                        obj_names = [d.class_name for d in water_food_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(water_food_area * 100)
                        parts.append(f"we detected {obj_list}, which occupies {area_pct}% of the image area")
                    
                    if (bottle_count + cup_count) > 0:
                        parts.append(f"we found {bottle_count + cup_count} drink container{'s' if (bottle_count + cup_count) > 1 else ''}")
                    
                    if parts:
                        return f"This measures water & food accessibility. Dogs need easy access to water and food. The score combines detected objects (60%) and drink containers (40%). In this image, {', and '.join(parts)}."
                    else:
                        return "This measures water & food accessibility. Dogs need easy access to water and food. The score combines detected objects (60%) and drink containers (40%). We didn't detect many food/water-related objects in this image."
                
                elif dimension == "safety_low_threat":
                    threat_detections = [d for d in detections if d.category == "threat"]
                    threat_count = debug_info.get("threat_count", 0)
                    threat_area = debug_info.get("threat_area", 0)
                    brightness = debug_info.get("mean_brightness", 0.5)
                    edge_density = debug_info.get("edge_density", 0)
                    
                    parts = []
                    if threat_detections:
                        obj_names = [d.class_name for d in threat_detections]
                        obj_counts = {}
                        for name in obj_names:
                            obj_counts[name] = obj_counts.get(name, 0) + 1
                        obj_list = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        area_pct = int(threat_area * 100)
                        parts.append(f"we detected {obj_list} (potential social complexity), which occupies {area_pct}% of the image area")
                    else:
                        parts.append("no threats detected")
                    
                    # Visual features: brightness (60%) + clutter (25%) + threats (15%)
                    visual_parts = []
                    if 0.35 <= brightness <= 0.75:
                        visual_parts.append("comfortable brightness level")
                    elif brightness < 0.35:
                        visual_parts.append("somewhat dim lighting")
                    else:
                        visual_parts.append("very bright lighting")
                    
                    if edge_density < 0.15:
                        visual_parts.append("low clutter")
                    elif edge_density > 0.25:
                        visual_parts.append("high clutter")
                    
                    if visual_parts:
                        parts.append(f"the scene has {', and '.join(visual_parts)}")
                    
                    return f"This measures low threat level & security. Dogs need safe, low-threat environments. The score combines brightness (60%), clutter (25%), and threats (15%). In this image, {', and '.join(parts)}."
                
                return ""
            
            # Display individual dimension scores based on pet type
            score_cols = st.columns(5)
            
            if current_mode == "Cat":
                dimensions = [
                    ("🧗 Vertical", pet_scores.vertical_opportunity, "Climbing & perching opportunities", "vertical_opportunity"),
                    ("🏠 Shelter", pet_scores.shelter_hiding, "Hiding spots & enclosed spaces", "shelter_hiding"),
                    ("☀️ Cozy", pet_scores.cozy_warmth, "Warm & comfortable areas", "cozy_warmth"),
                    ("🎯 Explore", pet_scores.exploration_richness, "Interesting objects to investigate", "exploration_richness"),
                    ("🛡️ Safety", pet_scores.safety_low_threat, "Low threat level & security", "safety_low_threat"),
                ]
            else:  # Dog mode
                dimensions = [
                    ("🏃 Floor/Play", pet_scores.floor_play_space, "Open floor space for play", "floor_play_space"),
                    ("🛋️ Rest", pet_scores.rest_cozy, "Comfortable resting spots", "rest_cozy"),
                    ("👃 Sniff", pet_scores.sniff_enrichment, "Sniffing & enrichment opportunities", "sniff_enrichment"),
                    ("🥣 Food/Water", pet_scores.water_food_ready, "Water & food accessibility", "water_food_ready"),
                    ("🛡️ Safety", pet_scores.safety_low_threat, "Low threat level & security", "safety_low_threat"),
                ]
            
            detections = debug_info.get("detections", []) if debug_info else []
            
            for col, (label, score, tooltip, dimension_key) in zip(score_cols, dimensions):
                with col:
                    pct = int(score * 100)
                    st.metric(label=label, value=f"{pct}%")
                    # Generate and display explanation
                    if current_mode == "Cat":
                        explanation = explain_cat_score(dimension_key, score, debug_info, detections)
                    else:
                        explanation = explain_dog_score(dimension_key, score, debug_info, detections)
                    if explanation:
                        st.caption(explanation)
            
            # Show annotated image with bounding boxes and detected objects
            if detections:
                st.write("#### Detected Objects")
                
                # Scale bounding boxes from resized detection image back to original image size
                # Detection happens on images resized to max 960px, but we draw on original size
                original_h, original_w = img_rgb.shape[:2]
                target_size = 960
                scale = target_size / max(original_h, original_w)
                if scale < 1.0:
                    resized_h = int(original_h * scale)
                    resized_w = int(original_w * scale)
                    scaled_detections = scale_bboxes_to_original_size(
                        detections, original_h, original_w, resized_h, resized_w
                    )
                else:
                    # Image wasn't resized, use detections as-is
                    scaled_detections = detections
                
                # Draw bounding boxes on original image
                annotated_img = draw_detections_on_image(img_rgb, scaled_detections, category_names)
                st.image(annotated_img, width="stretch")
                
                # Legend for categories based on pet type
                if current_mode == "Cat":
                    st.caption("**Category Legend:** 🧗 Vertical (Orange) | 🏠 Shelter (Purple) | ☀️ Cozy (Pink) | 🎯 Explore (Green) | ⚠️ Threat (Red) | Uncategorized (Gray)")
                else:
                    st.caption("**Category Legend:** 🏃 Floor (Orange) | 🛋️ Rest (Purple) | 👃 Sniff (Pink) | 🥣 Food/Water (Green) | ⚠️ Threat (Red) | Uncategorized (Gray)")
                
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
                        cat_display = category_names.get(cat_key, cat_key.title())
                        obj_counts = {}
                        for obj in objects:
                            obj_counts[obj] = obj_counts.get(obj, 0) + 1
                        obj_str = ", ".join([f"{k} ({v})" if v > 1 else k for k, v in obj_counts.items()])
                        st.write(f"**{cat_display}:** {obj_str}")
            else:
                st.info("No objects detected in this image.")
            
            # Room Improvement Suggestions (part of section 4)
            st.write("")
            st.write("#### 💡 Room Improvement Suggestions")
            st.caption(f"Tenant-friendly tips to make this space more {current_mode.lower()}-attractive.")
            
            # Toggle for illustrated diagrams
            generate_diagrams = st.checkbox(
                "🎨 Generate illustrated diagrams for each step",
                value=False,
                help="Uses AI to create simple visual diagrams for each step. May take a few seconds per step."
            )
            
            with st.spinner("🤔 Generating personalized suggestions..."):
                try:
                    config = load_config()
                    openai_api_key = config.get("openai_api_key")
                    
                    # Use appropriate suggestion engine based on mode
                    if current_mode == "Cat":
                        suggestion_engine = CatRoomSuggestionEngine(
                            openai_api_key=openai_api_key,
                            max_suggestions=5,
                            use_openai=bool(openai_api_key),
                        )
                    else:  # Dog mode
                        suggestion_engine = DogRoomSuggestionEngine(
                            openai_api_key=openai_api_key,
                            max_suggestions=5,
                            use_openai=bool(openai_api_key),
                        )
                    
                    suggestions = suggestion_engine.suggest(pet_scores, debug_info)
                    
                    if suggestions:
                        # Separate suggestions into free and paid
                        free_suggestions = [s for s in suggestions if s.cost == "free"]
                        paid_suggestions = [s for s in suggestions if s.cost != "free"]
                        
                        tab_free, tab_paid = st.tabs(["🆓 Free", "💵 Paid"])
                        
                        def render_suggestion(sug, expanded=False, show_amazon_links=False, generate_diagrams=False, api_key=None):
                            """Render a single suggestion in an expander."""
                            # Category emoji mapping for both cat and dog categories
                            category_emojis = {
                                # Cat categories
                                "vertical": "🧗",
                                "shelter": "🏠",
                                "cozy": "☀️",
                                "exploration": "🎯",
                                # Dog categories
                                "floor": "🏃",
                                "rest": "🛋️",
                                "sniff": "👃",
                                "water_food": "🥣",
                                # Shared
                                "safety": "🛡️",
                            }
                            cat_emoji = category_emojis.get(sug.category, "💡")
                            effort_badge = {"tiny": "⚡", "small": "🔧", "medium": "🔨"}.get(sug.effort, "")
                            
                            with st.expander(f"{cat_emoji} **{sug.title}** {effort_badge}", expanded=expanded):
                                st.markdown(f"**Why it helps:** {sug.why_it_helps}")
                                
                                # Generate a single diagram for the entire suggestion
                                if generate_diagrams and api_key:
                                    col_diagram, col_steps = st.columns([1, 2])
                                    
                                    with col_diagram:
                                        with st.spinner("🎨 Generating diagram..."):
                                            # Convert steps list to tuple for caching
                                            diagram_url, diagram_error = generate_suggestion_diagram(
                                                tuple(sug.steps), 
                                                api_key
                                            )
                                            if diagram_url:
                                                st.image(diagram_url, width="stretch")
                                            else:
                                                st.markdown(f"<div style='text-align:center;font-size:64px;padding:40px;background:#f0f0f0;border-radius:12px;'>{cat_emoji}</div>", unsafe_allow_html=True)
                                                if diagram_error:
                                                    st.error(f"Diagram error: {diagram_error}")
                                    
                                    with col_steps:
                                        st.markdown("**Steps:**")
                                        for step_num, step in enumerate(sug.steps, 1):
                                            st.markdown(f"{step_num}. {step}")
                                else:
                                    st.markdown("**Steps:**")
                                    for step_num, step in enumerate(sug.steps, 1):
                                        st.markdown(f"{step_num}. {step}")
                                
                                if sug.expected_score_lift:
                                    lift_parts = []
                                    for dim, lift in sug.expected_score_lift.items():
                                        dim_display = dim.replace("_", " ").title()
                                        lift_parts.append(f"{dim_display}: +{int(lift * 100)}%")
                                    st.caption(f"📈 Expected improvement: {', '.join(lift_parts)}")
                                
                                st.caption(f"Effort: {sug.effort.title()}")
                                
                                # Show Amazon links for paid suggestions
                                if show_amazon_links and sug.amazon_links:
                                    st.markdown("**🛒 Shop on Amazon:**")
                                    link_cols = st.columns(min(len(sug.amazon_links), 3))
                                    for idx, link in enumerate(sug.amazon_links[:3]):
                                        with link_cols[idx]:
                                            st.link_button(f"🔗 {link.label}", link.url, width="stretch")
                        
                        with tab_free:
                            if free_suggestions:
                                for i, sug in enumerate(free_suggestions):
                                    render_suggestion(
                                        sug, 
                                        expanded=(i == 0),
                                        generate_diagrams=generate_diagrams,
                                        api_key=openai_api_key
                                    )
                            else:
                                st.info("No free suggestions available for this space.")
                        
                        with tab_paid:
                            if paid_suggestions:
                                for i, sug in enumerate(paid_suggestions):
                                    render_suggestion(
                                        sug, 
                                        expanded=(i == 0), 
                                        show_amazon_links=True,
                                        generate_diagrams=generate_diagrams,
                                        api_key=openai_api_key
                                    )
                            else:
                                st.info("No paid suggestions needed for this space.")
                    else:
                        st.success(f"🎉 This space already looks great for {current_mode.lower()}s! No major improvements needed.")
                        
                except Exception as e:
                    st.warning(f"Could not generate suggestions: {e}")
                        
        except Exception as e:
            st.error(f"Error analyzing {current_mode.lower()} attractiveness: {e}")

    # ROW 5: Room Vulnerability Analysis
    st.write("---")
    st.write("### 5. ⚠️ Room Vulnerability and Improvement")
    st.caption(f"Identify potential hazards and vulnerable objects that could be damaged by or harm a {current_mode.lower()}.")
    
    with st.spinner("🔍 Analyzing room vulnerabilities..."):
        try:
            config = load_config()
            openai_api_key = config.get("openai_api_key")
            
            if not openai_api_key:
                st.warning("⚠️ OpenAI API key not configured. Vulnerability analysis requires an API key.")
            else:
                # Determine pet type for vulnerability analysis
                pet_type_for_vuln = current_mode.lower()  # "cat" or "dog"
                
                analyzer = PetVulnerabilityAnalyzer(
                    api_key=openai_api_key,
                    model="gpt-4o",
                    max_objects=15,
                )
                
                image_description, vulnerable_objects, raw_response = analyzer.analyze(
                    pil_image,
                    pet_type=pet_type_for_vuln,
                    renter_mode=True,
                )
                
                # Display debug information in collapsible section
                with st.expander("🔍 Debug: Room Description & Raw API Response", expanded=False):
                    st.write("#### 📝 Room Description")
                    st.write(image_description)
                    st.write("---")
                    st.write("#### 📄 Raw API Response")
                    st.code(raw_response, language="json")
                
                # Flatten structured vulnerabilities into a single list for processing
                all_vulnerable_objects: List[VulnerableObject] = []
                for risk_type, objects in vulnerable_objects.items():
                    all_vulnerable_objects.extend(objects)
                
                if all_vulnerable_objects:
                    # Show the original image (no bounding boxes since we don't request them)
                    st.image(img_rgb, width="stretch")
                    st.info("💡 Vulnerability analysis is based on image description. Items are listed below by risk type.")
                    
                    # Display vulnerabilities by risk type (structured format)
                    risk_type_labels = {
                        "breakable": "💥 Breakable",
                        "scratchable": "🐾 Scratchable",
                        "chewable": "🦷 Chewable",
                        "toxic": "☠️ Toxic",
                        "unstable": "⚖️ Unstable",
                    }
                    
                    # Create tabs for each risk type that has items
                    risk_types_with_items = [rt for rt in risk_type_labels.keys() if vulnerable_objects.get(rt)]
                    
                    if risk_types_with_items:
                        risk_tabs = st.tabs([risk_type_labels[rt] for rt in risk_types_with_items])
                        
                        for tab_idx, (risk_type, tab) in enumerate(zip(risk_types_with_items, risk_tabs)):
                            with tab:
                                objects = vulnerable_objects[risk_type]
                                if objects:
                                    # Group by severity within each risk type
                                    by_severity: Dict[str, List[VulnerableObject]] = {
                                        "high": [],
                                        "medium": [],
                                        "low": [],
                                    }
                                    for vuln_obj in objects:
                                        by_severity[vuln_obj.severity].append(vuln_obj)
                                    
                                    # Show high severity first, then medium, then low
                                    for severity in ["high", "medium", "low"]:
                                        severity_objects = by_severity[severity]
                                        if severity_objects:
                                            severity_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(severity, "⚪")
                                            st.markdown(f"### {severity_emoji} {severity.title()} Severity")
                                            for i, vuln_obj in enumerate(severity_objects):
                                                risk_emoji = {
                                                    "breakable": "💥",
                                                    "scratchable": "🐾",
                                                    "chewable": "🦷",
                                                    "toxic": "☠️",
                                                    "unstable": "⚖️",
                                                }.get(vuln_obj.risk_type, "⚠️")
                                                
                                                with st.expander(f"{risk_emoji} **{vuln_obj.label}**", expanded=(i == 0 and severity == "high")):
                                                    st.markdown(f"**Risk Type:** {vuln_obj.risk_type.title()}")
                                                    st.markdown(f"**Severity:** {vuln_obj.severity.title()}")
                                                    
                                                    st.markdown("**Why it's a concern:**")
                                                    for reason in vuln_obj.reasons:
                                                        st.markdown(f"- {reason}")
                                                    
                                                    # Free fixes
                                                    if vuln_obj.quick_fixes_free:
                                                        st.markdown("**🆓 Free Solutions:**")
                                                        for fix in vuln_obj.quick_fixes_free:
                                                            st.markdown(f"- {fix}")
                                                    
                                                    # Paid fixes
                                                    if vuln_obj.quick_fixes_paid:
                                                        st.markdown("**💵 Paid Solutions:**")
                                                        for fix in vuln_obj.quick_fixes_paid:
                                                            st.markdown(f"- {fix}")
                                else:
                                    st.info(f"No {risk_type} vulnerabilities detected.")
                    else:
                        st.info("No vulnerabilities found in any category.")
                    
                    # Room Improvement Suggestions subsection
                    st.write("---")
                    st.write("#### 💡 Room Improvement Suggestions")
                    st.caption(f"Tenant-friendly solutions to address the identified vulnerabilities and make this space safer for {current_mode.lower()}s.")
                    
                    # Aggregate all free and paid fixes from all vulnerabilities
                    all_free_fixes: List[Tuple[str, str, str]] = []  # (risk_type, label, fix)
                    all_paid_fixes: List[Tuple[str, str, str]] = []  # (risk_type, label, fix)
                    
                    for risk_type, objects in vulnerable_objects.items():
                        for vuln_obj in objects:
                            # Add free fixes
                            for fix in vuln_obj.quick_fixes_free:
                                all_free_fixes.append((risk_type, vuln_obj.label, fix))
                            # Add paid fixes
                            for fix in vuln_obj.quick_fixes_paid:
                                all_paid_fixes.append((risk_type, vuln_obj.label, fix))
                    
                    # Display suggestions in tabs
                    if all_free_fixes or all_paid_fixes:
                        tab_free, tab_paid = st.tabs(["🆓 Free", "💵 Paid"])
                        
                        with tab_free:
                            if all_free_fixes:
                                # Group fixes by risk type
                                fixes_by_risk: Dict[str, List[Tuple[str, str]]] = {}
                                for risk_type, label, fix in all_free_fixes:
                                    if risk_type not in fixes_by_risk:
                                        fixes_by_risk[risk_type] = []
                                    fixes_by_risk[risk_type].append((label, fix))
                                
                                risk_emoji_map = {
                                    "breakable": "💥",
                                    "scratchable": "🐾",
                                    "chewable": "🦷",
                                    "toxic": "☠️",
                                    "unstable": "⚖️",
                                }
                                
                                for risk_type, fixes in fixes_by_risk.items():
                                    risk_emoji = risk_emoji_map.get(risk_type, "⚠️")
                                    st.markdown(f"##### {risk_emoji} {risk_type.title()}")
                                    for label, fix in fixes:
                                        with st.expander(f"**{label}**", expanded=False):
                                            st.markdown(fix)
                            else:
                                st.info("No free solutions available.")
                        
                        with tab_paid:
                            if all_paid_fixes:
                                # Group fixes by risk type
                                fixes_by_risk: Dict[str, List[Tuple[str, str]]] = {}
                                for risk_type, label, fix in all_paid_fixes:
                                    if risk_type not in fixes_by_risk:
                                        fixes_by_risk[risk_type] = []
                                    fixes_by_risk[risk_type].append((label, fix))
                                
                                risk_emoji_map = {
                                    "breakable": "💥",
                                    "scratchable": "🐾",
                                    "chewable": "🦷",
                                    "toxic": "☠️",
                                    "unstable": "⚖️",
                                }
                                
                                for risk_type, fixes in fixes_by_risk.items():
                                    risk_emoji = risk_emoji_map.get(risk_type, "⚠️")
                                    st.markdown(f"##### {risk_emoji} {risk_type.title()}")
                                    for label, fix in fixes:
                                        with st.expander(f"**{label}**", expanded=False):
                                            st.markdown(fix)
                            else:
                                st.info("No paid solutions needed.")
                    else:
                        st.info("No improvement suggestions available.")
                else:
                    st.success(f"✅ No significant vulnerabilities detected! This space appears safe for {current_mode.lower()}s.")
                    
        except Exception as e:
            st.error(f"Error analyzing room vulnerabilities: {e}")
            import traceback
            st.code(traceback.format_exc())

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

# ... (End of your existing processing block)

    # --- NEW STEP 6: INSIDE THE CONDITIONAL BLOCK ---
    st.write("---")
    st.header("📸 Step 6: Custom Your Pet Analysis")
 
    # --- REVISED STEP 6 IN app.py ---
    uploaded_pet = st.file_uploader(
        "Upload your cat's or dog's photo for AI analysis. You can see the real world in your pet's eyes!", 
        type=['png', 'jpg'], 
        key="pet_analysis_uploader"
    )

    # CRITICAL FIX: Only execute this block if a file is actually present
    if uploaded_pet is not None:
        # 1. Clear the old analysis text if a new image is detected
        if 'pet_analysis_summary' in st.session_state:
            del st.session_state['pet_analysis_summary']
        if 'pet_species' in st.session_state:
            del st.session_state['pet_species']

        # 2. Save the NEW image to session state
        st.session_state['pet_image'] = Image.open(uploaded_pet)
        
        # 3. Only show the button if an image is ready
        if st.button("Analyze My Pet! 🐾", key="btn_to_analysis"):
            st.switch_page("pages/pet_analysis.py")
    else:
        # Optional: Show a message when no file is uploaded
        st.info("Please upload a pet photo to proceed to AI analysis.")