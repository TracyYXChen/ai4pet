import streamlit as st
import numpy as np
import cv2
import av
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

from transformers import pipeline

# 1. Load the AI Depth Model (Cached for speed)
@st.cache_resource
def load_depth_model():
    # 'depth-anything' is excellent, but 'vincent-cl/monodepth2-visdrone-v2' 
    # or the standard DPT are lighter. Let's use a standard robust one:
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    return pipe

# 2. The Geometric Perspective Warp (Approach 1: Keystone + Depth)
def apply_cat_perspective_warp(image_rgb, depth_map, strength=1.0):
    """
    Simulates a low-angle 'Cat View' using Perspective Keystone + Depth Warping.
    Fixed to remove streaks and correct the viewing angle.
    """
    h, w, _ = image_rgb.shape
    
    # --- STEP 1: KEYSTONE (The "Looking Up" shape) ---
    # To look up, parallel vertical lines (trees) should converge at the top.
    # To avoid black borders/streaks, we effectively "Zoom In" while we pinch.
    
    tilt = 0.4 * strength  # Pinch factor
    
    # Source: We take a TRAPEZOID from the center of the original image
    # (Narrower at bottom, Wider at top). 
    # When we stretch this to a square, the top gets squeezed (convergence)
    # and the bottom gets stretched (wide angle near ground).
    
    # Coordinates of the trapezoid to crop FROM the original:
    # Top: Full width
    # Bottom: Narrower (we zoom in on the ground detail)
    src_pts = np.float32([
        [0, 0],             # Top Left
        [w, 0],             # Top Right
        [w * tilt, h],      # Bottom Left (moved in)
        [w * (1-tilt), h]   # Bottom Right (moved in)
    ])
    
    # Destination: The full screen square
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    keystone_image = cv2.warpPerspective(image_rgb, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # --- STEP 2: DEPTH DISPLACEMENT (The "Low Height" Parallax) ---
    
    # Resize depth to match image
    depth = cv2.resize(np.array(depth_map), (w, h)).astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) # Normalize 0..1
    
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # DIRECTION FIX: 
    # We want to simulate looking UP from the ground.
    # This means the image content should slide DOWN (sky comes down).
    # So we SUBTRACT the shift.
    shift_y = depth * (h * 0.1 * strength)
    
    map_y_new = (map_y - shift_y).astype(np.float32) # Minus means move down
    map_x_new = map_x.astype(np.float32)
    
    final_image = cv2.remap(keystone_image, map_x_new, map_y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Final cleanup crop (just 1% to clean edge interpolation artifacts)
    crop = int(h * 0.01)
    final_image = final_image[crop:h-crop, crop:w-crop]
    
    # Resize back to original
    final_image = cv2.resize(final_image, (w, h))
    
    return final_image

st.set_page_config(page_title="Pet Vision Filters", layout="wide")
st.title("🐾 Pet Vision Camera Filters (Approx.)")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Vision mode", ["Cat", "Dog"], index=0)

    st.header("Tuning")
    blur_sigma = st.slider("Blur (acuity loss) sigma", 0.0, 6.0, 2.0, 0.1)

    # Color controls
    desat = st.slider("Desaturation", 0.0, 0.8, 0.20, 0.01)

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
    # rgb: float32 0..1
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    r2 = 0.15 * r + 0.35 * g + 0.50 * b
    g2 = 0.10 * r + 0.80 * g + 0.10 * b
    b2 = 0.05 * r + 0.25 * g + 0.70 * b
    return np.stack([r2, g2, b2], axis=-1)

def dog_color_transform(rgb: np.ndarray, strength: float) -> np.ndarray:
    """
    Approximate dog blue–yellow dichromacy.
    Intuition:
      - Reduce red/green opponency; push both toward a yellow-ish channel
      - Keep blue more distinguishable
    This is an artistic-but-useful approximation for a camera filter.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Base dichromacy-ish mapping (fast linear mix)
    # Reds/greens collapse into a "yellow" channel; blues remain more separate.
    r_d = 0.10 * r + 0.70 * g + 0.20 * b
    g_d = 0.05 * r + 0.80 * g + 0.15 * b
    b_d = 0.02 * r + 0.20 * g + 0.78 * b
    mapped = np.stack([r_d, g_d, b_d], axis=-1)

    # Blend with original to control strength
    return clamp01((1.0 - strength) * rgb + strength * mapped)

class PetVisionTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_gray = None

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        params = st.session_state.get("vision_params", {})
        if not params:
            params = dict(
                mode="Cat",
                blur_sigma=2.0,
                desat=0.2,
                lowlight_strength=0.8,
                noise_base=0.02,
                noise_extra=0.06,
                enable_motion=True,
                motion_gain=5.0,
                motion_boost=0.05,
                dog_strength=0.85,
            )

        img_bgr = frame.to_ndarray(format="bgr24")
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

        # 5) Optional motion emphasis
        gray = (0.299 * rgb2[..., 0] + 0.587 * rgb2[..., 1] + 0.114 * rgb2[..., 2]).astype(np.float32)
        if params["enable_motion"] and self.prev_gray is not None:
            diff = np.abs(gray - self.prev_gray)
            diff = np.clip(diff * float(params["motion_gain"]), 0, 1)
            rgb2 = clamp01(rgb2 + diff[..., None] * float(params["motion_boost"]))
        self.prev_gray = gray

        out_rgb = (rgb2 * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        
        # Convert filtered numpy array back to VideoFrame
        new_frame = av.VideoFrame.from_ndarray(out_bgr, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

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

# Input mode selection
st.subheader("Choose Input Mode")
input_mode = st.radio(
    "Select how you want to provide the image:",
    options=["📷 Live Camera", "🖼️ Upload Image"],
    horizontal=True,
    label_visibility="collapsed"
)

if input_mode == "📷 Live Camera":
    st.subheader("Live Camera Feed")
    webrtc_streamer(
        key="pet-vision",
        video_transformer_factory=PetVisionTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.info("Tip: If video is black, check browser camera permission (Chrome works best).")

else:  # Upload Image
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
                # Warp the original RGB image (no color filter)
                # Strength=0.8 means a strong lower-angle effect
                geo_rgb = apply_cat_perspective_warp(img_rgb, depth_map, strength=0.8)

                # 3. Complete Simulation (Combined)
                # Take the warped image (geo_rgb) and apply the biological filter
                geo_bgr = cv2.cvtColor(geo_rgb, cv2.COLOR_RGB2BGR)
                combined_bgr = apply_pet_filter_to_image(geo_bgr, params)
                combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

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

        with top_col2:
            st.subheader("Cat Reality")
            st.image(combined_rgb, use_container_width=True)
            st.success("**Cat Body + Cat Eyes**\n\nStanding height approx. 20cm. Objects loom over you, colors fade to blue/yellow, and detail blurs.")

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
                "If a human crawled on the floor, the world would look like this. "
                "Notice the 'Keystone Effect': because you are looking UP, vertical lines (trees, legs) "
                "converge inward, making them feel taller and more imposing."
            )

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