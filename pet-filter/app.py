import streamlit as st
import numpy as np
import cv2
import av
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

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
        # Load image
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply filter (no motion detection for static images)
        params = st.session_state.get("vision_params", {})
        filtered_bgr = apply_pet_filter_to_image(img_bgr, params)
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        # Display side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(pil_image, use_container_width=True)
        with col2:
            mode_name = params.get("mode", "Cat")
            st.markdown(f"**{mode_name} Vision Filter Applied**")
            st.image(filtered_rgb, use_container_width=True)
        
        # Download button for filtered image
        filtered_pil = Image.fromarray(filtered_rgb)
        buf = io.BytesIO()
        filtered_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Filtered Image",
            data=buf.getvalue(),
            file_name=f"pet_vision_filtered_{params.get('mode', 'cat').lower()}.png",
            mime="image/png"
        )
    else:
        st.info("👆 Upload an image above to see how it looks through pet eyes!")
