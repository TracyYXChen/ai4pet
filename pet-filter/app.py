import streamlit as st
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

st.set_page_config(page_title="Cat Vision Camera Filter", layout="wide")

st.title("🐱 Cat Vision Camera Filter (Approximation)")
st.caption("Dichromatic-ish color, lower acuity blur, low-light lift + grain, optional motion pop.")

with st.sidebar:
    st.header("Tuning")
    blur_sigma = st.slider("Blur (acuity loss) sigma", 0.0, 6.0, 2.0, 0.1)
    desat = st.slider("Desaturation", 0.0, 0.6, 0.15, 0.01)
    lowlight_strength = st.slider("Low-light lift strength", 0.0, 1.0, 0.8, 0.05)
    noise_base = st.slider("Noise base amount", 0.0, 0.10, 0.02, 0.005)
    noise_extra = st.slider("Extra noise in dark scenes", 0.0, 0.20, 0.06, 0.005)
    enable_motion = st.checkbox("Enable motion pop", value=True)
    motion_gain = st.slider("Motion gain", 0.0, 10.0, 5.0, 0.5)
    motion_boost = st.slider("Motion boost amount", 0.0, 0.20, 0.05, 0.01)

# Put params into session state so transformer can read them
st.session_state["cat_params"] = dict(
    blur_sigma=blur_sigma,
    desat=desat,
    lowlight_strength=lowlight_strength,
    noise_base=noise_base,
    noise_extra=noise_extra,
    enable_motion=enable_motion,
    motion_gain=motion_gain,
    motion_boost=motion_boost,
)


def apply_cat_filter(frame_bgr: np.ndarray, params: dict, prev_gray: np.ndarray = None):
    """
    Apply cat vision filter to a BGR image.
    Returns (filtered_bgr, gray) where gray can be used as prev_gray for motion detection.
    """
    # BGR -> RGB float 0..1
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # 1) Dichromacy-ish transform (fast approximation)
    r2 = 0.15 * r + 0.35 * g + 0.50 * b
    g2 = 0.10 * r + 0.80 * g + 0.10 * b
    b2 = 0.05 * r + 0.25 * g + 0.70 * b
    rgb2 = np.stack([r2, g2, b2], axis=-1)

    # Luma (for desat + low-light adaptation)
    luma = 0.2126 * rgb2[..., 0] + 0.7152 * rgb2[..., 1] + 0.0722 * rgb2[..., 2]

    # Slight desaturation
    d = float(params["desat"])
    rgb2 = rgb2 * (1.0 - d) + luma[..., None] * d

    # 2) Reduced acuity blur
    sigma = float(params["blur_sigma"])
    if sigma > 0.0:
        rgb2 = cv2.GaussianBlur(rgb2, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 3) Low-light lift + adaptive noise
    avg_luma = float(luma.mean())
    # lift goes 0..1 as scene gets darker than ~0.55
    lift = np.clip((0.55 - avg_luma) / 0.55, 0.0, 1.0) * float(params["lowlight_strength"])
    gamma = 1.0 - 0.35 * lift  # down to ~0.65 in dark scenes
    rgb2 = np.power(np.clip(rgb2, 0, 1), gamma)

    noise_amt = float(params["noise_base"]) + float(params["noise_extra"]) * lift
    if noise_amt > 0.0:
        noise = (np.random.rand(*luma.shape).astype(np.float32) - 0.5) * noise_amt
        rgb2 = np.clip(rgb2 + noise[..., None], 0, 1)

    # 4) Optional motion emphasis (temporal difference)
    gray = (0.299 * rgb2[..., 0] + 0.587 * rgb2[..., 1] + 0.114 * rgb2[..., 2]).astype(np.float32)
    if params["enable_motion"] and prev_gray is not None:
        diff = np.abs(gray - prev_gray)
        diff = np.clip(diff * float(params["motion_gain"]), 0, 1)
        rgb2 = np.clip(rgb2 + diff[..., None] * float(params["motion_boost"]), 0, 1)

    out_rgb = (rgb2 * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr, gray


class CatVisionTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_gray = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        # Read latest params from Streamlit session state (fallback if missing)
        params = st.session_state.get("cat_params", {})
        if not params:
            params = dict(
                blur_sigma=2.0, desat=0.15, lowlight_strength=0.8,
                noise_base=0.02, noise_extra=0.06,
                enable_motion=True, motion_gain=5.0, motion_boost=0.05
            )

        filtered, self.prev_gray = apply_cat_filter(img, params, self.prev_gray)
        return filtered


# Input mode selection
st.subheader("Choose Input Mode")
input_mode = st.radio(
    "Select how you want to provide the image:",
    options=["📷 Web Camera", "🖼️ Upload Picture"],
    horizontal=True,
    label_visibility="collapsed"
)

if input_mode == "📷 Web Camera":
    st.subheader("Live Camera Feed")
    webrtc_streamer(
        key="cat-vision",
        video_transformer_factory=CatVisionTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.info("Tip: If the feed is black, check browser camera permission and try a different browser (Chrome works best).")

else:  # Upload Picture
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Upload an image to apply the cat vision filter"
    )
    
    if uploaded_file is not None:
        # Load image
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply filter (no motion detection for static images)
        params = st.session_state.get("cat_params", {})
        params_static = params.copy()
        params_static["enable_motion"] = False  # Disable motion for static images
        
        filtered_bgr, _ = apply_cat_filter(img_bgr, params_static)
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        # Display side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(pil_image, use_container_width=True)
        with col2:
            st.markdown("**Cat Vision Filter Applied**")
            st.image(filtered_rgb, use_container_width=True)
        
        # Download button for filtered image
        filtered_pil = Image.fromarray(filtered_rgb)
        import io
        buf = io.BytesIO()
        filtered_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Filtered Image",
            data=buf.getvalue(),
            file_name="cat_vision_filtered.png",
            mime="image/png"
        )
    else:
        st.info("👆 Upload an image above to see how it looks through cat eyes!")
