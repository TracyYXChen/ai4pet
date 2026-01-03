import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2
import os

# 1. API Configuration
API_KEY = "Your-Gemini-API-Key-Here" 
genai.configure(api_key=API_KEY)

# --- HELPER FUNCTIONS ---

def reset_analysis():
    """
    CALLBACK: Clears old analysis when a new file is uploaded.
    This ensures Gemini runs automatically on the new image.
    """
    if 'pet_analysis_summary' in st.session_state:
        del st.session_state['pet_analysis_summary']
    if 'pet_species' in st.session_state:
        del st.session_state['pet_species']

def apply_personalized_filters(img_rgb, analysis_text, species_mode):
    """Adjusts visual parameters dynamically by parsing the Gemini analysis text."""
    img_np = np.array(img_rgb)
    h, w, _ = img_np.shape
    analysis_lower = analysis_text.lower()

    # Default visual baselines
    blur_sigma = 2.0
    warp_strength = 0.5 
    brightness_multiplier = 1.0

    # 1. Acuity (Blur): Adjusts for aging or lens cloudiness
    if any(word in analysis_lower for word in ["cloudy", "senior", "sclerosis", "cataract", "old"]):
        blur_sigma = 7.0 
    elif "near-sighted" in analysis_lower:
        blur_sigma = 5.0

    # 2. Perspective (Warp): Adjusts for skull type and eye placement
    if "brachycephalic" in analysis_lower or "flat-faced" in analysis_lower:
        warp_strength = 0.2
    elif "dolichocephalic" in analysis_lower or "long-snouted" in analysis_lower:
        warp_strength = 0.8

    # 3. Sensitivity: Adjusts for tapetum presence or pigment
    if "blue eyes" in analysis_lower or "lack of pigment" in analysis_lower:
        brightness_multiplier = 0.8
    elif "tapetum" in analysis_lower:
        brightness_multiplier = 1.3

    # --- APPLY TRANSFORMATIONS ---
    tilt = 0.4 * warp_strength
    src_pts = np.float32([[0, 0], [w, 0], [w * tilt, h], [w * (1-tilt), h]])
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    r, g, b = warped[:,:,0].astype(float), warped[:,:,1].astype(float), warped[:,:,2].astype(float)
    if species_mode == "Cat":
        rg_mix = (0.1 * r + 0.9 * g) * brightness_multiplier
        filtered = np.stack([rg_mix, rg_mix, b], axis=-1)
    else:
        yellow = (0.5 * r + 0.5 * g) * brightness_multiplier
        filtered = np.stack([yellow, yellow, b], axis=-1)

    final = cv2.GaussianBlur(np.clip(filtered, 0, 255).astype(np.uint8), (0, 0), sigmaX=blur_sigma)
    return final

def display_vision_explanation(analysis_text):
    """Provides a concise explanation of why the view is personalized."""
    st.write("---")
    st.subheader("🧪 The Science Behind This View")
    st.info("Unlike general species filters, this view is fine-tuned to your pet's specific anatomy identified by the AI.")
    
    col1, col2 = st.columns(2)
    analysis_lower = analysis_text.lower()
    
    with col1:
        st.markdown("**🦴 Physical Perspective**")
        if "brachycephalic" in analysis_lower:
            st.write("• **Flat-Faced Breed:** Front-facing eyes lead to better focus but less peripheral depth.")
        elif "dolichocephalic" in analysis_lower:
            st.write("• **Long-Snouted Breed:** The snout creates a specific visual blind spot and wider scanning range.")
        else:
            st.write("• **Standard Alignment:** Perspective is calculated on typical height and alignment.")

    with col2:
        st.markdown("**👁️ Visual Clarity**")
        if any(word in analysis_lower for word in ["cloudy", "senior", "sclerosis"]):
            st.write("• **Age-Related Blur:** Simulated lens clouding common in older pets.")
        elif "blue eyes" in analysis_lower:
            st.write("• **Pigment Sensitivity:** Blue eyes often lack a tapetum, making low-light vision unique.")
        else:
            st.write("• **Standard Acuity:** Clarity based on average retinal density for this species.")

# --- SESSION STATE INITIALIZATION ---
if 'pet_presets' not in st.session_state:
    st.session_state['pet_presets'] = {}

# NEW: Version counter to force-reset the uploader widget
if 'uploader_version' not in st.session_state:
    st.session_state['uploader_version'] = 0

st.title("🐾 AI Pet Feature & Vision Analysis")

# --- SIDEBAR: PRESETS & UPLOADER ---
with st.sidebar:
    st.header("📋 Saved Pet Presets")
    if st.session_state['pet_presets']:
        selected_name = st.selectbox("Switch pets:", list(st.session_state['pet_presets'].keys()))
        if st.button("Load Selected Pet"):
            preset = st.session_state['pet_presets'][selected_name]
            st.session_state['pet_image'] = preset['image']
            st.session_state['pet_analysis_summary'] = preset['analysis']
            st.session_state['pet_species'] = preset['species']
            st.rerun()
    
    st.write("---")
    st.subheader("🔄 Analyze a New Pet")
    # FIX: Using a dynamic key resets the uploader to empty when version increments
    new_pet_file = st.file_uploader(
        "Upload new pet photo", 
        type=['png', 'jpg', 'jpeg'], 
        key=f"pet_analysis_uploader_{st.session_state['uploader_version']}",
        on_change=reset_analysis
    )
    if new_pet_file:
        st.session_state['pet_image'] = Image.open(new_pet_file)
        st.rerun()
        
    st.write("---")
    # FIX: Restart button now resets uploader version and deletes image state
    if st.button("🔄 Restart & Clear Analysis", help="Clears memory to re-run AI analysis on the current image"):
        keys_to_reset = ['pet_analysis_summary', 'pet_species', 'pet_image']
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        # Increment version to "kill" the old uploader widget
        st.session_state['uploader_version'] += 1
        st.rerun()


# --- MAIN ANALYSIS BLOCK ---
if 'pet_image' in st.session_state:
    img = st.session_state['pet_image']
    # Requirement: Only show the image once, in thumbnail size
    st.image(img, caption="Target Pet Profile", width=200)
    
    st.write("---")
    st.subheader("👁️ Eye Feature Summary")

    # Only run if summary is missing (e.g., after an upload reset or manual clear)
    if 'pet_analysis_summary' not in st.session_state:
        with st.spinner("Gemini is analyzing your pet's features..."):
            try:
                # Maintain original model and prompt
                model = genai.GenerativeModel('gemini-3-flash-preview')
                prompt = """
                You are a veterinary ophthalmology expert. Analyze this pet image (cat or dog) and provide:
                
                1. **Species and Breed Profile:** Identify the pet (e.g., "Dog" or "Cat") and note if it is a brachycephalic (flat-faced) or dolichocephalic (long-snouted) breed.
                2. **Eye Feature Summary:** Detail the color, shape, and anatomical placement (frontal vs. lateral) of the eyes.
                3. **Unique Pathologies/Traits:** Identify heterochromia, specific pupil shapes (slit vs. round), or any visible cloudiness in the lens (indicating age or nuclear sclerosis).
                4. **Depth Perception Analysis:** Based on the breed's skull shape and eye placement, estimate the degree of binocular overlap and how it impacts their 3D depth perception compared to other breeds.
                5. **Clarity & Light Sensitivity:** - Assess how their specific eye color or breed might influence the tapetum lucidum efficiency (low-light vision).
                   - Determine if the pet likely sees world as a "near-sighted" or "far-sighted" observer based on their species and age indicators.
                6. **Color Perception:** Describe the specific color palette this pet likely sees (e.g., blues/yellows) and if their specific traits suggest a higher or lower sensitivity to certain hues.

                Please format your response in clear, concise bullet points for clarity.
                """
                response = model.generate_content([prompt, img])
                st.session_state['pet_analysis_summary'] = response.text
                st.session_state['pet_species'] = "Dog" if "dog" in response.text.lower() else "Cat"
            except Exception as e:
                st.error(f"AI Analysis failed: {e}")

    # Display Analysis
    if 'pet_analysis_summary' in st.session_state:
        st.markdown(st.session_state['pet_analysis_summary'])
        
        # Preset Saving UI
        p_name = st.text_input("Name to save as preset:", placeholder="e.g., Buddy")
        if st.button("💾 Save as Preset"):
            if p_name:
                st.session_state['pet_presets'][p_name] = {
                    "image": st.session_state['pet_image'],
                    "analysis": st.session_state['pet_analysis_summary'],
                    "species": st.session_state['pet_species']
                }
                st.success(f"Saved {p_name} to presets!")
        
        st.write("---")

        # Room Transformation Logic
        st.subheader("🏠 Personalized Room Transformation")
        room_source = st.session_state.get('selected_sample')
        # 2. Add the explanation message for the user
        st.info("💡 If you do not upload a new room photo, the system will use the original image selected on the home page by default.")
        
        # Room upload override
        new_room = st.file_uploader("🖼️ Upload a new room photo", type=['png', 'jpg', 'jpeg'], key="room_override")
        if new_room:
            room_source = new_room

        if room_source:
            if st.button("See Through Their Eyes 🪄"):
                room_img = Image.open(room_source).convert("RGB")
                custom_view = apply_personalized_filters(
                    room_img, 
                    st.session_state['pet_analysis_summary'],
                    species_mode=st.session_state['pet_species']
                )
                
                col1, col2 = st.columns(2)
                col1.image(room_img, caption="Human Perspective")
                col2.image(custom_view, caption=f"Personalized {st.session_state['pet_species']} View")
                # Requirement: Explanation of why the view is different
                display_vision_explanation(st.session_state['pet_analysis_summary'])
        else:
            st.warning("Please pick a room on the home page or upload one above.")

else:
    st.warning("No pet photo found. Please upload one in the sidebar to start!")