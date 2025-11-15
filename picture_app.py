import streamlit as st
import json
import yaml
import time
import base64
import hashlib
import requests
from io import BytesIO
from PIL import Image
from typing import Optional, Dict
from openai import OpenAI
import google.generativeai as genai
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Image Generation App",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# App title at the very top
st.title("🎨 AI Image Generation App")

# Add CSS to constrain image sizes to approximately 1/3 of screen width
st.markdown("""
<style>
    /* Constrain images in the first column (upload section) to max 33% of viewport width */
    div[data-testid="column"]:nth-of-type(1) .stImage > img {
        max-width: 33vw !important;
        width: auto !important;
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Load config from YAML file
@st.cache_data
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config or {}
    except FileNotFoundError:
        st.error("config.yaml file not found. Please create it with your API keys.")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing config.yaml: {e}")
        return {}

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = {}
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'pet_info' not in st.session_state:
    st.session_state.pet_info = None
if 'image_hash' not in st.session_state:
    st.session_state.image_hash = None

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def detect_pets_in_image(image: Image.Image, api_key: str, model: str = "gpt-4o") -> Optional[Dict]:
    """Detect and describe cats or dogs in the uploaded image using OpenAI Vision API"""
    try:
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        
        # Convert image to base64
        img_base64 = encode_image_to_base64(image)
        
        prompt_text = """Analyze this image and check if it contains any cats or dogs. 
If there are cats or dogs present, provide a detailed description of each one including:
- Species (cat or dog)
- Breed (if identifiable)
- Color/pattern of fur
- Size/appearance
- Any distinctive features
- Position in the image

If no cats or dogs are present, respond with "NO_PETS".

Format your response as JSON with this structure:
{
  "has_pets": true/false,
  "pets": [
    {
      "species": "cat" or "dog",
      "breed": "breed name or unknown",
      "description": "detailed description of appearance and features"
    }
  ]
}"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        pet_info = json.loads(result_text)
        
        return pet_info if pet_info.get("has_pets", False) else None
        
    except Exception as e:
        st.error(f"Error detecting pets with OpenAI: {str(e)}")
        return None

def detect_pets_in_image_gemini(image: Image.Image, api_key: str, model: str = "gemini-2.0-flash-exp") -> Optional[Dict]:
    """Detect and describe cats or dogs in the uploaded image using Gemini Vision API"""
    try:
        if not api_key:
            return None
        
        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel(model)
        
        prompt_text = """Analyze this image and check if it contains any cats or dogs. 
If there are cats or dogs present, provide a detailed description of each one including:
- Species (cat or dog)
- Breed (if identifiable)
- Color/pattern of fur
- Size/appearance
- Any distinctive features
- Position in the image

If no cats or dogs are present, respond with "NO_PETS".

Format your response as JSON with this structure:
{
  "has_pets": true/false,
  "pets": [
    {
      "species": "cat" or "dog",
      "breed": "breed name or unknown",
      "description": "detailed description of appearance and features"
    }
  ]
}

Return ONLY valid JSON, no other text."""
        
        response = genai_model.generate_content([prompt_text, image])
        result_text = response.text.strip()
        
        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        pet_info = json.loads(result_text)
        
        return pet_info if pet_info.get("has_pets", False) else None
        
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            st.warning("❌ Gemini API key is invalid.")
            return None
        else:
            st.error(f"Error detecting pets with Gemini: {error_msg}")
            return None

def enhance_prompt_with_pet_info(user_prompt: str, pet_info: Optional[Dict]) -> str:
    """Enhance the user's prompt to preserve specific pets from the original image"""
    if not pet_info or not pet_info.get("has_pets") or not pet_info.get("pets"):
        return user_prompt
    
    enhanced_prompt = user_prompt
    
    # Build pet preservation instructions
    pet_descriptions = []
    for pet in pet_info.get("pets", []):
        species = pet.get("species", "")
        breed = pet.get("breed", "unknown breed")
        description = pet.get("description", "")
        
        pet_desc = f"a {species}"
        if breed and breed != "unknown":
            pet_desc += f" ({breed})"
        if description:
            pet_desc += f" with {description}"
        
        pet_descriptions.append(pet_desc)
    
    if pet_descriptions:
        pet_instruction = f"IMPORTANT: The generated image must include the exact same {', '.join(pet_descriptions)} from the original uploaded image. Do not create a different or new {pet_info['pets'][0].get('species', 'pet')}. Preserve the specific appearance, colors, and features of the original pet(s)."
        enhanced_prompt = f"{user_prompt}\n\n{pet_instruction}"
    
    return enhanced_prompt

def generate_image_with_openai(prompt: str, api_key: str, model: str = "dall-e-3", size: str = "1024x1024") -> Optional[Image.Image]:
    """Generate image using OpenAI DALL-E API"""
    try:
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        
        # Download the image
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        
        return img
    except Exception as e:
        st.error(f"Error generating image with OpenAI: {str(e)}")
        return None

def generate_image_with_gemini(prompt: str, api_key: str) -> Optional[Image.Image]:
    """Generate image using Gemini API (if available)"""
    try:
        if not api_key:
            return None
        
        genai.configure(api_key=api_key)
        
        # Note: As of now, Gemini's generativeai library doesn't directly support image generation
        # This is a placeholder for future implementation
        # You might need to use Vertex AI or other Google Cloud services for Imagen
        
        # For now, return None and show a message
        st.info("ℹ️ Image generation with Gemini is currently not directly available through the generativeai library. "
                "You may need to use Google Cloud Vertex AI with Imagen for image generation.")
        return None
    except Exception as e:
        st.error(f"Error generating image with Gemini: {str(e)}")
        return None


def main():
    # Load config
    config = load_config()
    
    # Extract API keys and model names
    api_keys = config.get('api_keys', {})
    models = config.get('models', {})
    
    openai_key = api_keys.get('openai', '')
    gemini_key = api_keys.get('gemini', '')
    
    # Extract model names with defaults
    openai_image_model = models.get('openai', {}).get('image', 'dall-e-3')
    openai_vision_model = models.get('openai', {}).get('vision', 'gpt-4o')
    gemini_vision_model = models.get('gemini', {}).get('vision', 'gemini-2.0-flash-exp')
    
    # Check for missing API keys
    has_openai_key = bool(openai_key) and openai_key != "your_openai_api_key_here"
    has_gemini_key = bool(gemini_key) and gemini_key != "your_gemini_api_key_here"
    
    if not has_openai_key and not has_gemini_key:
        st.warning("⚠️ Please provide at least one API key in config.yaml to use this app.")
    
    # Configure Gemini if key is available
    if has_gemini_key:
        genai.configure(api_key=gemini_key)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image that will be used as the base for generation"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Calculate hash of image to detect if it changed
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            # Reset pet info if this is a new image
            if st.session_state.image_hash != image_hash:
                st.session_state.pet_info = None
                st.session_state.image_hash = image_hash
                st.session_state.uploaded_image = image
            
            # Display image - CSS will constrain it to approximately 1/3 of screen width
            st.image(image, caption="Uploaded Image")
            
            # Detect pets in the image
            if st.session_state.pet_info is None:
                with st.spinner("🔍 Detecting pets in image..."):
                    pet_info = None
                    if has_openai_key:
                        pet_info = detect_pets_in_image(image, openai_key, openai_vision_model)
                    elif has_gemini_key:
                        pet_info = detect_pets_in_image_gemini(image, gemini_key, gemini_vision_model)
                    
                    st.session_state.pet_info = pet_info
                    
                    if pet_info and pet_info.get("has_pets"):
                        st.success(f"✅ Detected {len(pet_info.get('pets', []))} pet(s) in the image!")
                        for idx, pet in enumerate(pet_info.get('pets', []), 1):
                            st.info(f"**Pet {idx}:** {pet.get('species', '').title()} - {pet.get('breed', 'Unknown breed')} - {pet.get('description', '')}")
                    else:
                        st.info("ℹ️ No cats or dogs detected in the image.")
            
            # Show detected pet info if available
            if st.session_state.pet_info and st.session_state.pet_info.get("has_pets"):
                st.markdown("### 🐾 Detected Pets")
                for idx, pet in enumerate(st.session_state.pet_info.get('pets', []), 1):
                    st.markdown(f"**{pet.get('species', '').title()} {idx}:** {pet.get('breed', 'Unknown breed')} - {pet.get('description', '')}")
                
                if st.button("Re-detect Pets"):
                    st.session_state.pet_info = None
                    st.rerun()
    
    with col2:
        st.markdown("### ✏️ Prompt & Generation")
        
        # Prompt input
        prompt = st.text_area(
            "Enter your image generation prompt:",
            placeholder="e.g., A futuristic cyberpunk cityscape with neon lights, flying cars, and towering skyscrapers at night",
            help="Enter a detailed prompt describing the image you want to generate",
            value=st.session_state.current_prompt
        )
        st.session_state.current_prompt = prompt
        
        # Generation options
        st.markdown("### 🎨 Generate Image")
        
        selected_generators = st.multiselect(
            "Choose AI models for image generation:",
            options=["OpenAI DALL-E"],
            default=["OpenAI DALL-E"] if has_openai_key else []
        )
        
        if not has_openai_key and "OpenAI DALL-E" in selected_generators:
            st.warning("⚠️ OpenAI API key required for DALL-E generation.")
            selected_generators = []
        
        # Generate button
        if st.button("Generate Images", type="primary", disabled=not st.session_state.uploaded_image):
            if not st.session_state.uploaded_image:
                st.error("Please upload an image first.")
            elif not prompt.strip():
                st.warning("Please enter a prompt to guide the image generation.")
            else:
                # Enhance prompt with pet information if pets were detected
                final_prompt = enhance_prompt_with_pet_info(prompt, st.session_state.pet_info)
                
                # Show the enhanced prompt if it was modified
                if final_prompt != prompt:
                    with st.expander("View Enhanced Prompt (with pet preservation)"):
                        st.text(final_prompt)
                
                # Generate images using the enhanced prompt
                progress_bar = st.progress(0)
                status_text = st.empty()
                generated_images = {}
                
                total_generators = len(selected_generators)
                current_generator = 0
                
                for generator in selected_generators:
                    current_generator += 1
                    progress_bar.progress(current_generator / total_generators)
                    status_text.text(f"🔄 Generating image with {generator}... ({current_generator}/{total_generators})")
                    
                    start_time = time.time()
                    
                    if generator == "OpenAI DALL-E":
                        generated_img = generate_image_with_openai(
                            final_prompt,
                            openai_key,
                            openai_image_model
                        )
                        
                        if generated_img:
                            generated_images[generator] = generated_img
                            generation_time = time.time() - start_time
                            status_text.text(f"✅ {generator} completed in {generation_time:.1f}s ({current_generator}/{total_generators})")
                    
                    elif generator == "Gemini":
                        generated_img = generate_image_with_gemini(
                            final_prompt,
                            gemini_key
                        )
                        
                        if generated_img:
                            generated_images[generator] = generated_img
                            generation_time = time.time() - start_time
                            status_text.text(f"✅ {generator} completed in {generation_time:.1f}s ({current_generator}/{total_generators})")
                
                # Store generated images
                st.session_state.generated_images = generated_images
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text("🎉 Image generation completed!")
    
    # Display generated images
    if st.session_state.generated_images:
        st.markdown("---")
        st.markdown("### 🖼️ Generated Images")
        
        num_images = len(st.session_state.generated_images)
        # Use 3 columns to ensure each image takes about 1/3 width
        # If we have fewer images, they'll still be properly sized
        cols = st.columns(min(3, num_images) if num_images > 0 else 1)
        
        for idx, (generator_name, img) in enumerate(st.session_state.generated_images.items()):
            col_idx = idx % len(cols)  # Cycle through available columns
            with cols[col_idx]:
                st.markdown(f"#### {generator_name}")
                # Images in columns will fill their column (already 1/3 width with 3 columns)
                st.image(img, use_container_width=True)
                
                # Download button for each image
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{generator_name.lower().replace(' ', '_')}_{timestamp}.png"
                
                st.download_button(
                    label=f"Download {generator_name}",
                    data=img_bytes,
                    file_name=filename,
                    mime="image/png",
                    key=f"download_{generator_name}_{timestamp}"
                )
        
        # Download all as JSON (metadata)
        if st.button("Download Metadata"):
            metadata = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": st.session_state.current_prompt,
                "pet_info": st.session_state.pet_info,
                "generators_used": list(st.session_state.generated_images.keys())
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generation_metadata_{timestamp}.json"
            
            st.download_button(
                label="Download Metadata JSON",
                data=json.dumps(metadata, indent=2),
                file_name=filename,
                mime="application/json",
                key=f"download_metadata_{timestamp}"
            )

if __name__ == "__main__":
    main()

