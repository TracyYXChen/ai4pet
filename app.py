import streamlit as st
import json
import re
import yaml
import time
from typing import List, Dict, Any
from openai import OpenAI
import google.generativeai as genai
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Pet Toy Recommendation Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# App title at the very top
st.title("LLM responses tracking for pet toys🐾 ")

# Load toy data
@st.cache_data
def load_toy_data():
    with open('toy_data.json', 'r') as f:
        return json.load(f)

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
if 'selected_species' not in st.session_state:
    st.session_state.selected_species = None
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None
if 'model_responses' not in st.session_state:
    st.session_state.model_responses = {}

def extract_brands_products_openai(text: str, api_key: str) -> List[str]:
    """Extract brand and product names using OpenAI API"""
    try:
        if not api_key:
            return []
        
        client = OpenAI(api_key=api_key)
        
        prompt = f"""
        Extract all brand names and product names mentioned in the following text about pet toys. 
        Return ONLY the brand/product names as a comma-separated list, nothing else.
        
        Text: {text}
        
        Examples of what to extract:
        - Brand names: Kong, Nylabone, West Paw, Chuckit, PetSafe, Catit, etc.
        - Product names: Interactive toys, Puzzle feeders, Laser pointers, etc.
        - Specific products: Kong Classic, Nylabone DuraChew, etc.
        
        Return format: Brand1, Brand2, Product1, Product2
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at extracting brand and product names from text. Return only the names as a comma-separated list."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        # Split by comma and clean up
        items = [item.strip() for item in result.split(',') if item.strip()]
        return items
        
    except Exception as e:
        st.error(f"Error extracting brands with OpenAI: {str(e)}")
        return []

def extract_brands_products_gemini(text: str, api_key: str, model: str) -> List[str]:
    """Extract brand and product names using Gemini API"""
    try:
        if not api_key:
            return []

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model)
        
        prompt = f"""
        Extract all brand names and product names mentioned in the following text about pet toys. 
        Return ONLY the brand/product names as a comma-separated list, nothing else.
        
        Text: {text}
        
        Examples of what to extract:
        - Brand names: Kong, Nylabone, West Paw, Chuckit, PetSafe, Catit, etc.
        - Product names: Interactive toys, Puzzle feeders, Laser pointers, etc.
        - Specific products: Kong Classic, Nylabone DuraChew, etc.
        
        Return format: Brand1, Brand2, Product1, Product2
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Split by comma and clean up
        items = [item.strip() for item in result.split(',') if item.strip()]
        return items
        
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            st.warning("❌ Gemini API key is invalid. Brand extraction skipped.")
            return []
        else:
            st.error(f"Error extracting brands with Gemini: {error_msg}")
            return []

def generate_response_with_openai_stream(question: str, references: List[Dict], model: str, api_key: str = None, container=None):
    """Generate streaming response using OpenAI API"""
    try:
        if not api_key:
            if container:
                container.error("OpenAI API key not provided")
            return
        
        client = OpenAI(api_key=api_key)
        
        # Format references for context
        ref_text = "\n".join([f"- {ref['title']}: {ref['link']}" for ref in references])
        
        prompt = f"""
        Based on the following question and references, provide a comprehensive answer about pet toys:
        
        Question: {question}
        
        References:
        {ref_text}
        
        Please provide a detailed response with specific toy recommendations, safety considerations, and practical advice.
        """
        
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable pet care expert specializing in pet toys and enrichment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
            stream=True
        )
        
        response_text = ""
        if container:
            response_placeholder = container.empty()
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
                if container:
                    response_placeholder.markdown(response_text)
        
        return response_text
        
    except Exception as e:
        error_msg = f"Error generating response with OpenAI: {str(e)}"
        if container:
            container.error(error_msg)
        return error_msg

def generate_response_with_gemini(question: str, references: List[Dict], model: str) -> str:
    """Generate response using Google Gemini API"""
    try:
        # Format references for context
        ref_text = "\n".join([f"- {ref['title']}: {ref['link']}" for ref in references])
        
        prompt = f"""
        Based on the following question and references, provide a comprehensive answer about pet toys:
        
        Question: {question}
        
        References:
        {ref_text}
        
        Please provide a detailed response with specific toy recommendations, safety considerations, and practical advice.
        """
        
        genai_model = genai.GenerativeModel(model)
        response = genai_model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            return "❌ Gemini API key is invalid. Please check your config.yaml file."
        else:
            return f"Error generating response with Gemini: {error_msg}"

def main():
    # Load data and config
    toy_data = load_toy_data()
    config = load_config()
    
    # Debug: Print config
    print("Full config:", config)
    
    # Extract API keys and model names
    api_keys = config.get('api_keys', {})
    models = config.get('models', {})
    
    print("API keys from config:", api_keys)
    print("Models from config:", models)
    
    openai_key = api_keys.get('openai', '')
    gemini_key = api_keys.get('gemini', '')
    
    # Extract model names with error handling
    openai_response_model = models.get('openai', {}).get('response')
    openai_extraction_model = models.get('openai', {}).get('extraction')
    gemini_response_model = models.get('gemini', {}).get('response')
    gemini_extraction_model = models.get('gemini', {}).get('extraction')
    
    # Check for missing model configurations
    if not openai_response_model:
        st.error("❌ OpenAI response model not specified in config.yaml")
    if not openai_extraction_model:
        st.error("❌ OpenAI extraction model not specified in config.yaml")
    if not gemini_response_model:
        st.error("❌ Gemini response model not specified in config.yaml")
    if not gemini_extraction_model:
        st.error("❌ Gemini extraction model not specified in config.yaml")
    
    print("Extracted values:")
    print(f"  openai_key: {openai_key[:20]}..." if openai_key else "  openai_key: None")
    print(f"  gemini_key: {gemini_key[:20]}..." if gemini_key else "  gemini_key: None")
    print(f"  openai_response_model: {openai_response_model}")
    print(f"  openai_extraction_model: {openai_extraction_model}")
    print(f"  gemini_response_model: {gemini_response_model}")
    print(f"  gemini_extraction_model: {gemini_extraction_model}")
    
    # Configuration row - all three dropdowns
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        species_options = ["cat", "dog"]
        selected_species = st.selectbox(
            "Choose your pet:",
            options=species_options,
            index=species_options.index(st.session_state.selected_species) if st.session_state.selected_species else 0
        )
        
        if selected_species != st.session_state.selected_species:
            st.session_state.selected_species = selected_species
            st.session_state.selected_question = None
            st.session_state.model_responses = {}
    
    with col2:
        selected_models = st.multiselect(
            "Choose AI models:",
            options=["GPT-4", "Gemini"],
            default=["GPT-4", "Gemini"]
        )
    
    with col3:
        if selected_species:
            # Filter questions by species
            species_questions = [item for item in toy_data if item['species'] == selected_species]
            
            # Question selection
            question_options = [f"{q['id']}. {q['question']}" for q in species_questions]
            selected_question_text = st.selectbox(
                "Select a question:",
                options=question_options,
                index=0
            )
        else:
            st.selectbox(
                "Select a question:",
                options=["Please select a pet first"],
                disabled=True
            )
    
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        genai.configure(api_key=gemini_key)
        
    # Main content area
    if selected_species and 'selected_question_text' in locals() and selected_question_text != "Please select a pet first":
        # Extract question ID and get full question data
        question_id = int(selected_question_text.split('.')[0])
        selected_question_data = next(q for q in species_questions if q['id'] == question_id)
        
        st.session_state.selected_question = selected_question_data
        
        
        # Add CSS to reduce spacing
        st.markdown("""
        **References:**
        <style>
        .references-container p {
            line-height: 1.2 !important;
        }
        .references-container {
            margin-bottom: 0.5em !important;
        }
        .stMarkdown {
            margin-bottom: 0.5em !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            # Build all references in a single markdown string
            references_text = ""
            for i, ref in enumerate(selected_question_data['references'], 1):
                references_text += f"{i}. {ref['title']} - [{ref['link']}]({ref['link']})\n"
            
            st.markdown(references_text)
        
        # Generate responses section
        
        # Add styling for blue button
        st.markdown("""
        <style>
        .stButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
        }
        .stButton > button:hover {
            background-color: #1565c0 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
            
        # Check if API keys are provided
        has_openai_key = bool(openai_key) and openai_key != "your_openai_api_key_here"
        has_gemini_key = bool(gemini_key) and gemini_key != "your_gemini_api_key_here"
        
        print("API Key Validation:")
        print(f"  openai_key exists: {bool(openai_key)}")
        print(f"  openai_key is not placeholder: {openai_key != 'your_openai_api_key_here'}")
        print(f"  has_openai_key: {has_openai_key}")
        print(f"  gemini_key exists: {bool(gemini_key)}")
        print(f"  gemini_key is not placeholder: {gemini_key != 'your_gemini_api_key_here'}")
        print(f"  has_gemini_key: {has_gemini_key}")
        
        if not has_openai_key and not has_gemini_key:
            st.warning("Please provide at least one API key to generate responses.")
        else:
            # Generate button
            if st.button("Generate AI Responses", type="secondary"):
                # Calculate total number of API calls (2 per model: response + extraction)
                total_calls = len(selected_models) * 2
                current_call = 0
                
                # Create progress bar right below the button
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Track total time
                total_start_time = time.time()
                
                responses = {}
                
                # Create layout with scrollable sections
                response_cols = st.columns(len(selected_models))
                response_containers = {}
                brand_containers = {}
                
                # Initialize containers for each model with scrollable sections
                for idx, model in enumerate(selected_models):
                    with response_cols[idx]:
                        # Response section with fixed height scrollable container
                        st.markdown(f"### 📝 {model} Response")
                        with st.container(height=300):
                            response_containers[model] = st.empty()
                        
                        # Brands section with fixed height scrollable container
                        st.markdown(f"### 🏷️ Brands/Products ({model})")
                        with st.container(height=200):
                            brand_containers[model] = st.empty()
                
                # Generate responses sequentially (Streamlit doesn't support threading well)
                responses = {}
                
                for model in selected_models:
                    if model in ["GPT-3.5", "GPT-4"] and has_openai_key and openai_response_model:
                        # Update progress for response generation
                        current_call += 1
                        progress_bar.progress(current_call / total_calls)
                        status_text.text(f"🔄 Generating {model} response... ({current_call}/{total_calls})")
                        
                        start_time = time.time()
                        response = generate_response_with_openai_stream(
                            selected_question_data['question'],
                            selected_question_data['references'],
                            openai_response_model,
                            openai_key,
                            response_containers[model]
                        )
                        response_time = time.time() - start_time
                        print(f"⏱️ {model} response generation took {response_time:.2f} seconds")
                        status_text.text(f"✅ {model} response completed in {response_time:.1f}s ({current_call}/{total_calls})")
                        responses[model] = response
                            
                    elif model == "Gemini" and has_gemini_key and gemini_response_model:
                        # Update progress for response generation
                        current_call += 1
                        progress_bar.progress(current_call / total_calls)
                        status_text.text(f"🔄 Generating {model} response... ({current_call}/{total_calls})")
                        
                        start_time = time.time()
                        response = generate_response_with_gemini(
                            selected_question_data['question'],
                            selected_question_data['references'],
                            gemini_response_model
                        )
                        response_time = time.time() - start_time
                        print(f"⏱️ {model} response generation took {response_time:.2f} seconds")
                        status_text.text(f"✅ {model} response completed in {response_time:.1f}s ({current_call}/{total_calls})")
                        responses[model] = response
                        # Display Gemini response in its container
                        response_containers[model].markdown(response)
                    
                    st.session_state.model_responses = responses
                
                # Extract brands after responses are complete
                if st.session_state.model_responses:
                    for model_name, response in st.session_state.model_responses.items():
                        # Update progress for brand extraction
                        current_call += 1
                        progress_bar.progress(current_call / total_calls)
                        status_text.text(f"🏷️ Extracting brands from {model_name}... ({current_call}/{total_calls})")
                        
                        # Extract brands/products using LLM
                        extracted_items = []
                        
                        # Use the same API as the response model
                        if model_name in ["GPT-4", "GPT-3.5"] and has_openai_key:
                            start_time = time.time()
                            extracted_items = extract_brands_products_openai(response, openai_key)
                            extraction_time = time.time() - start_time
                            print(f"⏱️ {model_name} brand extraction took {extraction_time:.2f} seconds")
                            status_text.text(f"✅ {model_name} brands extracted in {extraction_time:.1f}s ({current_call}/{total_calls})")
                        elif model_name == "Gemini" and has_gemini_key and gemini_extraction_model:
                            start_time = time.time()
                            extracted_items = extract_brands_products_gemini(response, gemini_key, gemini_extraction_model)
                            extraction_time = time.time() - start_time
                            print(f"⏱️ {model_name} brand extraction took {extraction_time:.2f} seconds")
                            status_text.text(f"✅ {model_name} brands extracted in {extraction_time:.1f}s ({current_call}/{total_calls})")
                        
                        # Update brand container
                        if extracted_items:
                            brand_text = "\n".join([f"• {item}" for item in extracted_items])
                            brand_containers[model_name].markdown(brand_text)
                        else:
                            brand_containers[model_name].info(f"No brands/products detected")
                    
                    # Complete progress bar and show total time
                    total_time = time.time() - total_start_time
                    progress_bar.progress(1.0)
                    status_text.text(f"🎉 All tasks completed in {total_time:.1f}s!")
                    
                    st.markdown("---")
                
                # Download responses
                if st.session_state.model_responses:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pet_toy_responses_{selected_species}_{timestamp}.json"
                    
                    download_data = {
                        "question": selected_question_data['question'],
                        "species": selected_species,
                        "timestamp": timestamp,
                        "responses": st.session_state.model_responses
                    }
                    
                    st.download_button(
                        label="Download Responses",
                        data=json.dumps(download_data, indent=2),
                        file_name=filename,
                        mime="application/json"
                    )
    
    else:
        st.info("Please select a pet species from the sidebar to get started.")
    

if __name__ == "__main__":
    main()
