# Pet Toy Recommendation Dashboard

A Streamlit-based dashboard that provides AI-powered pet toy recommendations for cats and dogs based on community insights and expert advice.

## Features

- **Species Selection**: Choose between cat and dog recommendations
- **Question Library**: Browse curated questions about pet toys with references
- **AI Model Support**: Generate responses using GPT-3.5, GPT-4, or Gemini Pro
- **Brand Extraction**: Automatically extract mentioned brands and product names
- **Reference Links**: Access to Reddit discussions and expert sources
- **Response Download**: Export AI responses as JSON files

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   - Copy `config.yaml` and replace the placeholder values:
     ```yaml
     api_keys:
       openai: "your_actual_openai_api_key"
       gemini: "your_actual_gemini_api_key"
     ```
   - **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
   - **Google Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Select Pet Species**: Choose "cat" or "dog" from the sidebar
2. **Choose Question**: Select from the list of curated questions
3. **Configure Models**: Select which AI models to use (GPT-4, Gemini Pro)
4. **Check API Status**: Verify API keys are configured in the sidebar
5. **Generate Responses**: Click "Generate AI Responses" to get AI-powered recommendations
6. **Review Results**: View responses and extracted brand/product names
7. **Download**: Export responses as JSON files for future reference

## Data Source

The dashboard uses `toy_data.json` which contains:
- Curated questions about pet toys
- References to Reddit discussions and expert sources
- Species-specific recommendations (cat/dog)

## Brand Extraction

The system automatically extracts mentioned brands and products using pattern matching for:
- Popular pet toy brands (Kong, Nylabone, West Paw, etc.)
- Pet food brands (Whiskas, Purina, Royal Canin, etc.)
- Product categories (Interactive, Puzzle, Feeder, etc.)

## API Usage

- **OpenAI**: Uses ChatCompletion API for GPT models
- **Google Gemini**: Uses GenerativeModel API for Gemini Pro
- Responses are cached in session state for efficiency

## File Structure

```
ai4pet/
├── app.py              # Main Streamlit application
├── toy_data.json       # Question and reference data
├── config.yaml         # API keys configuration (create from template)
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Contributing

To add new questions or references, edit the `toy_data.json` file following the existing structure:

```json
{
  "id": 11,
  "species": "cat",
  "question": "Your question here",
  "references": [
    {
      "title": "Reference title",
      "link": "https://example.com"
    }
  ]
}
```