# 🐱 Cat Vision Camera Filter

See the world through a cat's eyes! This Streamlit app applies a real-time filter that approximates how cats perceive vision, including:

- **Dichromatic color vision** - Cats see fewer colors than humans
- **Reduced acuity** - Slightly blurred vision compared to human 20/20
- **Low-light adaptation** - Enhanced visibility in dark scenes with added grain
- **Motion emphasis** - Cats are highly sensitive to movement

## Features

- **📷 Web Camera Mode** - Apply the filter to your live webcam feed in real-time
- **🖼️ Upload Picture Mode** - Upload any image and see it through cat eyes
- **Adjustable Parameters** - Fine-tune blur, desaturation, low-light effects, and motion sensitivity
- **Download Filtered Images** - Save your cat-vision images as PNG

## Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:

```bash
source $HOME/.local/bin/env
```

### Set Up the Environment

Navigate to this directory and sync dependencies:

```bash
cd pet-filter
uv sync
```

This creates a `.venv` virtual environment and installs all required packages.

## Running the App

```bash
uv run streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. **Choose Input Mode** - Select either "Web Camera" or "Upload Picture"
2. **Adjust Filters** - Use the sidebar sliders to tune the cat vision effect:
   - **Blur sigma** - Controls visual acuity loss
   - **Desaturation** - Reduces color saturation
   - **Low-light lift** - Brightens dark scenes
   - **Noise** - Adds grain, especially in dark conditions
   - **Motion pop** - Highlights moving objects (webcam only)
3. **Download** - In upload mode, download the filtered image with the button

## Troubleshooting

### Webcam not working?

- Check browser camera permissions
- Try Chrome (works best with WebRTC)
- Ensure no other app is using the camera

### Black screen in webcam mode?

- Refresh the page
- Click "START" if the stream hasn't begun
- Try a different browser

