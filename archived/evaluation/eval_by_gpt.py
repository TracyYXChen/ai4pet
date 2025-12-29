import json
import os
import base64
import yaml
from openai import OpenAI

index2label = {
    0: "none",
    1: "alarmed",
    2: "angry",
    3: "calm",
    4: "pleased"
}

def load_config():
    """Load configuration from config.yaml"""
    config_path = "/Users/yuexichen/Desktop/code/ai4pet/config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def classify_image_with_openai(image_path: str, api_key: str, model: str = "gpt-4o") -> str:
    """Classify cat emotion in image using OpenAI Vision API"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Encode image to base64
        img_base64 = encode_image_to_base64(image_path)
        
        # Create prompt asking for one of the labels
        valid_labels = list(index2label.values())
        prompt_text = f"""Analyze this image of a cat and classify its emotional state. 
You must respond with EXACTLY one of these words: {', '.join(valid_labels)}

Choose the word that best describes the cat's emotional state:
- "none": No clear emotion or neutral state
- "alarmed": The cat appears scared, startled, or alarmed
- "angry": The cat appears aggressive, hostile, or angry
- "calm": The cat appears relaxed, peaceful, or calm
- "pleased": The cat appears happy, content, or pleased

Respond with ONLY the single word, nothing else."""

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
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        
        result_text = response.choices[0].message.content.strip().lower()
        
        # Validate that the result is one of the valid labels
        if result_text in valid_labels:
            return result_text
        # If the response doesn't match exactly, try to find the closest match
        for label in valid_labels:
            if label.lower() in result_text or result_text in label.lower():
                return label
        # Default to "none" if no match found
        return "none"
            
    except Exception as e:
        print(f"Error classifying image {image_path}: {str(e)}")
        return "none"

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load existing results from checkpoint file"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint file: {e}")
            return {}
    return {}

def save_checkpoint(checkpoint_path: str, results: dict):
    """Save results to checkpoint file"""
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save checkpoint file: {e}")

def eval_by_gpt():
    """Evaluate images using OpenAI API with checkpointing"""
    # Load config
    config = load_config()
    api_key = config.get('api_keys', {}).get('openai', '')
    vision_model = config.get('models', {}).get('openai', {}).get('vision', 'gpt-4o')
    
    if not api_key:
        raise ValueError("OpenAI API key not found in config.yaml")
    
    # Setup checkpoint file
    checkpoint_path = "/Users/yuexichen/Desktop/code/ai4pet/evaluation/eval_results_checkpoint.json"
    
    # Load existing results if checkpoint exists
    image2emotions_pred = load_checkpoint(checkpoint_path)
    print(f"Loaded {len(image2emotions_pred)} existing results from checkpoint")
    
    # Load annotation data
    imgpath2label = "/Users/yuexichen/Desktop/code/ai4pet/evaluation/domestic-cats/test/_annotations.coco.json"
    test_dir = "/Users/yuexichen/Desktop/code/ai4pet/evaluation/domestic-cats/test"
    
    with open(imgpath2label, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get ground truth labels
    annotations = data['annotations']
    image2emotions_gt = {}
    for annotation in annotations:
        annotation_img_id = annotation['image_id']
        annotation_label = index2label[annotation['category_id']]
        image2emotions_gt[annotation_img_id] = annotation_label
    
    # Get image paths
    imageid2path = {}
    for image_info in data['images']:
        image_info_id = image_info['id']
        image_info_filename = image_info['file_name']
        imageid2path[image_info_id] = image_info_filename
    
    # Classify each image using OpenAI (skip if already in checkpoint)
    total_images = len(imageid2path)
    processed_count = len(image2emotions_pred)
    
    print(f"\nProcessing {total_images} images ({processed_count} already processed, {total_images - processed_count} remaining)...")
    print("=" * 80)
    
    for idx, (image_id, image_filename) in enumerate(imageid2path.items(), 1):
        # Skip if already processed
        if str(image_id) in image2emotions_pred:
            print(f"[{idx}/{total_images}] Skipping {image_filename} (already processed)")
            continue
        
        full_image_path = os.path.join(test_dir, image_filename)
        if os.path.exists(full_image_path):
            print(f"[{idx}/{total_images}] Processing {image_filename}...", end=' ', flush=True)
            predicted_label = classify_image_with_openai(full_image_path, api_key, vision_model)
            image2emotions_pred[str(image_id)] = predicted_label
            print(f"→ {predicted_label}")
            
            # Save checkpoint after each image
            save_checkpoint(checkpoint_path, image2emotions_pred)
        else:
            print(f"[{idx}/{total_images}] Warning: Image not found: {full_image_path}")
            image2emotions_pred[str(image_id)] = "none"
            save_checkpoint(checkpoint_path, image2emotions_pred)
    
    # Convert string keys back to integers for compatibility
    image2emotions_pred_int = {int(k): v for k, v in image2emotions_pred.items()}
    
    print("\n" + "=" * 80)
    print("All images processed!")
    
    return image2emotions_gt, image2emotions_pred_int, imageid2path

if __name__ == "__main__":
    imageid2emotions_gt, imageid2emotions_pred, imageid2path = eval_by_gpt()
    
    print("\nFinal Results:")
    print("=" * 80)
    for image_id in imageid2path.keys():
        image_filename = imageid2path[image_id]
        gt_label = imageid2emotions_gt.get(image_id, "unknown")
        pred_label = imageid2emotions_pred.get(image_id, "unknown")
        match = "✓" if gt_label == pred_label else "✗"
        print(f"{match} {image_filename:50s} | GT: {gt_label:10s} | Pred: {pred_label:10s}")
    
    # Calculate accuracy
    correct = sum(1 for image_id in imageid2path.keys() 
                  if imageid2emotions_gt.get(image_id) == imageid2emotions_pred.get(image_id))
    total = len(imageid2path)
    accuracy = correct / total if total > 0 else 0
    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")