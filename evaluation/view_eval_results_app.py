import json
import os

import streamlit as st
from PIL import Image


INDEX2LABEL = {
    0: "none",
    1: "alarmed",
    2: "angry",
    3: "calm",
    4: "pleased",
}


ANNOTATIONS_PATH = "/Users/yuexichen/Desktop/code/ai4pet/evaluation/domestic-cats/test/_annotations.coco.json"
IMAGES_DIR = "/Users/yuexichen/Desktop/code/ai4pet/evaluation/domestic-cats/test"
CHECKPOINT_PATH = "/Users/yuexichen/Desktop/code/ai4pet/evaluation/eval_results_checkpoint.json"


@st.cache_data
def load_data():
    """Load COCO annotations and prediction checkpoint."""
    # Load annotations (ground truth + image paths)
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Map image_id -> file_name
    imageid2path = {img["id"]: img["file_name"] for img in data["images"]}

    # Map image_id -> ground truth label (string using INDEX2LABEL)
    imageid2gt = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        cat_id = ann["category_id"]
        imageid2gt[image_id] = INDEX2LABEL.get(cat_id, "unknown")

    # Load predictions
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            preds_raw = json.load(f)
    else:
        preds_raw = {}

    # Keys in checkpoint are strings, convert to int
    imageid2pred = {int(k): v for k, v in preds_raw.items()}

    # Build a list of records for easier display
    records = []
    for image_id, filename in imageid2path.items():
        gt = imageid2gt.get(image_id, "unknown")
        pred = imageid2pred.get(image_id, "N/A")
        full_path = os.path.join(IMAGES_DIR, filename)
        correct = gt == pred
        records.append(
            {
                "image_id": image_id,
                "filename": filename,
                "path": full_path,
                "gt": gt,
                "pred": pred,
                "correct": correct,
            }
        )

    # Sort by image_id for a stable order
    records.sort(key=lambda r: r["image_id"])

    return records


def main():
    st.title("📊 OpenAI Emotion Classification – Result Viewer")

    records = load_data()
    if not records:
        st.error("No records found. Make sure you've run eval_by_gpt.py first.")
        return

    # Overall stats
    total = len(records)
    evaluated = sum(1 for r in records if r["pred"] != "N/A")
    correct = sum(1 for r in records if r["pred"] != "N/A" and r["gt"] == r["pred"])
    accuracy = correct / evaluated if evaluated > 0 else 0.0

    st.markdown(
        f"**Images:** {total}  |  **Evaluated:** {evaluated}  |  "
        f"**Accuracy:** {accuracy:.2%} ({correct}/{evaluated})"
    )

    # Filter: all / only wrong / only correct
    filter_mode = st.radio(
        "Filter",
        ["All", "Only wrong", "Only correct"],
        horizontal=True,
    )

    if filter_mode == "Only wrong":
        view_records = [r for r in records if r["pred"] != "N/A" and not r["correct"]]
    elif filter_mode == "Only correct":
        view_records = [r for r in records if r["pred"] != "N/A" and r["correct"]]
    else:
        view_records = records

    if not view_records:
        st.info("No records to display for this filter.")
        return

    # Select image by index or filename
    st.subheader("Browse examples")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        idx = st.slider(
            "Index",
            min_value=0,
            max_value=len(view_records) - 1,
            value=0,
            step=1,
        )

        # Optional: select by filename
        filenames = [r["filename"] for r in view_records]
        selected_filename = st.selectbox(
            "Or choose by filename",
            options=filenames,
            index=idx,
        )
        # Sync slider with filename selection
        if selected_filename != view_records[idx]["filename"]:
            idx = filenames.index(selected_filename)

    record = view_records[idx]

    with col_right:
        if os.path.exists(record["path"]):
            image = Image.open(record["path"])
            st.image(image, caption=record["filename"], use_container_width=True)
        else:
            st.error(f"Image file not found: {record['path']}")

    st.markdown("### Labels")
    gt_text = f"**Ground truth:** {record['gt']}"
    pred_text = f"**Prediction:** {record['pred']}"

    if record["pred"] == "N/A":
        st.warning(f"{gt_text}  |  {pred_text} (not evaluated yet)")
    elif record["correct"]:
        st.success(f"{gt_text}  |  {pred_text} ✅")
    else:
        st.error(f"{gt_text}  |  {pred_text} ❌")

    st.markdown(f"**Image ID:** {record['image_id']}  |  **Filename:** `{record['filename']}`")


if __name__ == "__main__":
    main()


