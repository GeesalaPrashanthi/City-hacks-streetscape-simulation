"""
Step 1: Run Mask2Former on downloaded images to crop sidewalk regions only.
Output: dataset_masked/Good, Fair, Poor with cropped sidewalk images.
"""
import torch
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from PIL import Image
import numpy as np
from pathlib import Path

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
print(f"Loading {MODEL_ID}...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)
model = model.to(DEVICE)
model.eval()
print("Model loaded!")

# Mapillary Vistas sidewalk label IDs
# Label 2 = sidewalk in Mapillary Vistas
SIDEWALK_LABELS = {2, 3}  # sidewalk + curb

INPUT_BASE = Path("dataset")
OUTPUT_BASE = Path("dataset_masked")

def mask_and_crop(image_path: Path, output_path: Path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
    )[0]

    seg_map = pred.cpu().numpy()
    sidewalk_mask = np.isin(seg_map, list(SIDEWALK_LABELS))

    if sidewalk_mask.sum() < 500:
        # Not enough sidewalk detected - save original resized
        img_resized = img.resize((224, 224))
        img_resized.save(output_path)
        return False

    # Crop to bounding box of sidewalk region
    rows = np.any(sidewalk_mask, axis=1)
    cols = np.any(sidewalk_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding
    pad = 20
    rmin = max(0, rmin - pad)
    rmax = min(img.height, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img.width, cmax + pad)

    cropped = img.crop((cmin, rmin, cmax, rmax))

    # Apply mask overlay - darken non-sidewalk regions
    img_array = np.array(cropped)
    crop_mask = sidewalk_mask[rmin:rmax, cmin:cmax]
    img_array[~crop_mask] = (img_array[~crop_mask] * 0.3).astype(np.uint8)

    result = Image.fromarray(img_array).resize((224, 224))
    result.save(output_path)
    return True

total = 0
masked = 0
failed = 0

for condition in ["Good", "Fair", "Poor"]:
    input_dir = INPUT_BASE / condition
    output_dir = OUTPUT_BASE / condition
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.jpg"))
    print(f"\nProcessing {len(images)} {condition} images...")

    for i, img_path in enumerate(images):
        output_path = output_dir / img_path.name
        if output_path.exists():
            total += 1
            masked += 1
            continue
        try:
            success = mask_and_crop(img_path, output_path)
            total += 1
            if success:
                masked += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(images)} done ({masked} masked, {failed} failed)")
        except Exception as e:
            failed += 1
            print(f"  ✗ {img_path.name}: {e}")

print(f"\nDone! Total: {total}, Masked: {masked}, Failed: {failed}")
print("Output in dataset_masked/")
