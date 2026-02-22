import base64
import json
import os
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

IMG_SIZE_DEFAULT = 224
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview").strip()
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-latest,gemini-2.5-flash-lite").split(",")
    if model.strip()
]


def resolve_model_path():
    env_model = os.getenv("MODEL_PATH", "").strip()
    if env_model:
        return Path(env_model)

    for candidate in ["sidewalk_classifier_fair.pt", "sidewalk_classifier.pt"]:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    return Path("sidewalk_classifier.pt")


MODEL_PATH = resolve_model_path()


def make_classifier_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

if torch.cuda.is_available():
    CLASSIFIER_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    CLASSIFIER_DEVICE = "mps"
else:
    CLASSIFIER_DEVICE = "cpu"


def load_classifier():
    checkpoint = torch.load(MODEL_PATH, map_location=CLASSIFIER_DEVICE)
    classes = checkpoint["classes"]
    arch = checkpoint.get("arch", "efficientnet_b2")
    img_size = int(checkpoint.get("img_size", IMG_SIZE_DEFAULT))

    if arch == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, len(classes)),
        )
    elif arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, len(classes)),
        )
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {arch}")

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(CLASSIFIER_DEVICE)
    model.eval()
    return model, classes, arch, img_size


def predict_sidewalk_quality(image_bytes: bytes):
    if CLASSIFIER_MODEL is None:
        raise HTTPException(status_code=503, detail=f"Classifier unavailable: {CLASSIFIER_LOAD_ERROR}")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    tensor = CLASSIFIER_TF(image).unsqueeze(0).to(CLASSIFIER_DEVICE)
    with torch.no_grad():
        logits = CLASSIFIER_MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    probabilities = {CLASSIFIER_CLASSES[i]: round(float(prob), 4) for i, prob in enumerate(probs)}
    predicted_idx = int(max(range(len(probs)), key=lambda idx: probs[idx]))
    predicted_class = CLASSIFIER_CLASSES[predicted_idx]
    confidence = round(float(probs[predicted_idx]), 4)

    return predicted_class, confidence, probabilities


def build_gemini_prompt(predicted_class: str, confidence: float, probabilities: dict, custom_prompt: str):
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    prob_text = ", ".join([f"{label}={score * 100:.1f}%" for label, score in sorted_probs])
    confidence_pct = confidence * 100.0

    base_prompt = (
        "You are an ADA sidewalk accessibility reviewer.\n"
        "Use ONLY the uploaded image for visual evidence.\n"
        "Classifier output is authoritative and must not be changed.\n"
        f"Classifier rating: {predicted_class}\n"
        f"Classifier confidence: {confidence_pct:.1f}%\n"
        f"Class probabilities: {prob_text}\n\n"
        "Return ONLY valid JSON (no markdown, no extra text) with this exact schema:\n"
        '{"rating":"Good|Fair|Poor","confidence_pct":0.0,"why":"...","actions":["...","...","..."],"expected_result":"..."}\n\n'
        "Rules:\n"
        "- rating MUST equal the classifier rating exactly.\n"
        "- why must be 2-3 sentences based only on visible evidence.\n"
        "- if confidence < 55.0 then why must start with 'Uncertain:'.\n"
        "- actions must contain exactly 3 specific repair/maintenance actions.\n"
        "- If rating is Good, actions should be maintenance actions to keep it Good.\n"
        "- expected_result is one concise sentence."
    )

    if custom_prompt.strip():
        base_prompt += f"\nAdditional user request: {custom_prompt.strip()}"

    return base_prompt


def extract_json_object(text: str):
    if not text:
        return None
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def call_gemini_text(
    image_bytes: bytes,
    image_mime_type: str,
    prompt: str,
    max_output_tokens: int = 320,
    temperature: float = 0.3,
):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, "Set GEMINI_API_KEY to enable Gemini recommendations.", None

    model_candidates = [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS
    tried_models = []
    last_404_error = ""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": image_mime_type or "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                },
            ]
        }],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
    }

    for model_name in model_candidates:
        if not model_name or model_name in tried_models:
            continue
        tried_models.append(model_name)

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore")
            if exc.code == 404:
                last_404_error = error_text[:220]
                continue
            return None, f"Gemini API error ({exc.code}): {error_text[:220]}", model_name
        except Exception as exc:  # pragma: no cover - network/remote failures
            return None, f"Gemini request failed: {exc}", model_name

        candidates = body.get("candidates") or []
        if not candidates:
            return None, "Gemini returned no response candidates.", model_name

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text_chunks = [part.get("text", "").strip() for part in parts if part.get("text")]
        if not text_chunks:
            return None, "Gemini returned an empty response.", model_name

        return "\n".join(text_chunks), None, model_name

    return (
        None,
        f"No available Gemini model for this key/project. Tried: {', '.join(tried_models)}. Last 404: {last_404_error}",
        None,
    )


def validate_summary_payload(payload: dict, predicted_class: str, confidence: float):
    if not isinstance(payload, dict):
        return None

    rating = str(payload.get("rating", "")).strip()
    if rating not in {"Good", "Fair", "Poor"}:
        return None
    if rating != predicted_class:
        rating = predicted_class

    confidence_pct = payload.get("confidence_pct")
    try:
        confidence_pct = float(confidence_pct)
    except (TypeError, ValueError):
        confidence_pct = confidence * 100.0

    why = str(payload.get("why", "")).strip()
    if not why:
        return None

    actions = payload.get("actions")
    if not isinstance(actions, list):
        return None
    actions = [str(item).strip() for item in actions if str(item).strip()]
    if len(actions) < 3:
        return None
    actions = actions[:3]

    expected_result = str(payload.get("expected_result", "")).strip()
    if not expected_result:
        return None

    return {
        "rating": rating,
        "confidence_pct": confidence_pct,
        "why": why,
        "actions": actions,
        "expected_result": expected_result,
    }


def format_summary_payload(payload: dict):
    confidence_pct = payload.get("confidence_pct")
    if isinstance(confidence_pct, (int, float)):
        confidence_text = f"{float(confidence_pct):.1f}%"
    else:
        confidence_text = "n/a"

    return "\n".join(
        [
            f"Rating: {payload['rating']} ({confidence_text})",
            f"Why this rating: {payload['why']}",
            "How to improve to Good:",
            f"1. {payload['actions'][0]}",
            f"2. {payload['actions'][1]}",
            f"3. {payload['actions'][2]}",
            f"Expected result: {payload['expected_result']}",
        ]
    )


def fallback_summary_text(predicted_class: str, confidence: float):
    confidence_pct = confidence * 100.0
    uncertain_prefix = "Uncertain: " if confidence_pct < 55.0 else ""

    if predicted_class == "Good":
        why = (
            f"{uncertain_prefix}The surface appears mostly continuous and walkable with limited visible obstructions. "
            "No major trip hazards are clearly dominant in this frame."
        )
        actions = [
            "Schedule routine cleaning and debris removal along the walking path.",
            "Trim vegetation and maintain edge clearances to preserve sidewalk width.",
            "Perform periodic spot sealing/patching where early wear appears.",
        ]
        expected = "Sidewalk quality is maintained at Good with reduced risk of future deterioration."
    elif predicted_class == "Fair":
        why = (
            f"{uncertain_prefix}The sidewalk appears usable but shows moderate surface wear and localized defects that can worsen over time. "
            "These issues can impact comfort and accessibility if not corrected."
        )
        actions = [
            "Patch visible cracks and minor depressions in the main walking line.",
            "Grind or level small vertical offsets to reduce trip risk.",
            "Repaint/restore edge and crossing cues where faded or unclear.",
        ]
        expected = "Defects are reduced and accessibility improves toward a stable Good condition."
    else:
        why = (
            f"{uncertain_prefix}The image indicates significant deterioration and/or hazards affecting safe pedestrian movement. "
            "Current conditions likely present frequent accessibility barriers."
        )
        actions = [
            "Repair or replace severely broken pavement sections in the travel path.",
            "Correct major height differentials and unstable edges near curb transitions.",
            "Reconstruct affected segments to restore a smooth, continuous ADA-friendly surface.",
        ]
        expected = "Major hazards are removed and the corridor can be restored to Good usability."

    return "\n".join(
        [
            f"Rating: {predicted_class} ({confidence_pct:.1f}%)",
            f"Why this rating: {why}",
            "How to improve to Good:",
            f"1. {actions[0]}",
            f"2. {actions[1]}",
            f"3. {actions[2]}",
            f"Expected result: {expected}",
        ]
    )


def build_sidewalk_presence_prompt():
    return (
        "Determine whether this image contains a real, visible pedestrian sidewalk or footpath.\n"
        "Treat logos, icons, screenshots, drawings, diagrams, blank images, and synthetic graphics as NO sidewalk.\n"
        "Return ONLY JSON with this schema:\n"
        '{"has_sidewalk": true|false, "confidence": 0.0-1.0, "reason": "short reason"}'
    )


def detect_sidewalk_presence(image_bytes: bytes, image_mime_type: str):
    prompt = build_sidewalk_presence_prompt()
    text, error, model = call_gemini_text(
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        prompt=prompt,
        max_output_tokens=120,
        temperature=0.0,
    )
    if error:
        return None, error, model

    parsed = extract_json_object(text or "")
    if not parsed or "has_sidewalk" not in parsed:
        return None, f"Could not parse sidewalk-detection output: {text}", model

    has_sidewalk = bool(parsed.get("has_sidewalk"))
    confidence = parsed.get("confidence")
    reason = str(parsed.get("reason", "")).strip()

    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return {
        "has_sidewalk": has_sidewalk,
        "confidence": confidence,
        "reason": reason,
    }, None, model


def call_gemini_summary(
    image_bytes: bytes,
    image_mime_type: str,
    prompt: str,
    predicted_class: str,
    confidence: float,
):
    text, error, model = call_gemini_text(
        image_bytes,
        image_mime_type,
        prompt,
        max_output_tokens=520,
        temperature=0.2,
    )
    if error:
        fallback = fallback_summary_text(predicted_class, confidence)
        return fallback, f"{error} | Used fallback summary template.", model

    parsed = extract_json_object(text or "")
    validated = validate_summary_payload(parsed, predicted_class, confidence)
    if validated is not None:
        return format_summary_payload(validated), None, model

    retry_prompt = (
        prompt
        + "\n\nCRITICAL: Your previous response was invalid. Return ONLY one valid JSON object matching the schema."
    )
    retry_text, retry_error, retry_model = call_gemini_text(
        image_bytes,
        image_mime_type,
        retry_prompt,
        max_output_tokens=520,
        temperature=0.1,
    )
    final_model = retry_model or model
    if retry_error:
        fallback = fallback_summary_text(predicted_class, confidence)
        return fallback, f"{retry_error} | Used fallback summary template.", final_model

    retry_parsed = extract_json_object(retry_text or "")
    retry_validated = validate_summary_payload(retry_parsed, predicted_class, confidence)
    if retry_validated is not None:
        return format_summary_payload(retry_validated), None, final_model

    fallback = fallback_summary_text(predicted_class, confidence)
    return fallback, "Gemini response was incomplete; used fallback summary template.", final_model


try:
    CLASSIFIER_MODEL, CLASSIFIER_CLASSES, CLASSIFIER_ARCH, CLASSIFIER_IMG_SIZE = load_classifier()
    CLASSIFIER_TF = make_classifier_transform(CLASSIFIER_IMG_SIZE)
    CLASSIFIER_LOAD_ERROR = ""
    print(
        f"Loaded model: {MODEL_PATH} | Arch: {CLASSIFIER_ARCH} | "
        f"Classes: {CLASSIFIER_CLASSES} | ImgSize: {CLASSIFIER_IMG_SIZE} | Device: {CLASSIFIER_DEVICE}"
    )
except Exception as exc:  # pragma: no cover - startup environment issue
    CLASSIFIER_MODEL = None
    CLASSIFIER_CLASSES = []
    CLASSIFIER_ARCH = ""
    CLASSIFIER_IMG_SIZE = IMG_SIZE_DEFAULT
    CLASSIFIER_TF = make_classifier_transform(CLASSIFIER_IMG_SIZE)
    CLASSIFIER_LOAD_ERROR = str(exc)
    print(f"Classifier unavailable: {CLASSIFIER_LOAD_ERROR}")

print("Loading Brookline data...")
assets = gpd.read_file("aboveGroundAssets.geojson")
assets_utm = assets.to_crs("EPSG:32619")
print(f"Loaded {len(assets)} assets")

OBSTACLE_TYPES = ["BIKE_RACK", "TRASH_BIN", "UTILITY_POLE", "PLANTER", "CABINET", "BOX", "HYDRANT", "SIGNAL_POLE"]
INACCESSIBLE_MATERIALS = ["Gravel"]
POOR_MATERIALS = ["Brick"]

def find_obstacles(sidewalk_utm, obstacles_utm):
    nearby = obstacles_utm[obstacles_utm.geometry.distance(sidewalk_utm.geometry) < 2.0]
    result = []
    for _, obs in nearby.iterrows():
        result.append({
            "type": str(obs.get('asset_type', 'UNKNOWN')),
            "condition": str(obs.get('condition', 'Unknown')),
            "distance_m": round(float(obs.geometry.distance(sidewalk_utm.geometry)), 2)
        })
    return result

def analyze_sidewalks():
    sidewalks_utm = assets_utm[assets_utm['asset_type'] == 'SIDEWALK'].copy()
    obstacles_utm = assets_utm[assets_utm['asset_type'].isin(OBSTACLE_TYPES)]
    sidewalks_orig = assets[assets['asset_type'] == 'SIDEWALK']

    results = []
    print(f"Analyzing {len(sidewalks_utm)} sidewalks...")

    for idx, row in sidewalks_utm.iterrows():
        orig = sidewalks_orig.loc[idx]
        sidewalk_type = str(row.get('Type', 'Sidewalk'))
        condition = str(row.get('condition', 'Unknown'))
        material = str(row.get('Material', 'Unknown'))
        image_url = str(orig.get('image_url', ''))

        # Real violations only - no fake measurements
        violations = []
        severity = "compliant"

        # 1. Missing sidewalk entirely
        if sidewalk_type == 'No Sidewalk':
            violations.append("No sidewalk present — pedestrians forced onto road")
            severity = "critical"

        # 2. Poor condition
        elif condition == 'Poor':
            violations.append("Poor condition — surface hazard for wheelchair users")
            severity = "high"

        # 3. Inaccessible material
        if material in INACCESSIBLE_MATERIALS:
            violations.append(f"{material} surface — not ADA compliant")
            severity = "high"

        # 4. Difficult material
        elif material in POOR_MATERIALS and condition != 'Good':
            violations.append(f"{material} surface in {condition} condition — accessibility risk")
            if severity == "compliant":
                severity = "medium"

        # 5. Real obstacles from data
        obstacles = find_obstacles(row, obstacles_utm)
        if obstacles:
            violations.append(f"{len(obstacles)} obstacle(s) within 2m of path")
            if severity == "compliant":
                severity = "medium"

        # 6. Under construction
        if condition == 'Under Construction':
            violations.append("Under construction — temporarily inaccessible")
            if severity == "compliant":
                severity = "medium"

        ada_compliant = len(violations) == 0

        results.append({
            "feature_id": str(row.get('feature_id', idx)),
            "geometry": orig.geometry.__geo_interface__,
            "sidewalk_type": sidewalk_type,
            "condition": condition,
            "material": material,
            "image_url": image_url,
            "ada_compliant": ada_compliant,
            "severity": severity,
            "violations": violations,
            "obstacles": obstacles,
            "obstacle_count": len(obstacles)
        })

    print("Analysis complete!")
    return results

print("Running sidewalk analysis...")
SIDEWALK_RESULTS = analyze_sidewalks()

compliant = sum(1 for s in SIDEWALK_RESULTS if s['ada_compliant'])
critical = sum(1 for s in SIDEWALK_RESULTS if s['severity'] == 'critical')
high = sum(1 for s in SIDEWALK_RESULTS if s['severity'] == 'high')
medium = sum(1 for s in SIDEWALK_RESULTS if s['severity'] == 'medium')
missing = sum(1 for s in SIDEWALK_RESULTS if s['sidewalk_type'] == 'No Sidewalk')
poor_condition = sum(1 for s in SIDEWALK_RESULTS if s['condition'] == 'Poor')
obstructed = sum(1 for s in SIDEWALK_RESULTS if s['obstacle_count'] > 0)

SUMMARY = {
    "total": len(SIDEWALK_RESULTS),
    "compliant": compliant,
    "non_compliant": len(SIDEWALK_RESULTS) - compliant,
    "critical": critical,
    "high": high,
    "medium": medium,
    "missing_sidewalk": missing,
    "poor_condition": poor_condition,
    "obstructed": obstructed,
    "compliance_rate": round(compliant / len(SIDEWALK_RESULTS) * 100, 1)
}
print(f"Summary: {SUMMARY}")

@app.get("/sidewalks")
def get_sidewalks():
    return {"sidewalks": SIDEWALK_RESULTS, "summary": SUMMARY}

@app.get("/summary")
def get_summary():
    return SUMMARY

@app.get("/violations")
def get_violations():
    violations = [s for s in SIDEWALK_RESULTS if not s['ada_compliant']]
    return {"violations": violations, "count": len(violations)}


@app.post("/predict-sidewalk")
async def predict_sidewalk(
    image: UploadFile = File(...),
    include_gemini: bool = Form(True),
    guidance_prompt: str = Form(""),
    enforce_sidewalk_check: bool = Form(True),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    gemini_summary = None
    gemini_error = None
    prompt_used = ""
    gemini_model_used = None
    sidewalk_check = {
        "checked": False,
        "has_sidewalk": None,
        "confidence": None,
        "reason": "",
        "error": None,
        "model": None,
    }
    classification_skipped = False

    if enforce_sidewalk_check:
        sidewalk_check["checked"] = True
        check_result, check_error, check_model = detect_sidewalk_presence(image_bytes, image.content_type)
        sidewalk_check["model"] = check_model

        if check_result is None:
            sidewalk_check["error"] = check_error
        else:
            sidewalk_check["has_sidewalk"] = check_result["has_sidewalk"]
            sidewalk_check["confidence"] = check_result["confidence"]
            sidewalk_check["reason"] = check_result["reason"]

            if not check_result["has_sidewalk"]:
                classification_skipped = True
                gemini_model_used = check_model
                gemini_summary = (
                    "No sidewalk detected in this image, so Good/Fair/Poor classification was skipped. "
                    "Upload a real street-side sidewalk photo to get condition scoring and improvement guidance."
                )
                return {
                    "predicted_class": None,
                    "confidence": None,
                    "probabilities": {},
                    "classification_skipped": classification_skipped,
                    "sidewalk_check": sidewalk_check,
                    "gemini_summary": gemini_summary,
                    "gemini_error": gemini_error,
                    "gemini_model": gemini_model_used,
                    "gemini_prompt": prompt_used,
                }

    predicted_class, confidence, probabilities = predict_sidewalk_quality(image_bytes)

    if include_gemini:
        prompt_used = build_gemini_prompt(predicted_class, confidence, probabilities, guidance_prompt)
        gemini_summary, gemini_error, gemini_model_used = call_gemini_summary(
            image_bytes,
            image.content_type,
            prompt_used,
            predicted_class,
            confidence,
        )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "classification_skipped": classification_skipped,
        "sidewalk_check": sidewalk_check,
        "gemini_summary": gemini_summary,
        "gemini_error": gemini_error,
        "gemini_model": gemini_model_used,
        "gemini_prompt": prompt_used,
    }
