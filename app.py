import os

# ----------------- Early env setup (MUST be before importing transformers/ultralytics) -----------------
BASE_DIR = os.path.dirname(__file__)

# Make caches writable on Render (ephemeral FS) and reduce thread-related memory spikes
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("HF_ASSETS_CACHE", "/tmp/hf/assets")
# Do NOT rely on TRANSFORMERS_CACHE (deprecated). If you keep it, keep it only for older versions:
# os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf")

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# Reduce CPU thread fan-out that can spike memory
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

print("RUNNING:", os.path.abspath(__file__))
print("STATIC DIR SHOULD BE:", os.path.join(BASE_DIR, "static"))

# ----------------- Imports -----------------
import uuid
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import cv2
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# ----------------- Flask setup -----------------
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB
ALLOWED = {"png", "jpg", "jpeg", "webp"}

# Timeouts (seconds)
DETECTION_TIMEOUT = 120
CAPTION_TIMEOUT = 120

# IMPORTANT: Keep concurrency low on small instances to avoid OOM.
executor = ThreadPoolExecutor(max_workers=1)

# ----------------- Lazy-loaded models with locks (avoid race double-load) -----------------
yolo = None
captioner = None

_yolo_lock = threading.Lock()
_cap_lock = threading.Lock()


def _get_yolo():
    global yolo
    if yolo is None:
        with _yolo_lock:
            if yolo is None:
                # NOTE: ensure yolov8n.pt exists (best: download at build time)
                yolo = YOLO("yolov8n.pt")
    return yolo


def _get_captioner():
    global captioner
    if captioner is None:
        with _cap_lock:
            if captioner is None:
                # This is heavy. If Render RAM is small, switch to a smaller model.
                captioner = pipeline(
                    "image-to-text",
                    model="Salesforce/blip-image-captioning-large",
                )
    return captioner


# ----------------- Helpers -----------------
def allowed_file(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


def with_timeout(fn, *args, timeout: int):
    fut = executor.submit(fn, *args)
    return fut.result(timeout=timeout)


# ----------------- Core functions -----------------
def detect_objects(image_path: str, conf_thres=0.20, img_size=960):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    model = _get_yolo()
    res = model.predict(img, conf=conf_thres, imgsz=img_size, verbose=False)[0]
    names = res.names

    objects = []
    for b in res.boxes:
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        label = names[cls_id]
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        objects.append({"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]})

    return img, objects


def summarize_objects(objects):
    counts = Counter([o["label"] for o in objects]) if objects else Counter()
    avg_conf = (sum(o["confidence"] for o in objects) / len(objects)) if objects else 0.0
    return {"counts": dict(counts), "avg_conf": float(avg_conf)}


def caption_image(img_bgr) -> str:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    cap = _get_captioner()
    out = cap(pil, max_new_tokens=60)
    if not out or "generated_text" not in out[0]:
        return "A photo with several visible elements."
    return out[0]["generated_text"].strip()


def build_paragraph(caption: str, counts_dict: dict) -> str:
    counts = dict(counts_dict or {})

    people = counts.pop("person", 0)
    obj_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    parts = [caption.rstrip(".") + "."]

    if people:
        parts.append(f"Detected {people} person instance(s).")

    if obj_items:
        objs = ", ".join([f"{k} x{v}" for k, v in obj_items])
        parts.append(f"Detected objects include: {objs}.")

    return " ".join(parts)


# ----------------- Routes -----------------
@app.get("/health")
def health():
    return "ok", 200


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/chat", methods=["POST"])
def chat_api():
    t0 = time.time()

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded."}), 400

    file = request.files["image"]
    if not file.filename or file.filename.strip() == "":
        return jsonify({"ok": False, "error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Invalid file type. Use PNG/JPG/JPEG/WEBP."}), 400

    safe_name = secure_filename(file.filename)
    saved_name = f"{uuid.uuid4().hex}_{safe_name}"
    path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(path)
    image_url = f"/uploads/{saved_name}"

    # YOLO detections
    try:
        img_bgr, objects = with_timeout(detect_objects, path, timeout=DETECTION_TIMEOUT)
    except FuturesTimeoutError:
        return jsonify({"ok": False, "error": "Detection timed out. Try a smaller image."}), 504
    except Exception as e:
        return jsonify({"ok": False, "error": f"Detection failed: {type(e).__name__}"}), 500

    summary = summarize_objects(objects)

    # BLIP caption
    try:
        cap = with_timeout(caption_image, img_bgr, timeout=CAPTION_TIMEOUT)
    except FuturesTimeoutError:
        cap = "A photo with several visible elements."
    except Exception:
        cap = "A photo with several visible elements."

    overall = build_paragraph(cap, summary["counts"])

    elapsed_ms = int((time.time() - t0) * 1000)
    return jsonify(
        {
            "ok": True,
            "image_url": image_url,
            "overall_text": overall,
            "elapsed_ms": elapsed_ms,
            "detections": summary["counts"],
        }
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False, "error": "File too large. Max is 8MB."}), 413


if __name__ == "__main__":
    # Local dev only. On Render you'll run via gunicorn.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
