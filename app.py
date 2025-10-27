import os
import time
import json
import threading
import numpy as np
import cv2
from scipy.signal import find_peaks, butter, filtfilt
from flask import Flask, render_template, jsonify, request
import base64
import re
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------- Load environment variables ----------------
load_dotenv()

# Use GOOGLE_API_KEY instead of GEMINI_API_KEY (important fix)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")
CAM_INDEX = int(os.getenv("CAM_INDEX", 0))
CAPTURE_SECONDS = int(os.getenv("CAPTURE_SECONDS", 8))

# ---------------- Configure Gemini ----------------
genai.configure(api_key=GEMINI_API_KEY)

# ---------------- Initialize Flask ----------------
app = Flask(__name__)
vitals_data = {"status": "idle"}

# ---------------- Helper functions ----------------
def bandpass_filter(signal, fs, low=0.5, high=5.0, order=3):
    # Protect against invalid filter design when sampling rate is very low.
    # Normalized critical frequencies must satisfy 0 < Wn < 1.
    if fs is None or fs <= 0:
        return signal
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    # If normalized frequencies are not in (0,1) or low>=high, skip filtering.
    if not (0 < low_norm < high_norm < 1):
        return signal
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


def compute_ppg_features(ppg_signal, red_ch, green_ch, fps):
    sig = green_ch - np.mean(green_ch) if len(green_ch) > 0 else np.array([])
    sig = bandpass_filter(sig, fps)
    # ensure distance for peak finding is at least 1
    try:
        distance = int(0.3 * fps)
    except Exception:
        distance = 1
    distance = max(1, distance)
    peaks, _ = find_peaks(sig, distance=distance) if len(sig) > 0 else (np.array([]), {})
    ibi = (np.diff(peaks) / fps) if len(peaks) > 1 else np.array([])
    hr = 60 / np.median(ibi) if len(ibi) > 0 else 0
    hrv = np.std(ibi) * 1000 if len(ibi) > 1 else 0
    ppg_amp = float(np.max(sig) - np.min(sig))
    red_mean = np.mean(red_ch)
    green_mean = np.mean(green_ch)
    ac_red = np.std(red_ch)
    ac_green = np.std(green_ch)
    dc_red = max(1e-6, red_mean)
    dc_green = max(1e-6, green_mean)
    r_ratio = (ac_red / dc_red) / (ac_green / dc_green)
    spo2_est = max(60, min(100, 110 - 25 * r_ratio))
    breathing_rate = (60 / np.mean(ibi) * 0.33) if len(ibi) > 0 and np.mean(ibi) != 0 else 16
    features = {
        "Heart Rate (BPM)": round(hr, 1),
        "HRV (ms)": round(hrv, 1),
        "SpO2 (%)": round(spo2_est, 1),
        "PPG Amplitude": round(ppg_amp, 1),
        "Breathing Rate (BPM)": round(breathing_rate, 1),
        "Signal Length": len(sig),
        "FPS": fps,
    }
    return features, sig, peaks


def suggest_doctor(category):
    mapping = {
        "cardio": "Cardiologist (Heart & ECG evaluation)",
        "respiratory": "Pulmonologist (Lungs & Oxygen monitoring)",
        "anemia": "Hematologist (Blood tests & review)",
        "stress": "Psychologist / Primary Care for stress",
        "skin": "Dermatologist (Skin & circulation review)",
        "general": "Primary Care Physician",
    }
    return mapping.get(category.lower(), "Primary Care Physician")


def capture_ppg():
    global vitals_data
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        vitals_data = {"status": "error", "message": "Cannot open camera"}
        return

    fps = 30.0
    ppg_signal, red_ch, green_ch = [], [], []
    start = time.time()

    while time.time() - start < CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        b, g, r = cv2.split(frame)
        red_ch.append(np.mean(r))
        green_ch.append(np.mean(g))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ppg_signal.append(np.mean(gray))
        cv2.imshow("PPG Capture Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    features, sig, peaks = compute_ppg_features(
        np.array(ppg_signal), np.array(red_ch), np.array(green_ch), fps
    )

    # ---------------- Gemini AI Section ----------------
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            "You are a professional health assistant. Format output in Markdown.\n"
            "Include: short summary, explanation of vitals, and suggested doctor category.\n"
            f"Vitals data: {json.dumps(features)}"
        )
        resp = model.generate_content(prompt)
        ai_text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        msg = str(e)
        m = re.search(r"Please retry in\s*([0-9]+(?:\.[0-9]+)?)s", msg)
        if m:
            wait = float(m.group(1))
            ai_text = (
                "AI assessment unavailable: quota exceeded. "
                f"Please retry after ~{int(wait)}s."
            )
        elif "quota" in msg.lower() or "429" in msg:
            ai_text = "AI assessment unavailable: quota or rate limit exceeded. Try again later or enable billing."
        else:
            ai_text = f"AI assessment failed: {msg}"

    vitals_data = {
        "status": "done",
        "features": features,
        "ai_text": ai_text,
        "waveform": sig.tolist(),
        "peaks": peaks.tolist(),
        "suggested_doctor": suggest_doctor("general"),
    }


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """Accepts JSON {frames: [dataURL,...]} from client, decodes frames,
    computes simple PPG features and returns a JSON response similar to
    the server-side capture flow.
    """
    payload = request.get_json(silent=True)
    if not payload or "frames" not in payload:
        return jsonify({"error": "No frames provided"}), 400

    frames = payload.get("frames", [])
    if len(frames) == 0:
        return jsonify({"error": "Empty frames list"}), 400

    red_ch, green_ch, ppg_signal = [], [], []
    for dataurl in frames:
        try:
            # strip header like 'data:image/jpeg;base64,'
            if "," in dataurl:
                _, b64 = dataurl.split(",", 1)
            else:
                b64 = dataurl
            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            b, g, r = cv2.split(img)
            red_ch.append(np.mean(r))
            green_ch.append(np.mean(g))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ppg_signal.append(np.mean(gray))
        except Exception:
            continue

    if len(ppg_signal) == 0:
        return jsonify({"error": "Unable to decode frames"}), 400

    fps = max(1.0, len(ppg_signal) / float(CAPTURE_SECONDS))
    features, sig, peaks = compute_ppg_features(
        np.array(ppg_signal), np.array(red_ch), np.array(green_ch), fps
    )

    # Try to generate AI assessment, but don't fail on errors
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            "You are a professional health assistant. Format output in Markdown.\n"
            "Include: short summary, explanation of vitals, and suggested doctor category.\n"
            f"Vitals data: {json.dumps(features)}"
        )
        resp = model.generate_content(prompt)
        ai_text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        msg = str(e)
        m = re.search(r"Please retry in\s*([0-9]+(?:\.[0-9]+)?)s", msg)
        if m:
            wait = float(m.group(1))
            ai_text = (
                "AI assessment unavailable: quota exceeded. "
                f"Please retry after ~{int(wait)}s."
            )
        elif "quota" in msg.lower() or "429" in msg:
            ai_text = "AI assessment unavailable: quota or rate limit exceeded. Try again later or enable billing."
        else:
            ai_text = f"AI assessment unavailable: {msg}"

    resp = {
        "status": "done",
        "features": features,
        "ai_text": ai_text,
        "waveform": sig.tolist(),
        "peaks": peaks.tolist(),
    }
    return jsonify(resp)

# ---------------- Flask routes ----------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/scan")
def scan():
    vitals_data["status"] = "capturing"
    threading.Thread(target=capture_ppg).start()
    return jsonify({"status": "started"})


@app.route("/results")
def results():
    return jsonify(vitals_data)


# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(debug=True)
