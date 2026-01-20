from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import os
import wave
import json
import uuid
import requests
from dotenv import load_dotenv

from vosk import Model, KaldiRecognizer

import firebase_admin
from firebase_admin import credentials, firestore

from transformers import pipeline


# -----------------------------
# Load env
# -----------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
ALLOWED_EXTENSIONS = {"wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
FIREBASE_SERVICE_ACCOUNT = os.getenv("FIREBASE_SERVICE_ACCOUNT")

# -----------------------------
# Firebase init (Firestore)
# -----------------------------
if not FIREBASE_SERVICE_ACCOUNT or not os.path.exists(FIREBASE_SERVICE_ACCOUNT):
    raise RuntimeError("Firebase service account JSON missing. Set FIREBASE_SERVICE_ACCOUNT in .env")

cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT)
firebase_admin.initialize_app(cred)
db = firestore.client()


# -----------------------------
# Vosk model init
# -----------------------------
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15"))

if not os.path.exists(VOSK_MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at: {VOSK_MODEL_PATH}")

vosk_model = Model(VOSK_MODEL_PATH)
print("✅ Vosk model loaded")


# -----------------------------
# Offline Summarizer init (NO API)
# -----------------------------
# This downloads model first time (internet needed once)
# Afterwards it runs locally.
print("⏳ Loading summarizer model locally...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("✅ Summarizer loaded")


# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def firebase_signup(email, password):
    """Create user in Firebase Auth using REST API."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload)
    return r.json(), r.status_code


def firebase_login(email, password):
    """Login user using Firebase Auth REST API."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload)
    return r.json(), r.status_code


def transcribe_audio(file_path):
    """Transcribe WAV mono PCM using Vosk."""
    audio = wave.open(file_path, "rb")

    if (audio.getnchannels() != 1 or audio.getsampwidth() != 2 or not (8000 <= audio.getframerate() <= 48000)):
        audio.close()
        return None, "Audio must be WAV mono PCM (16-bit)."

    rec = KaldiRecognizer(vosk_model, audio.getframerate())
    text = ""

    while True:
        data = audio.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part = json.loads(rec.Result()).get("text", "")
            text += part + " "

    final = json.loads(rec.FinalResult()).get("text", "")
    text += final
    audio.close()

    return text.strip(), None


def summarize_text_offline(text):
    if not text or not text.strip():
        return None, "Empty transcript, cannot summarize."

    # Make summary contextual
    prompt = f"""
You are an expert meeting summarizer.

Create a contextual summary with:
1) Main topic
2) Key points (bullets)
3) Action items (if any)
4) Important entities (names, dates, places)

Transcript:
{text}
"""

    max_chars = 1200
    chunks = [prompt[i:i + max_chars] for i in range(0, len(prompt), max_chars)]

    summaries = []
    for chunk in chunks:
        out = summarizer(chunk, max_length=160, min_length=60, do_sample=False)
        summaries.append(out[0]["summary_text"])

    return " ".join(summaries).strip(), None



def require_login():
    return "user" in session


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not FIREBASE_API_KEY:
            flash("Firebase API Key missing in .env")
            return render_template("signup.html")

        data, status = firebase_signup(email, password)

        if status == 200:
            session["user"] = {
                "email": data["email"],
                "uid": data["localId"],
                "idToken": data["idToken"]
            }
            flash("Signup successful ✅")
            return redirect(url_for("dashboard"))

        msg = data.get("error", {}).get("message", "Signup failed")
        flash(f"Signup failed: {msg}")
        return render_template("signup.html")

    # ✅ IMPORTANT: This returns the page when user opens /signup in browser
    return render_template("signup.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        data, status = firebase_login(email, password)

        if status == 200:
            session["user"] = {
                "email": data["email"],
                "uid": data["localId"],
                "idToken": data["idToken"]
            }
            flash("Login successful ✅")
            return redirect(url_for("dashboard"))

        msg = data.get("error", {}).get("message", "Login failed")
        flash(f"Login failed: {msg}")
        return render_template("login.html")

    return render_template("login.html")



@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.")
    return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("dashboard.html")


@app.route("/upload", methods=["POST"])
def upload():
    if not require_login():
        return redirect(url_for("login"))

    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .wav files allowed"}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # safer random filename
    unique_name = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(unique_name))
    file.save(filepath)

    # Transcription
    transcript, err = transcribe_audio(filepath)
    if err:
        return jsonify({"error": err}), 400

    # Summarization (offline)
    summary, err2 = summarize_text_offline(transcript)
    if err2:
        return jsonify({"error": err2}), 400

    # Save to Firestore
    user = session["user"]
    doc = {
        "uid": user["uid"],
        "email": user["email"],
        "filename": unique_name,
        "transcription": transcript,
        "summary": summary,
        "created_at": firestore.SERVER_TIMESTAMP
    }
    db.collection("records").add(doc)

    return jsonify({
        "message": "Processed successfully",
        "transcription": transcript,
        "summary": summary
    })


@app.route("/history")
def history():
    if "user" not in session:
        return jsonify([])

    uid = session["user"]["uid"]

    docs = (
        db.collection("records")
        .where("uid", "==", uid)
        .limit(10)
        .stream()
    )

    results = []
    for d in docs:
        x = d.to_dict()
        x["id"] = d.id
        x.pop("transcription", None)
        results.append(x)

    return jsonify(results)



# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
