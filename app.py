import os, io, csv, sqlite3
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, make_response
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-development-secret")
DB_PATH = os.path.join(os.path.dirname(__file__), "diabetes_app.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.joblib")

FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age"
]
MODEL = None

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            glucose REAL, bmi REAL, age INTEGER,
            risk_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")

def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
    else:
        from train_model import train_and_save
        MODEL = train_and_save(MODEL_PATH)

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not username or not email or len(password) < 8:
            flash("Enter all fields. Password must contain at least 8 characters.", "error")
            return render_template("register.html")
        try:
            with get_db() as conn:
                conn.execute("INSERT INTO users(username,email,password_hash,created_at) VALUES(?,?,?,?)",
                    (username, email, generate_password_hash(password), datetime.now().isoformat(timespec="seconds")))
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("That username or email is already registered.", "error")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identity = request.form.get("identity", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE lower(email)=? OR lower(username)=?",
                                (identity, identity)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid username/email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        history = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 5",
                               (session["user_id"],)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?",
                             (session["user_id"],)).fetchone()[0]
    return render_template("dashboard.html", history=history, total=total)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    if request.method == "POST":
        try:
            values = {
                "pregnancies": float(request.form["pregnancies"]),
                "glucose": float(request.form["glucose"]),
                "blood_pressure": float(request.form["blood_pressure"]),
                "skin_thickness": float(request.form["skin_thickness"]),
                "insulin": float(request.form["insulin"]),
                "bmi": float(request.form["bmi"]),
                "diabetes_pedigree": float(request.form["diabetes_pedigree"]),
                "age": float(request.form["age"])
            }
            if any(v < 0 for v in values.values()) or values["age"] < 1 or values["age"] > 120:
                raise ValueError("Values must be non-negative and age must be between 1 and 120.")
            X = np.array([[values[f] for f in FEATURES]], dtype=float)
            probability = float(MODEL.predict_proba(X)[0][1]) * 100
            level = "High" if probability >= 65 else ("Moderate" if probability >= 35 else "Low")
            with get_db() as conn:
                cur = conn.execute("""INSERT INTO predictions
                    (user_id,created_at,glucose,bmi,age,risk_probability,risk_level)
                    VALUES(?,?,?,?,?,?,?)""",
                    (session["user_id"], datetime.now().isoformat(timespec="seconds"),
                     values["glucose"], values["bmi"], int(values["age"]), probability, level))
                prediction_id = cur.lastrowid
            result = {"id": prediction_id, "probability": round(probability, 1), "level": level}
        except (ValueError, KeyError):
            flash("Please enter valid numeric values in every field.", "error")
    return render_template("predict.html", result=result)

@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC",
                            (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows)

@app.route("/report/<int:prediction_id>")
@login_required
def report(prediction_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                           (prediction_id, session["user_id"])).fetchone()
    if not row:
        flash("Prediction not found.", "error")
        return redirect(url_for("history"))
    content = f"""DIABETES RISK ESTIMATION REPORT
================================
User: {session.get('username')}
Prediction ID: {row['id']}
Date: {row['created_at']}
Glucose input: {row['glucose']}
BMI input: {row['bmi']}
Age input: {row['age']}
Estimated model probability: {row['risk_probability']:.1f}%
Model category: {row['risk_level']}

IMPORTANT:
This is an educational demonstration using a model trained on synthetic data.
It is NOT a diagnosis, medical advice, or a clinically validated risk score.
A low score does not rule out diabetes. Consult a qualified healthcare
professional for screening, diagnosis, and treatment decisions.
"""
    return send_file(io.BytesIO(content.encode("utf-8")), mimetype="text/plain",
                     as_attachment=True, download_name=f"diabetes_report_{row['id']}.txt")

@app.route("/tips")
@login_required
def tips():
    return render_template("tips.html")

@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404

if __name__ == "__main__":
    init_db()
    load_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
