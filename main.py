import os
import uuid
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import bcrypt
from process_video import process_video_for_speed

UPLOAD_DIR = "uploads"
HISTORY_DIR = "history"
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# JWT Configuration
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "fallback-secret-key-12345")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)

# Simple in-memory user storage (replace with database in production)
USERS = {
    "admin": {
        "password": bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt()),
        "role": "admin"
    },
    "user": {
        "password": bcrypt.hashpw("user123".encode('utf-8'), bcrypt.gensalt()),
        "role": "user"
    }
}

# Store processing history
HISTORY_FILE = os.path.join(HISTORY_DIR, "processing_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history_data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_data, f, indent=2)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = USERS.get(username)
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    if bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        access_token = create_access_token(
            identity=username,
            additional_claims={"role": user["role"]}
        )
        return jsonify({
            "access_token": access_token,
            "username": username,
            "role": user["role"]
        }), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/api/verify-token", methods=["GET"])
@jwt_required()
def verify_token():
    current_user = get_jwt_identity()
    user = USERS.get(current_user)
    return jsonify({
        "username": current_user,
        "role": user["role"] if user else "user"
    }), 200

@app.route("/api/history", methods=["GET"])
@jwt_required()
def get_history():
    history = load_history()
    # Sort by timestamp (newest first)
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(history), 200

@app.route("/api/history/<history_id>", methods=["GET"])
@jwt_required()
def get_history_detail(history_id):
    history = load_history()
    record = next((h for h in history if h.get("id") == history_id), None)
    if not record:
        return jsonify({"error": "History record not found"}), 404
    return jsonify(record), 200

@app.route("/api/history/<history_id>", methods=["DELETE"])
@jwt_required()
def delete_history(history_id):
    history = load_history()
    record = next((h for h in history if h.get("id") == history_id), None)
    
    if not record:
        return jsonify({"error": "History record not found"}), 404
    
    # Remove from history
    history = [h for h in history if h.get("id") != history_id]
    save_history(history)
    
    # Delete associated files
    if record.get("output_video_path") and os.path.exists(record["output_video_path"]):
        try:
            os.remove(record["output_video_path"])
        except:
            pass
    
    if record.get("input_video_path") and os.path.exists(record["input_video_path"]):
        try:
            os.remove(record["input_video_path"])
        except:
            pass
    
    return jsonify({"message": "History record deleted"}), 200

@app.route("/api/stats", methods=["GET"])
@jwt_required()
def get_stats():
    history = load_history()
    
    total_videos = len(history)
    total_vehicles = sum(len(h.get("all_logs", [])) for h in history)
    total_violations = sum(len(h.get("overspeed_summary", [])) for h in history)
    
    # Calculate average speed across all videos
    all_speeds = []
    for h in history:
        for log in h.get("all_logs", []):
            if isinstance(log.get("speed"), (int, float)):
                all_speeds.append(log["speed"])
    
    avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
    max_speed = max(all_speeds) if all_speeds else 0
    
    # Recent activity (last 7 days)
    seven_days_ago = datetime.now() - timedelta(days=7)
    recent_videos = []
    for h in history:
        try:
            timestamp = datetime.fromisoformat(h.get("timestamp", ""))
            if timestamp >= seven_days_ago:
                recent_videos.append(h)
        except:
            pass
    
    return jsonify({
        "total_videos": total_videos,
        "total_vehicles": total_vehicles,
        "total_violations": total_violations,
        "avg_speed": round(avg_speed, 1),
        "max_speed": round(max_speed, 1),
        "recent_activity": len(recent_videos),
        "violation_rate": round((total_violations / total_vehicles * 100) if total_vehicles > 0 else 0, 1)
    }), 200

@app.route("/upload-and-process", methods=["POST"])
@jwt_required()
def upload_and_process():
    current_user = get_jwt_identity()
    
    if "video" not in request.files:
        return jsonify({"error": "No file part named 'video' found"}), 400

    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400

    orig_name = secure_filename(f.filename)
    unique_name = f"{uuid.uuid4().hex}_{orig_name}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    f.save(save_path)

    try:
        overspeed_limit_kmh = float(request.form.get("overspeed_limit_kmh", 60))
    except:
        overspeed_limit_kmh = 60.0
    try:
        distance_meters = float(request.form.get("distance_meters", 20.0))
    except:
        distance_meters = 20.0

    try:
        result = process_video_for_speed(save_path,
                                         overspeed_limit_kmh=overspeed_limit_kmh,
                                         distance_meters=distance_meters)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    if isinstance(result, dict) and result.get("error"):
        return jsonify(result), 500

    output_video_path = result.get("output_video_path")
    overspeed_summary = result.get("overspeed_summary", [])
    all_logs = result.get("all_logs", [])

    output_url = None
    download_name = None
    if output_video_path and os.path.isfile(output_video_path):
        download_name = os.path.basename(output_video_path)
        output_url = f"/download/{download_name}"

    # Save to history
    history_record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "user": current_user,
        "original_filename": orig_name,
        "input_video_path": save_path,
        "output_video_path": output_video_path,
        "download_name": download_name,
        "overspeed_limit": overspeed_limit_kmh,
        "distance_meters": distance_meters,
        "total_vehicles": len(all_logs),
        "total_violations": len(overspeed_summary),
        "overspeed_summary": overspeed_summary,
        "all_logs": all_logs
    }
    
    history = load_history()
    history.append(history_record)
    save_history(history)

    response = {
        "status": "completed",
        "output_video_url": output_url,
        "download_name": download_name,
        "overspeed_summary": overspeed_summary,
        "all_logs": all_logs,
        "history_id": history_record["id"]
    }
    return jsonify(response), 200

@app.route("/download/<filename>", methods=["GET"])
@jwt_required()
def download_file(filename):
    safe_name = secure_filename(filename)
    found = None
    for root, dirs, files in os.walk(UPLOAD_DIR):
        if safe_name in files:
            found = os.path.join(root, safe_name)
            break
    if not found or not os.path.isfile(found):
        return abort(404, description="File not found")
    return send_file(found, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)