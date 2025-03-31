from flask import Flask, request, jsonify, send_from_directory, Response
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Set security headers to enable SharedArrayBuffer
@app.after_request
def set_headers(response: Response):
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Access-Control-Allow-Origin"] = "*"  # Allow all origins (modify for production)
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/get_ngrok_url")
def serve_ngrok_url():
    """Returns the current ngrok URL for the frontend."""
    try:
        import requests
        response = requests.get("http://127.0.0.1:4040/api/tunnels")
        url = response.json()["tunnels"][0]["public_url"]
        return jsonify({"url": url})
    except Exception as e:
        return jsonify({"error": "Ngrok is not running", "details": str(e)})

@app.route("/upload", methods=["POST"])
def upload_audio():
    """Handles file uploads."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return jsonify({"message": "File uploaded successfully!", "filename": file.filename})

# ✅ Serve a dummy favicon to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, ''), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
