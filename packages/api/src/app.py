from flask import Flask, request, jsonify, make_response
from transummary import TranSummary
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
cors = CORS(app)


@app.route("/parse-video", methods=["POST"])
def parse_video():
    data = request.get_json()
    url = data.get("url", "")
    trans = TranSummary(url=url)
    segments = trans.summarize_segments()
    return jsonify({"videoId": trans.yt_id, "segments": segments})


@app.route("/")
def home():
    return "Up"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=4000)
