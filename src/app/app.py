from flask import Flask, request, render_template_string, jsonify
import os
from ..infer.infer import infer_single_sentence
from ..config import cfg

# Simple Flask app — loads the BiLSTM model checkpoint on start.
MODEL_PATH = os.path.join(cfg.models_dir, 'best_bilstm_crf.pt')

app = Flask(__name__)

HTML = """
<!doctype html>
<title>Arabic Diacritizer Demo</title>
<h1>Arabic Diacritizer (BiLSTM+CRF)</h1>
<form action="/diacritize" method="post">
  <textarea name="sentence" rows=4 cols=60 placeholder="Enter undiacritized Arabic sentence..."></textarea><br>
  <input type="submit" value="Diacritize">
</form>
{% if result %}
  <h2>Result</h2>
  <div style="font-size:20px;">{{ result }}</div>
{% endif %}
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/diacritize", methods=["POST"])
def diacritize():
    sentence = request.form.get("sentence", "").strip()
    if not sentence:
        return render_template_string(HTML, result="Please provide a sentence.")
    try:
        out = infer_single_sentence(sentence, MODEL_PATH)
    except Exception as e:
        return render_template_string(HTML, result=f"Error: {e}")
    return render_template_string(HTML, result=out)

@app.route("/api/diacritize", methods=["POST"])
def api_diacritize():
    data = request.json
    sentence = data.get("sentence", "")
    out = infer_single_sentence(sentence, MODEL_PATH)
    return jsonify({"diacritized": out})

if __name__ == "__main__":
    # Ensure model exists
    if not os.path.exists(MODEL_PATH):
        print("Model not found at", MODEL_PATH)
    app.run(host="0.0.0.0", port=5000, debug=False)
