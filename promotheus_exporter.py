print(">>> TES 1: Skrip prometheus_exporter.py MULAI DI SINI <<<", flush=True)

try:
    from flask import Flask, request, jsonify
    print(">>> TES 2: Berhasil impor Flask <<<", flush=True)

    import mlflow
    print(">>> TES 3: Berhasil impor MLflow <<<", flush=True)

    import pandas as pd
    print(">>> TES 4: Berhasil impor Pandas <<<", flush=True)

    import json
    print(">>> TES 5: Semua impor library dasar selesai <<<", flush=True)

    from prometheus_client import Counter, generate_latest, REGISTRY, CONTENT_TYPE_LATEST
    print(">>> TES 6: Berhasil impor Prometheus Client <<<", flush=True)

except ImportError as e:
    print(f">>> ERROR IMPORT: Gagal mengimpor library. <<<\n{e}", flush=True)
    exit(1)

# ===== Inisialisasi Flask =====
app = Flask(__name__)
print(">>> TES 7: Aplikasi Flask berhasil diinisialisasi <<<", flush=True)

# ===== METRICS Prometheus =====
by_path_counter = Counter("by_path_counter", "Request count by request paths", ["path"])
http_requests_total = Counter("http_requests_total", "Total HTTP requests received", ["method", "endpoint"])
model_inference_success_total = Counter("model_inference_success_total", "Total successful inference requests")
model_inference_fail_total = Counter("model_inference_fail_total", "Total failed inference requests")

print(">>> TES 8: Prometheus Counter berhasil diinisialisasi <<<", flush=True)

# ===== MUAT MODEL =====
model_uri = r"file:///D:/Kuliah/Coding%20Camp%202025/MSML_Era_Syafina/Eksperimen_SML_Era-Syafina/Membangun_Model/mlartifacts/696721140867107780/9e472f0aabaf4b0aae6bb33c6cc78c53/artifacts/recommender_pyfunc"

model = None
try:
    print(f">>> TES 9: Mencoba memuat model dari: {model_uri} <<<", flush=True)
    model = mlflow.pyfunc.load_model(model_uri)
    print(">>> TES 10: Model pyfunc berhasil dimuat. <<<", flush=True)
except Exception as e:
    print(f">>> ERROR SAAT LOAD MODEL: {e} <<<", flush=True)

# ===== ROUTE =====
@app.route("/", methods=["GET"])
def home():
    return "<h1>Movie Recommender API</h1><p>Gunakan endpoint /recommend untuk mendapatkan rekomendasi.</p>"

@app.route("/recommend", methods=["POST"])
def recommend():
    http_requests_total.labels(method="POST", endpoint="/recommend").inc()
    by_path_counter.labels(path="/recommend").inc()

    if model is None:
        model_inference_fail_total.inc()
        return jsonify({"error": "Model tidak berhasil dimuat."}), 500

    try:
        req_data = request.get_json()
        titles = req_data.get("titles", [])

        if not titles or not isinstance(titles, list):
            model_inference_fail_total.inc()
            return jsonify({"error": "Input JSON tidak valid."}), 400

        input_df = pd.DataFrame({"title": titles})
        results = model.predict(input_df)

        model_inference_success_total.inc()

        response_data = [{"input_title": title, "recommendations": results[i]} for i, title in enumerate(titles)]
        return jsonify(response_data)

    except Exception as e:
        model_inference_fail_total.inc()
        return jsonify({"error": str(e)}), 500

@app.route("/metrics")
def metrics():
    return generate_latest(REGISTRY), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# ===== START SERVER =====
if __name__ == "__main__":
    print(">>> TES 11: Masuk ke blok __main__ untuk menjalankan server <<<", flush=True)
    app.run(host="0.0.0.0", port=8000, debug=True)
