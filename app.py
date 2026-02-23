from flask import Flask, request, jsonify
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

app = Flask(__name__)

# Load model + tokenizer once at startup
MODEL_NAME = "facebook/m2m100_418M"  # smaller, faster model for dev
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)

# Get all supported languages from tokenizer
SUPPORTED_LANGS = tokenizer.lang_code_to_id.keys()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Translator API is running. Use /translate with POST JSON.",
        "supported_languages": list(SUPPORTED_LANGS)
    })

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json(force=True)
        text = data.get("text")
        src_lang = data.get("src_lang", "en")
        tgt_lang = data.get("tgt_lang", "hi")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if src_lang not in SUPPORTED_LANGS or tgt_lang not in SUPPORTED_LANGS:
            return jsonify({
                "error": "Unsupported language code",
                "supported_languages": list(SUPPORTED_LANGS)
            }), 400

        # Set source language
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt")

        # Generate translation
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=100
        )

        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        return jsonify({
            "source_text": text,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "translation": translation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)