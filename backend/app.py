from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import re
from collections import Counter
import math
import json

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────

SPAM_KEYWORDS = [
    "free", "win", "winner", "cash", "prize", "congratulations", "click here",
    "buy now", "limited offer", "act now", "urgent", "guaranteed", "risk-free",
    "make money", "earn", "income", "credit", "loan", "debt", "insurance",
    "viagra", "pills", "pharmacy", "discount", "cheap", "sale", "offer",
    "subscribe", "unsubscribe", "opt-out", "promotion", "marketing",
    "100%", "absolutely", "amazing", "best", "deal", "double", "extra",
    "fantastic", "incredible", "miracle", "special", "incredible",
]

AI_PATTERNS = [
    "as an ai", "as a language model", "i cannot", "i'm unable",
    "delve", "certainly", "it's important to note", "in conclusion",
    "furthermore", "moreover", "in summary", "to summarize",
    "it is worth noting", "it should be noted", "importantly",
    "significantly", "notably", "of course", "absolutely",
    "comprehensive", "straightforward", "i'd be happy",
    "as mentioned", "in the realm of", "leverage", "utilize",
    "multifaceted", "nuanced", "it's crucial", "pivotal",
    "game-changing", "innovative", "seamlessly", "robust",
]

def extract_features(text: str) -> dict:
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    char_count = len(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_len = word_count / sentence_count

    # Spam signals
    spam_hits = [kw for kw in SPAM_KEYWORDS if kw in text_lower]
    spam_keyword_ratio = len(spam_hits) / max(word_count, 1)
    exclamation_ratio = text.count('!') / max(char_count, 1)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
    url_count = len(re.findall(r'http[s]?://|www\.', text_lower))
    dollar_signs = text.count('$') + text.count('€') + text.count('£')
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(char_count, 1)
    repeated_punct = len(re.findall(r'[!?]{2,}', text))

    # AI signals
    ai_hits = [p for p in AI_PATTERNS if p in text_lower]
    ai_phrase_ratio = len(ai_hits) / max(word_count, 1) * 10
    type_token_ratio = len(set(words)) / max(word_count, 1)

    # Vocabulary sophistication
    long_words = [w for w in words if len(w) > 8]
    long_word_ratio = len(long_words) / max(word_count, 1)

    # Sentence length variance
    sent_lens = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    sent_variance = np.var(sent_lens) if len(sent_lens) > 1 else 0

    # Punctuation diversity
    punct_types = len(set(c for c in text if c in '.,;:!?-()[]{}"\'-'))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_len": round(avg_word_len, 3),
        "avg_sentence_len": round(avg_sentence_len, 3),
        "spam_keyword_ratio": round(spam_keyword_ratio, 4),
        "exclamation_ratio": round(exclamation_ratio, 4),
        "caps_ratio": round(caps_ratio, 4),
        "url_count": url_count,
        "dollar_signs": dollar_signs,
        "digit_ratio": round(digit_ratio, 4),
        "repeated_punct": repeated_punct,
        "ai_phrase_ratio": round(ai_phrase_ratio, 4),
        "type_token_ratio": round(type_token_ratio, 4),
        "long_word_ratio": round(long_word_ratio, 4),
        "sent_variance": round(sent_variance, 3),
        "punct_diversity": punct_types,
        "spam_keywords_found": spam_hits[:8],
        "ai_phrases_found": ai_hits[:6],
    }

# ─────────────────────────────────────────────
#  Neural Network (NumPy, from scratch)
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

class NeuralNet:
    """3-class neural network: HUMAN(0), AI(1), SPAM(2)"""

    def __init__(self):
        np.random.seed(42)
        # Input: 14 numerical features
        # Architecture: 14 → 32 → 16 → 3
        self.W1 = np.random.randn(14, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 16) * 0.1
        self.b2 = np.zeros(16)
        self.W3 = np.random.randn(16, 3) * 0.1
        self.b3 = np.zeros(3)
        self._pretrain()

    def _pretrain(self):
        """Inject domain knowledge via synthetic training."""
        import time
        samples = self._generate_synthetic_data(1500)
        lr = 0.02
        total = 100
        start = time.time()
        for epoch in range(total):
            np.random.shuffle(samples)
            for x, y in samples:
                self._update(x, y, lr)
            if epoch == 40: lr = 0.008
            if epoch == 75: lr = 0.002

            # Progress bar
            pct     = (epoch + 1) / total
            filled  = int(30 * pct)
            bar     = "#" * filled + "-" * (30 - filled)
            elapsed = time.time() - start
            eta     = (elapsed / pct) * (1 - pct) if pct > 0 else 0
            print(f"\r  [{bar}] {epoch+1}/{total}  "
                  f"lr={lr:.4f}  "
                  f"elapsed={elapsed:.1f}s  "
                  f"eta={eta:.1f}s   ",
                  end="", flush=True)
        print()  # newline after training completes

    def _generate_synthetic_data(self, n):
        """
        Generate pre-normalised feature vectors in the SAME [0,1] scale
        that _feature_vector() produces at inference time.

        Feature order (14 dims):
          0  char_count   / 2000   (clipped 0-1)
          1  word_count   / 500
          2  sentence_cnt / 30
          3  avg_word_len / 10
          4  avg_sent_len / 40
          5  spam_kw_ratio * 20
          6  excl_ratio   * 100
          7  caps_ratio   * 5
          8  url_count    / 5
          9  dollar_signs / 5
         10  digit_ratio  * 10
         11  repeated_pct / 5
         12  ai_phrase_ratio (already 0-1)
         13  type_token_ratio (already 0-1)
        """
        data = []
        rng = np.random.default_rng(42)
        U = lambda lo, hi: float(rng.uniform(lo, hi))
        per = n // 3

        for _ in range(per):
            # ── HUMAN ──────────────────────────────────────────────────────
            # Moderate length, low caps, low spam, medium TTR, varied sentences
            wc = rng.uniform(50, 300)
            x = np.array([
                min(wc * rng.uniform(4.5, 6.5) / 2000, 1),   # 0 char/2000
                min(wc / 500, 1),                              # 1 word/500
                min(rng.uniform(3, 18) / 30, 1),               # 2 sent/30
                U(0.35, 0.62),                                  # 3 awl/10
                min(rng.uniform(10, 25) / 40, 1),              # 4 asl/40
                U(0.0,  0.04),                                  # 5 spam *20
                U(0.0,  0.012),                                 # 6 excl *100
                U(0.0,  0.10),                                  # 7 caps *5
                U(0.0,  0.10),                                  # 8 url/5
                U(0.0,  0.02),                                  # 9 dollar/5
                U(0.0,  0.08),                                  # 10 digit*10
                U(0.0,  0.04),                                  # 11 rpunct/5
                U(0.0,  0.04),                                  # 12 ai_phrase
                U(0.55, 0.85),                                  # 13 ttr
            ], dtype=float)
            data.append((np.clip(x, 0, 1), 0))

        for _ in range(per):
            # ── AI ─────────────────────────────────────────────────────────
            # High TTR, many long words, AI phrases, uniform sentence length
            wc = rng.uniform(100, 500)
            x = np.array([
                min(wc * rng.uniform(5.5, 7.0) / 2000, 1),
                min(wc / 500, 1),
                min(rng.uniform(4, 22) / 30, 1),
                U(0.50, 0.78),                                  # longer words
                min(rng.uniform(18, 35) / 40, 1),              # longer sentences
                U(0.0,  0.015),                                 # almost no spam kw
                U(0.0,  0.003),                                 # very few !
                U(0.0,  0.05),                                  # low caps
                U(0.0,  0.04),                                  # few urls
                U(0.0,  0.01),
                U(0.0,  0.04),
                U(0.0,  0.02),
                U(0.10, 1.00),                                  # HIGH ai phrases
                U(0.72, 1.00),                                  # HIGH ttr
            ], dtype=float)
            data.append((np.clip(x, 0, 1), 1))

        for _ in range(per):
            # ── SPAM ───────────────────────────────────────────────────────
            # High caps, lots of !, spam keywords, urls, dollar signs
            wc = rng.uniform(15, 120)
            x = np.array([
                min(wc * rng.uniform(4.0, 6.0) / 2000, 1),
                min(wc / 500, 1),
                min(rng.uniform(1, 10) / 30, 1),
                U(0.25, 0.55),
                min(rng.uniform(5, 20) / 40, 1),
                U(0.25, 1.00),                                  # HIGH spam kw*20
                U(0.10, 1.00),                                  # HIGH excl*100
                U(0.30, 1.00),                                  # HIGH caps*5
                U(0.10, 1.00),                                  # URLs
                U(0.05, 1.00),                                  # dollar signs
                U(0.05, 0.50),
                U(0.05, 1.00),                                  # repeated punct
                U(0.0,  0.03),
                U(0.35, 0.72),                                  # lower ttr
            ], dtype=float)
            data.append((np.clip(x, 0, 1), 2))

        return data

    def _feature_vector(self, feat: dict) -> np.ndarray:
        return np.array([
            min(feat["char_count"] / 2000, 1),
            min(feat["word_count"] / 500, 1),
            min(feat["sentence_count"] / 30, 1),
            min(feat["avg_word_len"] / 10, 1),
            min(feat["avg_sentence_len"] / 40, 1),
            min(feat["spam_keyword_ratio"] * 20, 1),
            min(feat["exclamation_ratio"] * 100, 1),
            min(feat["caps_ratio"] * 5, 1),
            min(feat["url_count"] / 5, 1),
            min(feat["dollar_signs"] / 5, 1),
            min(feat["digit_ratio"] * 10, 1),
            min(feat["repeated_punct"] / 5, 1),
            min(feat["ai_phrase_ratio"], 1),
            feat["type_token_ratio"],
        ], dtype=float)

    def _forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = softmax(self.z3)
        return self.out

    def _update(self, x, y_idx, lr):
        probs = self._forward(x)
        target = np.zeros(3); target[y_idx] = 1
        d_out = probs - target

        dW3 = np.outer(self.a2, d_out)
        db3 = d_out
        d_a2 = d_out @ self.W3.T
        d_z2 = d_a2 * (self.z2 > 0)

        dW2 = np.outer(self.a1, d_z2)
        db2 = d_z2
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (self.z1 > 0)

        dW1 = np.outer(x, d_z1)
        db1 = d_z1

        self.W3 -= lr * dW3; self.b3 -= lr * db3
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1

    def predict(self, feat: dict):
        x = self._feature_vector(feat)
        probs = self._forward(x)
        activations = {
            "layer1": self.a1.tolist(),
            "layer2": self.a2.tolist(),
            "output": self.out.tolist(),
        }
        return {
            "human": round(float(probs[0]), 4),
            "ai":    round(float(probs[1]), 4),
            "spam":  round(float(probs[2]), 4),
        }, activations


# Initialise model once at startup
print("Training neural network...")
model = NeuralNet()
print("Model ready")

# ─────────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) < 10:
        return jsonify({"error": "Text too short (min 10 chars)"}), 400
    if len(text) > 10000:
        return jsonify({"error": "Text too long (max 10 000 chars)"}), 400

    features = extract_features(text)
    probabilities, activations = model.predict(features)

    label_map = {0: "human", 1: "ai", 2: "spam"}
    predicted_idx = int(np.argmax([probabilities["human"],
                                   probabilities["ai"],
                                   probabilities["spam"]]))
    predicted_label = label_map[predicted_idx]
    confidence = max(probabilities.values())

    # Build explanation
    signals = []
    if features["spam_keywords_found"]:
        signals.append(f"Spam keywords: {', '.join(features['spam_keywords_found'][:5])}")
    if features["ai_phrases_found"]:
        signals.append(f"AI phrases: {', '.join(features['ai_phrases_found'][:4])}")
    if features["caps_ratio"] > 0.08:
        signals.append(f"High CAPS ratio ({features['caps_ratio']*100:.1f}%)")
    if features["exclamation_ratio"] > 0.003:
        signals.append("Excessive exclamation marks")
    if features["url_count"] > 0:
        signals.append(f"{features['url_count']} URL(s) detected")
    if features["type_token_ratio"] > 0.8:
        signals.append("High lexical diversity (AI-like)")
    if features["ai_phrase_ratio"] > 0.02:
        signals.append("AI-style phrasing detected")
    if features["avg_sentence_len"] > 25:
        signals.append("Long, complex sentences (AI-like)")

    return jsonify({
        "prediction": predicted_label,
        "confidence": round(confidence * 100, 1),
        "probabilities": probabilities,
        "features": features,
        "signals": signals,
        "activations": {
            "layer1_sample": activations["layer1"][:16],
            "layer2_sample": activations["layer2"][:16],
            "output": activations["output"],
        },
        "architecture": {"input": 14, "hidden1": 32, "hidden2": 16, "output": 3}
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "NeuralNet 14→32→16→3"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)