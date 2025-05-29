**📘 Project Overview & Feature Explanation: Multi-Modal Malicious URL Detection System**

---

### 🎯 Purpose

This system is designed to detect potentially malicious URLs using a combination of:

* **Textual pattern recognition** (via CNN trained on character-level URL tokens)
* **Visual pattern analysis** (via TF-IDF-based grayscale image generation from URLs)

By integrating these two modalities (multi-modal learning), the system improves accuracy, generalization, and resilience against obfuscated phishing URLs (e.g., via `ngrok`, `bit.ly`, etc.).

---

### 🔍 Key Features

#### 1. **Character-Level Text CNN**

* Converts URLs into sequences of characters
* Encodes them as fixed-length numeric tensors
* Trained on labeled malicious/benign URLs
* Learns patterns like `login`, `verify`, domain nesting

#### 2. **TF-IDF to Grayscale Image**

* Uses n-gram-based TF-IDF vectorization
* Converts each URL into a fixed-size (64x64) image
* Bright pixels indicate high-weight n-grams
* Fed into a CNN to detect visual threat patterns

#### 3. **Fusion Model**

* Combines feature vectors from the text and image models
* Fully connected classifier with sigmoid output
* Trained to detect threats with improved accuracy

#### 4. **FastAPI Backend**

* Endpoint: `/scan/url`
* Accepts a URL input, returns prediction + TF-IDF image URL
* Serves images as static files

#### 5. **React Frontend**

* Input bar to submit URL for scanning
* Displays result: `malicious` or `benign`
* Shows confidence score and grayscale image visualization

---

### 🧠 Why Multi-Modal?

* **Single-mode models** (text or image alone) suffer in generalization
* Multi-modal fusion helps catch:

  * Phishing via shorteners
  * Hidden characters in subdomains
  * Token-pattern abuse ("@", long URLs, IPs)

---

### 🧪 Edge Case Handling

* URLs with long routing paths
* IP-based URLs
* Suspicious subdomain nesting
* Whitelist logic and rule-based heuristics (optional enhancement)

---

### 💡 Use Case Scenarios

* **Security dashboards**
* **Email/messaging link scanners**
* **Browser extensions or proxy filters**
* **Educational demos for phishing detection**

---

### 📈 Future Enhancements

* Add Base64 image support (no file saving)
* Connect to VirusTotal/PhishTank APIs for additional validation
* Add SHAP/LIME explainability for CNN features
* Integrate database logging for analytics and history

---

This project blends traditional NLP + modern CV to deliver a responsive, intelligent malicious URL scanner that is explainable and user-friendly.
