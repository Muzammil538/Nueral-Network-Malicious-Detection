**📦 Project Installation Guide: Multi-Modal Malicious URL Detection System**

---

### ✅ Prerequisites

* Python 3.9 or 3.10 (recommended)
* pip
* Virtual environment (recommended)

---

### 📁 Folder Structure

```
project/
├── dataset/full_urls.csv
├── preprocess/
│   ├── char_tokenizer.py
│   ├── tfidf_to_image.py
│   └── tfidf_vectorizer.pkl
├── models/
│   ├── text_model.py
│   ├── image_model.py
│   ├── fusion_model.py
│   └── train.py
├── backend/
│   ├── app.py
│   └── utils.py
├── saved_models/
│   ├── text_model.pth
│   ├── image_model.pth
│   └── fusion_model.pth
├── frontend/  (React app)
└── requirements.txt
```

---

### 🔧 Setup Instructions

#### 1. Clone the Repo & Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Prepare Dataset

Ensure `dataset/full_urls.csv` exists:

```csv
url,label
https://google.com,0
http://login-update.ngrok.io,1
```

#### 3. Train and Save TF-IDF Vectorizer

```bash
python preprocess/save_vectorizer.py
```

#### 4. Train the Models

```bash
python models/train.py
```

This saves `text_model.pth`, `image_model.pth`, and `fusion_model.pth` to `saved_models/`

#### 5. Start the FastAPI Backend

```bash
uvicorn backend.app:app --reload --port 5000
```

#### 6. Start the React Frontend (from `frontend/` folder)

```bash
npm install
npm start
```

---

### 🌐 API Endpoint

```http
POST http://localhost:5000/scan/url
Body: { "url": "http://example.com/login" }
```

Response:

```json
{
  "prediction": "malicious",
  "confidence": 0.948,
  "image_url": "http://localhost:5000/static/images/<img>.png"
}
```

---

### ✅ Done

You now have a complete working system to:

* Scan URLs
* Predict malicious intent
* Show visual TF-IDF image
* Serve via frontend/backend architecture
