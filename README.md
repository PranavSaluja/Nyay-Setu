# Nyay-Setu - Your AI Lawyer (Backend)

## Setup Instructions

### 1. Clone the repository and enter the project directory
```
git clone <repo-url>
cd nyay-setu-backend
```

### 2. Create and activate a virtual environment (recommended)
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up your `.env` file
Create a `.env` file in the project root with the following content:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```
Replace `your_google_gemini_api_key_here` with your actual Gemini API key.

### 5. Run the FastAPI app using Uvicorn
```
uvicorn main:app --reload
```

The backend will be available at `http://127.0.0.1:8000/`.

---

## Endpoints Overview
- `GET /` — Health check
- `POST /process-document/` — Upload and process a PDF legal document
- `POST /receive-context/` — Store document context (MVP, global vars)
- `POST /query-document/` — Query the document context with a user question

---

**Note:**
- This MVP uses global variables for context storage, which is not suitable for production or multi-user scenarios. Proper session or database management will be implemented in future versions. 