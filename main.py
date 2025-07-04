import os
import io
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI app setup
app = FastAPI(
    title="Nyay-Setu - Your AI Lawyer",
    description="A voice-driven, multilingual AI legal assistant backend",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MVP GLOBAL STORAGE ---
# NOTE: For production, use proper session management or database
document_context = {
    "content": None,
    "summary": None,
    "filename": None,
    "processed_at": None
}

# --- Request Models ---
class QueryRequest(BaseModel):
    query: str
    language: str = "en"

# --- Helper Functions ---
def extract_pdf_text(pdf_content: bytes) -> str:
    """Extract text from PDF bytes"""
    reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

async def generate_summary(text: str) -> str:
    """Generate document summary using Gemini"""
    prompt = (
        "You are an AI legal assistant. Analyze this legal document and provide a concise summary "
        "highlighting the key legal points, important clauses, parties involved, and main provisions. "
        "Focus on information that would be useful for answering questions about this document.\n\n"
        f"Document Content:\n{text}\n\n"
        "Provide a structured summary covering the main legal aspects."
    )
    
    response = await asyncio.to_thread(
        lambda: GEMINI_MODEL.generate_content(prompt)
    )
    return response.text.strip()

async def query_document(query: str, language: str) -> str:
    """Query the document using Gemini"""
    prompt = (
        "You are Nyay-Setu, an AI legal assistant. Answer the user's question based ONLY on the provided document.\n"
        "Guidelines:\n"
        "- Only use information from the document provided\n"
        "- If the information is not in the document, say so clearly\n"
        "- Provide informational responses, not legal advice\n"
        "- Be precise and cite relevant sections when possible\n"
        f"- Respond in this language: {language}\n\n"
        f"Document Summary:\n{document_context['summary']}\n\n"
        f"Full Document Content:\n{document_context['content']}\n\n"
        f"User Question: {query}\n\n"
        "Answer:"
    )
    
    response = await asyncio.to_thread(
        lambda: GEMINI_MODEL.generate_content(prompt)
    )
    return response.text.strip()

# --- API Endpoints ---
@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Nyay-Setu AI Legal Assistant API",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/status")
async def get_status():
    """Check if a document is loaded and ready for queries"""
    if document_context["content"]:
        return {
            "document_loaded": True,
            "filename": document_context["filename"],
            "processed_at": document_context["processed_at"],
            "summary_preview": document_context["summary"][:200] + "..." if len(document_context["summary"]) > 200 else document_context["summary"],
            "document_length": len(document_context["content"])
        }
    else:
        return {
            "document_loaded": False,
            "message": "No document loaded. Please upload a PDF document first."
        }

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    Extracts text, generates summary, and stores context for querying.
    """
    global document_context
    
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported. Please upload a PDF document."
        )
    
    try:
        # Read and extract text from PDF
        pdf_content = await file.read()
        extracted_text = extract_pdf_text(pdf_content)
        
        if not extracted_text:
            raise HTTPException(
                status_code=400, 
                detail="No readable text found in the PDF. Please ensure the PDF contains text content."
            )
        
        print(f"Processing document: {file.filename}")
        print(f"Extracted text length: {len(extracted_text)} characters")
        
        # Generate summary using Gemini
        summary = await generate_summary(extracted_text)
        
        # Store context globally (MVP approach)
        from datetime import datetime
        document_context = {
            "content": extracted_text,
            "summary": summary,
            "filename": file.filename,
            "processed_at": datetime.now().isoformat()
        }
        
        print(f"Document processed successfully: {file.filename}")
        
        return {
            "success": True,
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "document_length": len(extracted_text),
            "summary_preview": summary[:300] + "..." if len(summary) > 300 else summary,
            "ready_for_queries": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process document: {str(e)}"
        )

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Ask a question about the uploaded document.
    Returns an AI-generated answer based on the document content.
    """
    # Check if document is loaded
    if not document_context["content"]:
        raise HTTPException(
            status_code=404,
            detail="No document loaded. Please upload a PDF document first using /upload-document"
        )
    
    try:
        print(f"Processing query: {request.query} (Language: {request.language})")
        
        # Get AI response
        response = await query_document(request.query, request.language)
        
        return {
            "question": request.query,
            "answer": response,
            "language": request.language,
            "document": document_context["filename"]
        }
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process your question: {str(e)}"
        )

@app.delete("/clear-document")
async def clear_document():
    """Clear the currently loaded document"""
    global document_context
    document_context = {
        "content": None,
        "summary": None,
        "filename": None,
        "processed_at": None
    }
    return {"message": "Document cleared successfully"}

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred", "details": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)