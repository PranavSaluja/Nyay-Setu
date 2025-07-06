import os
import io
import asyncio
import base64
import requests # Required for ElevenLabs API calls

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# Defaulting to a common voice ID if not specified in .env,
#add
# but it's best to explicitly set ELEVENLABS_VOICE_ID in your .env
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "1qEiC6qsybMkmnNdVMbK") 

# Validate environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure Gemini AI model
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# Initialize FastAPI application
app = FastAPI(
    title="Nyay-Setu - Your AI Lawyer",
    description="A voice-driven, multilingual AI legal assistant backend",
    version="1.0.0"
)

# Configure CORS middleware to allow requests from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, consider narrowing for production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# --- MVP GLOBAL STORAGE FOR DOCUMENT CONTEXT ---
# NOTE: For production environments, this should be replaced with proper session management,
# a database, or a caching system (e.g., Redis) to manage state across requests and users.
document_context = {
    "content": None,        # Stores the full text content of the uploaded PDF
    "summary": None,        # Stores the AI-generated summary of the document
    "filename": None,       # Stores the original filename of the uploaded PDF
    "processed_at": None    # Timestamp of when the document was processed
}

# --- Pydantic Models for API Request/Response Schemas ---
class QueryRequest(BaseModel):
    """
    Request model for the /ask endpoint, carrying Base64 encoded audio
    and desired response language.
    """
    audio_base64: str = Field(..., description="Base64 encoded audio data from the user's speech")
    language: str = Field("en", description="Desired language for the AI's response (e.g., 'en', 'hi', 'gu')")

class AskResponse(BaseModel):
    """
    Response model for the /ask endpoint, providing the AI's answer as audio,
    the original transcribed query, and context information.
    """
    audio_response_base64: str = Field(..., description="Base64 encoded audio of the AI's response")
    original_query: str = Field(..., description="The transcribed text of the user's query")
    response_language: str = Field(..., description="The language of the AI's response")
    document_filename: str | None = Field(None, description="The filename of the document used for the query")

# --- Helper Functions for Core Logic ---

def extract_pdf_text(pdf_content: bytes) -> str:
    """
    Extracts plain text from PDF file bytes.
    Raises an error if the PDF is encrypted or malformed.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or "" # extract_text() can return None
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

async def generate_summary(text: str) -> str:
    """
    Generates a concise summary of the legal document text using the Gemini AI model.
    """
    prompt = (
        "You are an AI legal assistant. Analyze this legal document and provide a concise summary , everytime start with As a lawyer I can tell you ..."
        "highlighting the key legal points, important clauses, parties involved, and main provisions. "
        "Focus on information that would be useful for answering questions about this document.\n\n"
        f"Document Content:\n{text}\n\n"
        "Provide a structured summary covering the main legal aspects."
    )
    
    try:
        # Use asyncio.to_thread to run blocking Gemini call in a separate thread
        response = await asyncio.to_thread(
            lambda: GEMINI_MODEL.generate_content(prompt)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary with Gemini: {e}")
        # Consider more specific error handling or fallback
        raise HTTPException(status_code=500, detail="Failed to generate document summary.")

async def query_document(query: str, language: str) -> str:
    """
    Queries the loaded document using the Gemini AI model, providing answers
    based strictly on the document content and responding in the specified language.
    """
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
    
    try:
        # Use asyncio.to_thread to run blocking Gemini call in a separate thread
        response = await asyncio.to_thread(
            lambda: GEMINI_MODEL.generate_content(prompt)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error querying document with Gemini: {e}")
        # Consider more specific error handling or fallback
        raise HTTPException(status_code=500, detail="Failed to get AI response from document.")

async def elevenlabs_speech_to_text(audio_base64: str) -> str:
    """
    Converts Base64 encoded audio to text using the ElevenLabs Speech-to-Text API.
    Expected audio format: WebM.
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        # 'Content-Type' is automatically set by 'requests' when using 'files' parameter for multipart/form-data
    }
    
    audio_bytes = base64.b64decode(audio_base64)
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    
    # Define the file part of the multipart/form-data request
    # The key for the audio file MUST be "file" as per ElevenLabs documentation
    files = {
        "file": ("audio.webm", io.BytesIO(audio_bytes), "audio/webm") # ('filename', file_object, 'content_type')
    }
    
    # Define the form fields (non-file parts) of the multipart/form-data request
    # 'model_id' is REQUIRED by ElevenLabs STT API
    data = {
        "model_id": "scribe_v1", # Recommended transcription model for ElevenLabs STT
        # You can add "language_code" here if you get it from the frontend,
        # e.g., "language_code": "en" or "language_code": request.language
        # This can help improve transcription accuracy if the language is known.
    }
    
    try:
        # Use asyncio.to_thread to run blocking requests.post in a separate thread
        response = await asyncio.to_thread(
            lambda: requests.post(url, headers=headers, files=files, data=data) 
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        # Check if the response contains the expected 'text' field
        transcribed_text = response.json().get("text", "")
        if not transcribed_text:
            print(f"ElevenLabs STT API returned empty text. Full response: {response.json()}")
            return "" # Return empty string if no text found
        return transcribed_text
        
    except requests.exceptions.RequestException as e:
        # Catch specific request exceptions (network issues, bad responses)
        print(f"ElevenLabs STT API request error: {e}")
        # Print the raw response content from ElevenLabs for detailed debugging
        if response is not None and hasattr(response, 'text'):
            print(f"ElevenLabs STT API raw response content: {response.text}")
        raise HTTPException(
            status_code=500, detail=f"Speech-to-text conversion failed: {e}. Check server logs for details."
        )
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred in elevenlabs_speech_to_text: {e}")
        raise HTTPException(
            status_code=500, detail=f"Speech-to-text conversion failed unexpectedly: {e}"
        )


async def elevenlabs_text_to_speech(text: str, language: str) -> str:
    """
    Converts text to Base64 encoded audio using the ElevenLabs Text-to-Speech API.
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json" # This content type is for the JSON payload
    }
    
    # Payload for the TTS API
    json_data = {
        "text": text,
        "model_id": "eleven_multilingual_v2", # Good for multilingual support
        "voice_settings": {
            "stability": 0.75, # Controls consistency of the voice, less = more expressive
            "similarity_boost": 0.75 # Boosts similarity to the original voice, less = more creative
        },
        # Adding language_code can sometimes improve pronunciation for specific languages,
        # but check if your chosen model_id ('eleven_multilingual_v2') fully supports language enforcement.
        # "language_code": language, # Uncomment if you map your 'language' to ISO 639-1 codes ElevenLabs understands
    }
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    
    try:
        # Use asyncio.to_thread to run blocking requests.post in a separate thread
        response = await asyncio.to_thread(
            lambda: requests.post(url, headers=headers, json=json_data)
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # ElevenLabs returns raw audio bytes, encode them to Base64
        return base64.b64encode(response.content).decode("utf-8")
        
    except requests.exceptions.RequestException as e:
        # Catch specific request exceptions
        print(f"ElevenLabs TTS API request error: {e}")
        # Print the raw response content from ElevenLabs for detailed debugging
        if response is not None and hasattr(response, 'text'):
            print(f"ElevenLabs TTS API raw response content: {response.text}")
        raise HTTPException(
            status_code=500, detail=f"Text-to-speech conversion failed: {e}. Check server logs for details."
        )
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred in elevenlabs_text_to_speech: {e}")
        raise HTTPException(
            status_code=500, detail=f"Text-to-speech conversion failed unexpectedly: {e}"
        )


# --- API Endpoints ---

@app.get("/")
async def root():
    """API health check endpoint."""
    return {
        "message": "Nyay-Setu AI Legal Assistant API",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/status")
async def get_status():
    """
    Checks if a PDF document has been successfully uploaded and processed,
    making the AI lawyer ready for queries.
    """
    if document_context["content"]:
        return {
            "document_loaded": True,
            "filename": document_context["filename"],
            "processed_at": document_context["processed_at"],
            # Provide a preview of the summary if available
            "summary_preview": document_context["summary"][:200] + "..." if document_context["summary"] and len(document_context["summary"]) > 200 else document_context["summary"],
            "document_length": len(document_context["content"])
        }
    else:
        return {
            "document_loaded": False,
            "message": "No document loaded. Please upload a PDF document first using /upload-document."
        }

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload and process a PDF document.
    It extracts text, generates a summary, and stores it in the global context.
    """
    global document_context
    
    # Validate file type to ensure it's a PDF
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only PDF files are supported. Please upload a PDF document."
        )
    
    try:
        # Read the PDF content and extract text
        pdf_content = await file.read()
        extracted_text = extract_pdf_text(pdf_content)
        
        if not extracted_text.strip(): # Check for truly empty or whitespace-only text
            raise HTTPException(
                status_code=400, 
                detail="No readable text found in the PDF. Please ensure the PDF contains text content or is not just images."
            )
        
        print(f"Processing document: {file.filename}")
        print(f"Extracted text length: {len(extracted_text)} characters")
        
        # Generate a summary of the extracted text using Gemini
        summary = await generate_summary(extracted_text)
        
        # Store the processed document content and summary in the global context
        from datetime import datetime
        document_context = {
            "content": extracted_text,
            "summary": summary,
            "filename": file.filename,
            "processed_at": datetime.now().isoformat() # Record timestamp
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
        # Re-raise FastAPIs HTTPExceptions directly
        raise
    except ValueError as ve: # Catch specific ValueErrors from helper functions
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process document: An unexpected error occurred. {str(e)}"
        )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: QueryRequest):
    """
    Endpoint for users to ask questions about the uploaded document using voice input.
    The user's speech is converted to text, processed by the AI, and the answer
    is returned as synthetic speech.
    """
    # Ensure a document is loaded before attempting to answer questions
    if not document_context["content"]:
        raise HTTPException(
            status_code=404,
            detail="No document loaded. Please upload a PDF document first using /upload-document."
        )
    
    try:
        # Step 1: Convert user's Base64 encoded audio to text using ElevenLabs STT
        print("Starting user audio to text conversion...")
        user_query_text = await elevenlabs_speech_to_text(request.audio_base64)
        
        if not user_query_text.strip(): # Check if transcription is empty or just whitespace
            print("ElevenLabs STT returned no readable text for the user query.")
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe your speech. Please ensure you spoke clearly or try again."
            )
        
        print(f"User Transcribed Query: '{user_query_text}' (Language requested: {request.language})")
        
        # Step 2: Query the loaded document with the transcribed text using Gemini
        print("Querying document with transcribed text using Gemini...")
        ai_response_text = await query_document(user_query_text, request.language)
        
        # Step 3: Convert AI's text response to Base64 encoded audio using ElevenLabs TTS
        print("Converting AI response text to audio...")
        ai_response_audio_base64 = await elevenlabs_text_to_speech(ai_response_text, request.language)
        
        print("Voice response successfully generated and returned.")
        return AskResponse(
            audio_response_base64=ai_response_audio_base64,
            original_query=user_query_text,
            response_language=request.language,
            document_filename=document_context["filename"]
        )
        
    except HTTPException:
        # Re-raise FastAPIs HTTPExceptions directly
        raise 
    except Exception as e:
        # Catch any other unexpected errors during the voice query process
        print(f"Error processing voice query in /ask endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process your voice question due to an internal error: {str(e)}"
        )

@app.delete("/clear-document")
async def clear_document():
    """
    Endpoint to clear the currently loaded document from memory.
    This effectively resets the AI lawyer's context.
    """
    global document_context
    document_context = {
        "content": None,
        "summary": None,
        "filename": None,
        "processed_at": None
    }
    print("Document context cleared.")
    return {"message": "Document cleared successfully. AI lawyer is now ready for a new document."}

# --- Global Exception Handlers for consistent error responses ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handles FastAPI's HTTPException and returns a structured JSON error."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handles any unhandled exceptions and returns a generic 500 internal server error.
    Prints the full traceback for debugging.
    """
    import traceback
    print(f"Unhandled exception: {exc}")
    traceback.print_exc() # Print full traceback to console
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected internal server error occurred", "details": str(exc)}
    )

# --- Main execution block for running the FastAPI application ---
if __name__ == "__main__":
    import uvicorn
    # 'reload=True' is great for development as it restarts the server on code changes.
    # For production, set reload=False and potentially use a more robust ASGI server
    # like Gunicorn with Uvicorn workers.
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)