# NaySetu: Inclusive GenAI Legal Assistant 🏛️🗣️🇮🇳

[![GitHub license](https://img.shields.io/github/license/YourGitHubUser/NaySetu?style=flat-square)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/YourGitHubUser/NaySetu?style=flat-square)](https://github.com/YourGitHubUser/NaySetu/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/YourGitHubUser/NaySetu?style=flat-square)](https://github.com/YourGitHubUser/NaySetu/network/members)
[![Lines of Code](https://img.shields.io/tokei/lines/github/YourGitHubUser/NaySetu?style=flat-square)](https://github.com/YourGitHubUser/NaySetu)
[![Built with FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Built with Next.js](https://img.shields.io/badge/Frontend-Next.js-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![Powered by Gemini AI](https://img.shields.io/badge/LLM-Gemini_AI-blueviolet?style=flat-square&logo=google)](https://ai.google.dev/gemini)

---

## 💡 Project Overview

**NaySetu** (derived from "Nyaya Setu" meaning "Bridge to Justice" in Hindi) is a groundbreaking **Inclusive GenAI Legal Assistant** designed to bridge India's significant legal literacy gap. It democratizes access to complex legal information by transforming dense legal documents into simple, interactive, and multilingual voice conversations.

Currently, over 88% of Indians don't speak fluent English, and legal documents are often filled with technical jargon. NaySetu empowers individuals – from gig workers and students to everyday citizens – to understand their rights and obligations by providing clear, concise legal explanations in their preferred native language.

This project was proudly developed as part of **BUILDathon 2025 by Material+**.

## ✨ Key Features

*   **Multilingual Voice Conversations:** Users can upload any legal document (contracts, rental agreements, job offers) and engage in natural voice conversations in their chosen Indian language (Hindi, Tamil, Telugu, Bengali, Kannada, etc.).
*   **Intelligent Legal Summarization & Q&A:** Leverages advanced GenAI capabilities to break down complex legal clauses and terms into easy-to-understand explanations.
*   **Responsible AI Guardrails:** Implements hallucination detection and legal accuracy filters to ensure reliable and trustworthy information.
*   **Intuitive User Interface (UI):** A clean, responsive, and voice-first design for a seamless user experience.
*   **Scalable Architecture:** Built to handle a large volume of queries, ready for nationwide impact.

## 🚀 How It Works (Simplified Flow)

1.  **Document Upload:** User uploads a legal document (e.g., PDF, image).
2.  **AI Pre-processing:** The document content is extracted, parsed, chunked, and prepared for the AI model.
3.  **User Voice Query:** User asks questions in their native language via voice input.
4.  **Intelligent Response Generation:** The AI processes the query against the document's context, generating a relevant, simplified, and accurate legal explanation.
5.  **Multilingual Voice Output:** The explanation is translated into the user's chosen language and converted into natural-sounding speech, then played back.

## 🛠️ Tech Stack

NaySetu is built with a robust and modern technology stack:

| Layer                        | Key Technologies                                   | Core Function & Benefit                                                                                                  |
| :--------------------------- | :------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **Core AI & Understanding**  | Google Gemini API, LangChain, Pinecone             | **AI-driven legal intelligence:** Powers accurate comprehension, summarization, and contextual, simplified explanations. |
| **User Experience (UI/UX)**  | React (Next.js), Tailwind CSS, GSAP                | **Engaging & Responsive Interface:** Delivers a fluid, intuitive, and modern user experience across devices.                |
| **Backend & APIs**           | FastAPI (Python)                                   | **High-Performance Backbone:** Provides scalable, secure, and efficient backend services for all operations.               |
| **Infrastructure & Ecosystem** | AWS, Bhashini (AI4Bharat)                          | **Reliable Cloud & National Integration:** Ensures robust global hosting and leverages India's national AI initiatives.   |

## 📈 Impact & Vision

NaySetu aims to have a profound impact by:

*   **Democratizing Legal Access:** Making legal awareness a right, not a privilege, for all Indians.
*   **Protecting Vulnerable Communities:** Empowering gig workers, domestic helpers, and students to understand contracts and avoid exploitation.
*   **Fostering Language Equity:** Eliminating linguistic barriers to justice by providing support in regional languages.
*   **Driving Digital India:** Aligning with national initiatives to promote digital and legal inclusion.

Our vision is to be the go-to AI legal buddy for every Indian, ensuring justice is accessible in every voice.

## 🤝 Contribution & Team

This project was developed by:

*   **Pranav Saluja** (Your Name) - *Full-Stack GenAI Developer*
*   **Nikhil Deshmukh** - *Role (e.g., AI/ML Engineer, Frontend Developer)*
*   **Bharath Sai** - *Role (e.g., Backend Developer, UI/UX Designer)*

Special thanks to our mentors from BUILDathon 2025 by Material+:
*   **Neha Sharma**
*   **Rudranil Sarkar**

## ▶️ Live Demo (Business Use Cases)

Due to the use of powerful paid APIs for optimal performance, a public live deployment is not currently available. However, we are eager to showcase NaySetu's capabilities for business use cases!

**To schedule a live demo, please connect with me on [LinkedIn](https://www.linkedin.com/in/pranavsaluja/)** (or via `pranavsaluja.work@gmail.com`).

## ⚙️ Local Development (For Contributors/Reviewers)

To set up NaySetu for local development:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourGitHubUser/NaySetu.git
    cd NaySetu
    ```
2.  **Backend Setup:**
    ```bash
    cd backend/
    pip install -r requirements.txt
    # Create a .env file with your API keys (e.g., ELEVENLABS_API_KEY, GOOGLE_GEMINI_API_KEY, AWS_CREDENTIALS)
    # python -m uvicorn main:app --reload
    ```
    *Refer to `backend/README.md` for detailed instructions.*
3.  **Frontend Setup:**
    ```bash
    cd frontend/
    npm install
    # Create a .env.local file with frontend-specific environment variables
    # npm run dev
    ```
    *Refer to `frontend/README.md` for detailed instructions.*

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
