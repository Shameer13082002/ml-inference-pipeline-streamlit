# 🚀 AI Resume Matcher + ATS Builder

An AI-powered resume optimization tool that analyzes how well a resume matches a job description, identifies missing keywords, and generates a recruiter-ready, ATS-optimized resume — all through a Streamlit web app.

## What It Does

- **Resume Parsing** — Extracts text from uploaded PDF or DOCX resumes
- **Semantic Matching** — Embeds resume and job description using a sentence-transformer model and compares them via vector similarity search in Pinecone
- **ATS Keyword Scoring** — Computes keyword overlap between resume and job description, surfacing matched and missing terms
- **AI Resume Generation** — Uses Llama 3.3 70B (via Groq) to rewrite the resume in a company-specific ATS-friendly format (supports style profiles for companies like Google, Amazon, TCS, Infosys, Cognizant)
- **PDF Export** — Generates a clean, formatted PDF of the optimized resume using ReportLab
- **Career Chatbot** — A built-in assistant for follow-up questions on resume structure, ATS strategy, and company-specific formatting advice

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | Pinecone |
| Document Parsing | pypdf, python-docx |
| PDF Generation | ReportLab |

## How It Works

1. User uploads a resume (PDF/DOCX) and pastes a job description
2. The app extracts resume text and generates embeddings
3. Resume and JD embeddings are compared via Pinecone for a semantic match score
4. A keyword-overlap algorithm computes an ATS score and lists matched/missing terms
5. The LLM generates a rewritten resume tailored to the target company's style and the job description's keywords
6. The optimized resume is rendered in-app and downloadable as a formatted PDF

## Setup

### Prerequisites
- Python 3.9+
- A [Groq API key](https://console.groq.com)
- A [Pinecone API key](https://www.pinecone.io) with an index named `resume-matcher`

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:
