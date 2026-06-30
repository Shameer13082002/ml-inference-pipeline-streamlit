import os
import re
import tempfile
import uuid
import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader
from docx import Document

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from groq import Groq

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Resume Matcher + ATS Builder",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# SESSION
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "generated_resume" not in st.session_state:
    st.session_state.generated_resume = ""

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("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")
PINECONE_API_KEY = os.getenv("pcsk_6V5R9n_Hpj1WCfo2setvxnmYHqAeV9pFysTh6FyJTzgsCtgpRUhEGEBgrdL7vZMpfWAyqi")

# =====================================================
# API KEY GUARD — shows clear error on Hugging Face
# if secrets are not configured
# =====================================================

if not GROQ_API_KEY or not PINECONE_API_KEY:
    st.error(
        "⚠️ **API keys are missing!**\n\n"
        "Please go to your Hugging Face Space → **Settings** → "
        "**Variables and secrets** and add:\n"
        "- `GROQ_API_KEY`\n"
        "- `PINECONE_API_KEY`"
    )
    st.stop()

# =====================================================
# INIT CLIENTS
# =====================================================

client = Groq(api_key=GROQ_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("resume-matcher")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

.stApp{
    background:linear-gradient(135deg,#5B1FA8 0%,#7928CA 35%,#C0289C 70%,#FF0080 100%);
    background-attachment:fixed;
    font-family:'DM Sans',sans-serif;
    color:white;
}

#MainMenu, footer, header{
    visibility:hidden;
}

.block-container{
    padding-top:2rem;
    max-width:1400px;
}

.main-title{
    font-size:60px;
    font-family:'Syne',sans-serif;
    font-weight:800;
    text-align:center;
    background:linear-gradient(90deg,#00F5A0,#00D9F5,#F15BB5,#fee440);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:white;
    opacity:0.8;
    margin-bottom:40px;
    font-size:18px;
}

.card{
    background:rgba(255,255,255,0.10);
    border-radius:24px;
    padding:25px;
    border:1px solid rgba(255,255,255,0.2);
    backdrop-filter:blur(10px);
    box-shadow:0 8px 32px rgba(0,0,0,0.25);
}

textarea{
    background:rgba(0,0,0,0.35)!important;
    color:white!important;
    border:2px solid #00F5A0!important;
    border-radius:18px!important;
}

textarea::placeholder{
    color:white!important;
    opacity:0.5;
}

.stFileUploader{
    background:rgba(0,0,0,0.35)!important;
    border:2px dashed #00F5A0!important;
    border-radius:20px!important;
    padding:18px!important;
}

.stButton>button{
    width:100%;
    border:none;
    border-radius:16px;
    padding:15px;
    font-size:18px;
    font-weight:700;
    color:white;
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    transition:0.3s;
}

.stButton>button:hover{
    transform:translateY(-2px);
}

.score-box{
    background:linear-gradient(135deg,#00F260,#0575E6);
    border-radius:22px;
    padding:30px;
    text-align:center;
    color:white;
    box-shadow:0 8px 30px rgba(0,0,0,0.25);
}

.score-title{
    font-size:16px;
    opacity:0.8;
}

.score-value{
    font-size:58px;
    font-weight:800;
}

.copy-box{
    position:relative;
    background:rgba(0,0,0,0.35);
    border:2px solid #00F5A0;
    border-radius:20px;
    padding:25px;
    margin-top:15px;
    color:white;
    line-height:1.8;
    max-height:700px;
    overflow:auto;
}

.copy-btn{
    position:absolute;
    right:15px;
    bottom:15px;
    border:none;
    border-radius:12px;
    padding:10px 18px;
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    font-weight:700;
    cursor:pointer;
}

.chat-user{
    background:rgba(0,198,255,0.2);
    border:1px solid #00c6ff;
    padding:16px;
    border-radius:16px;
    margin-bottom:12px;
}

.chat-ai{
    background:rgba(0,245,160,0.15);
    border:1px solid #00F5A0;
    padding:16px;
    border-radius:16px;
    margin-bottom:16px;
}

</style>

<script>
function copyText(id){
    const text=document.getElementById(id).innerText;
    navigator.clipboard.writeText(text);
}
</script>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🚀 AI Resume Matcher + ATS Builder</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">ATS Optimization • Semantic Matching • AI Resume Builder • Career Chatbot</div>',
    unsafe_allow_html=True
)

# =====================================================
# COMPANY ATS FORMATS
# =====================================================

COMPANY_STYLES = {

    "Cognizant": """
    Cognizant resume style:
    - ATS friendly
    - enterprise project heavy
    - agile terminology
    - cloud technologies
    - quantified achievements
    - strong technical skills section
    """,

    "TCS": """
    TCS resume style:
    - clean traditional ATS format
    - certifications highlighted
    - SDLC keywords
    - enterprise delivery
    """,

    "Infosys": """
    Infosys style:
    - leadership
    - automation
    - AI/Cloud projects
    - modern architecture
    """,

    "Google": """
    Google style:
    - X-Y-Z impact format
    - scalable systems
    - metrics driven
    - leadership
    """,

    "Amazon": """
    Amazon style:
    - leadership principles
    - ownership
    - customer obsession
    - scale
    """
}

# =====================================================
# HELPERS
# =====================================================

def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)

def extract_docx_text(docx_file):
    doc = Document(docx_file)
    return "\n".join(p.text for p in doc.paragraphs)

def clean_text(text):
    for ch in ["**", "#", "```", "__"]:
        text = text.replace(ch, "")
    return text.strip()

def render_copy_box(content):

    uid = str(uuid.uuid4()).replace("-", "")

    safe = (
        content
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    st.markdown(f"""
    <div class="copy-box">
        <div id="{uid}">{safe}</div>

        <button class="copy-btn"
        onclick="copyText('{uid}')">
        📋 Copy
        </button>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# ATS SCORE
# =====================================================

def calculate_ats_score(resume_text, jd_text):

    jd_words = set(
        re.findall(r'\b[a-zA-Z]{3,}\b', jd_text.lower())
    )

    res_words = set(
        re.findall(r'\b[a-zA-Z]{3,}\b', resume_text.lower())
    )

    matched = jd_words & res_words
    missing = jd_words - res_words

    score = round(
        (len(matched) / len(jd_words)) * 100,
        1
    ) if jd_words else 0

    return score, list(matched)[:30], list(missing)[:30]

# =====================================================
# RESUME GENERATION
# =====================================================

def generate_ats_resume(
    resume_text,
    job_description,
    company_name,
    user_instruction=""
):

    company_style = COMPANY_STYLES.get(
        company_name,
        "Modern ATS friendly resume"
    )

    prompt = f"""
You are an elite ATS resume writer.

Generate a PREMIUM ATS FRIENDLY resume.

TARGET COMPANY:
{company_name}

COMPANY STYLE:
{company_style}

USER INSTRUCTION:
{user_instruction}

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME:
{resume_text}

RULES:
- NO markdown
- NO stars
- NO ##
- Modern resume
- ATS parseable
- Strong metrics
- Quantified achievements
- Beautiful formatting
- Recruiter friendly
- Top tier resume website quality
- Include all JD keywords naturally
- ATS score should be above 95%
- Add impactful projects
- Use strong action verbs

FORMAT:

NAME
CONTACT

PROFESSIONAL SUMMARY

CORE SKILLS

EXPERIENCE

PROJECTS

EDUCATION

CERTIFICATIONS

TECHNICAL SKILLS

Generate complete resume now.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":"system",
                "content":"You are a world-class ATS resume expert."
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.4,
        max_tokens=3500
    )

    return clean_text(
        response.choices[0].message.content
    )

# =====================================================
# PDF CREATION
# =====================================================

def create_pdf(content):

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    path = temp.name
    temp.close()

    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        leftMargin=0.6*inch,
        rightMargin=0.6*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.HexColor("#111827"),
        alignment=TA_LEFT,
        spaceAfter=10
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=16,
        textColor=colors.HexColor("#374151")
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#4f46e5"),
        spaceBefore=14,
        spaceAfter=6
    )

    story = []

    lines = content.split("\n")

    first = True

    for line in lines:

        line = line.strip()

        if not line:
            story.append(Spacer(1, 5))
            continue

        if first:
            story.append(Paragraph(line, title_style))
            first = False
            continue

        if line.isupper() and len(line) < 40:
            story.append(HRFlowable(
                width="100%",
                thickness=1,
                color=colors.HexColor("#4f46e5")
            ))

            story.append(
                Paragraph(line, section_style)
            )

            continue

        story.append(
            Paragraph(line, body_style)
        )

    doc.build(story)

    return path

# =====================================================
# INPUT UI
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📄 Upload Resume")

    resume = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste JD",
        height=250
    )

    company_name = st.selectbox(
        "🏢 Target Company",
        [
            "General",
            "Cognizant",
            "TCS",
            "Infosys",
            "Google",
            "Amazon"
        ]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# MAIN ANALYSIS
# =====================================================

if st.button("🔍 Analyze & Generate ATS Resume"):

    if resume and job_description:

        with st.spinner("Analyzing resume..."):

            # EXTRACT

            if resume.name.endswith(".pdf"):
                resume_text = extract_pdf_text(resume)
            else:
                resume_text = extract_docx_text(resume)

            # PINECONE

            vector = embeddings.embed_query(
                resume_text
            )

            index.upsert(
                vectors=[
                    {
                        "id": "candidate1",
                        "values": vector,
                        "metadata": {
                            "text": resume_text
                        }
                    }
                ]
            )

            jd_vector = embeddings.embed_query(
                job_description
            )

            result = index.query(
                vector=jd_vector,
                top_k=1
            )

            semantic_score = round(
                result["matches"][0]["score"] * 100,
                1
            )

            ats_score, matched, missing = calculate_ats_score(
                resume_text,
                job_description
            )

            # GENERATE ATS RESUME

            optimized_resume = generate_ats_resume(
                resume_text,
                job_description,
                company_name
            )

            st.session_state.generated_resume = optimized_resume

        # =====================================================
        # SCORES
        # =====================================================

        st.markdown("---")

        s1, s2 = st.columns(2)

        with s1:

            st.markdown(f"""
            <div class="score-box">
                <div class="score-title">
                🎯 Semantic Match
                </div>

                <div class="score-value">
                {semantic_score}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s2:

            st.markdown(f"""
            <div class="score-box">
                <div class="score-title">
                📄 ATS Score
                </div>

                <div class="score-value">
                {ats_score}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =====================================================
        # MATCHED
        # =====================================================

        st.markdown("## ✅ Matched Keywords")

        render_copy_box(
            " • ".join(matched)
        )

        # =====================================================
        # MISSING
        # =====================================================

        st.markdown("## ❌ Missing Keywords")

        render_copy_box(
            " • ".join(missing)
        )

        # =====================================================
        # GENERATED RESUME
        # =====================================================

        st.markdown("## 🏆 ATS Optimized Resume")

        render_copy_box(
            optimized_resume
        )

        # =====================================================
        # PDF
        # =====================================================

        pdf_path = create_pdf(
            optimized_resume
        )

        with open(pdf_path, "rb") as f:

            st.download_button(
                label="⬇️ Download ATS Resume PDF",
                data=f,
                file_name="ATS_Resume.pdf",
                mime="application/pdf"
            )

        st.success(
            "✅ ATS optimized resume generated successfully!"
        )

    else:

        st.warning(
            "⚠️ Upload resume and paste JD."
        )

# =====================================================
# CHATBOT
# =====================================================

st.markdown("---")

st.markdown("## 🤖 AI Resume Career Assistant")

chat_query = st.text_input(
    "Ask anything like: 'Give Cognizant resume format' or 'How to improve ATS score?'"
)

if st.button("💬 Ask AI"):

    if chat_query:

        current_resume = st.session_state.generated_resume

        prompt = f"""
You are a world-class ATS expert and recruiter.

USER QUESTION:
{chat_query}

CURRENT RESUME:
{current_resume}

INSTRUCTIONS:
- Provide ATS-friendly advice
- Give Cognizant/TCS/Infosys/Google style guidance
- Suggest recruiter-friendly improvements
- Explain ATS optimization
- Make responses professional
- Suggest modern resume structures
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an elite recruiter and ATS coach."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1800
        )

        ai_reply = response.choices[0].message.content

        st.session_state.chat_history.append(
            ("You", chat_query)
        )

        st.session_state.chat_history.append(
            ("AI", ai_reply)
        )

# =====================================================
# CHAT HISTORY
# =====================================================

if st.session_state.chat_history:

    st.markdown("## 💬 Chat History")

    for role, msg in st.session_state.chat_history:

        if role == "You":

            st.markdown(f"""
            <div class="chat-user">
            <b>🧑 You:</b><br><br>
            {msg}
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="chat-ai">
            <b>🤖 AI:</b><br><br>
            {msg}
            </div>
            """, unsafe_allow_html=True)
