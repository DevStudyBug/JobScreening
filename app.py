import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
from dotenv import load_dotenv
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text() or ""  # Handle None case
        text += extracted_text
    return text.strip()

# Function to get ATS analysis response
def get_gemini_response(input_text, jd):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Act as a highly experienced ATS (Applicant Tracking System) specializing in software engineering,
    data science, data analytics, and big data roles. Evaluate the resume against the provided job description.
    Consider that the job market is competitive and provide insights for improvement.

    Resume: {input_text}
    Job Description: {jd}

    You must respond with ONLY a valid JSON object and nothing else. No markdown, no extra text.
    The JSON should have this exact structure:
    {{
      "JD Match": "X%",
      "MissingKeywords": ["keyword1", "keyword2", "..."],
      "MatchedKeywords": ["keyword1", "keyword2", "..."],
      "ProfileSummary": "detailed summary here",
      "StrengthAreas": ["strength1", "strength2", "..."],
      "ImprovementAreas": ["area1", "area2", "..."],
      "RecommendedSkills": ["skill1", "skill2", "..."]
    }}
    """
    
    response = model.generate_content(prompt)
    
    try:
        # Clean the response text by removing any non-JSON content
        response_text = response.text.strip()
        # If response is wrapped in markdown code blocks, remove them
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()
            
        return json.loads(response_text)  # Convert AI response to JSON
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            return {
                "error": f"Failed to process the response: {str(e)}",
                "raw_response": response.text
            }
        else:
            return {"error": "Failed to process the response. Please try again."}

# Function to summarize job description
def summarize_job_description(jd):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Analyze the following job description and extract the key elements:
    
    {jd}
    
    Provide a summary in JSON format with the following structure:
    {{
      "JobTitle": "title of the position",
      "RequiredSkills": ["skill1", "skill2", "..."],
      "RequiredExperience": "X years in...",
      "RequiredQualifications": ["qualification1", "qualification2", "..."],
      "JobResponsibilities": ["responsibility1", "responsibility2", "..."],
      "CompanyOverview": "brief company description",
      "KeywordsSummary": "summary of the most important keywords"
    }}
    """
    
    response = model.generate_content(prompt)
    
    try:
        response_text = response.text.strip()
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()
            
        return json.loads(response_text)
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            return {
                "error": f"Failed to summarize job description: {str(e)}",
                "raw_response": response.text
            }
        else:
            return {"error": "Failed to summarize job description. Please try again."}

# Function to extract candidate data from CV
def extract_cv_data(resume_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Extract the following information from this resume:
    
    {resume_text}
    
    Return the data in JSON format with this structure:
    {{
      "CandidateName": "Full Name",
      "ContactInfo": {{
        "Email": "email address",
        "Phone": "phone number"
      }},
      "Education": [
        {{
          "Degree": "degree name",
          "Institution": "institution name",
          "GraduationYear": "year"
        }}
      ],
      "WorkExperience": [
        {{
          "Title": "job title",
          "Company": "company name",
          "Duration": "start date - end date",
          "Responsibilities": ["responsibility1", "responsibility2"]
        }}
      ],
      "Skills": ["skill1", "skill2", "..."],
      "Certifications": ["certification1", "certification2", "..."]
    }}
    """
    
    response = model.generate_content(prompt)
    
    try:
        response_text = response.text.strip()
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()
            
        return json.loads(response_text)
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            return {
                "error": f"Failed to extract CV data: {str(e)}",
                "raw_response": response.text
            }
        else:
            return {"error": "Failed to extract CV data. Please try again."}

# Function to generate interview invite email
def generate_interview_email(candidate_data, jd_summary):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate interview dates (next 5 business days)
    today = datetime.now()
    interview_dates = []
    days_added = 0
    while len(interview_dates) < 3:
        next_day = today + timedelta(days=days_added+2)  # Start from day after tomorrow
        if next_day.weekday() < 5:  # Monday to Friday
            interview_dates.append(next_day.strftime("%A, %B %d, %Y"))
        days_added += 1
    
    prompt = f"""
    Create a professional interview invitation email for {candidate_data.get('CandidateName', 'the candidate')} 
    for the position of {jd_summary.get('JobTitle', 'the open position')}.
    
    Include these details:
    - The candidate's name: {candidate_data.get('CandidateName', 'Candidate')}
    - The position they applied for: {jd_summary.get('JobTitle', 'Open Position')}
    - Available interview dates: {', '.join(interview_dates)}
    - Available times: 10:00 AM, 2:00 PM, 4:00 PM
    - Interview format: Initial 30-minute video interview
    - Brief overview of the interview process
    - Request to confirm preferred date and time
    
    The tone should be professional yet welcoming, and the email should be concise.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Page configuration
st.set_page_config(
    page_title="RecruitEase | ATS Resume Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --main-blue: #1E88E5;
        --light-blue: #BBD9F2;
        --dark-blue: #0D47A1;
        --accent-blue: #64B5F6;
    }
    
    /* Text and Headers */
    h1, h2, h3 {
        color: var(--dark-blue);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background-color: var(--main-blue);
        padding: 1.5rem;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Input fields */
    .stTextInput, .stTextArea {
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--main-blue);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: var(--dark-blue);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Result sections */
    .result-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid var(--main-blue);
    }
    
    /* Keyword pills */
    .keyword-pill {
        display: inline-block;
        padding: 5px 12px;
        margin: 5px;
        background-color: var(--light-blue);
        color: var(--dark-blue);
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Matched keyword pills */
    .matched-keyword {
        background-color: #c8e6c9;
        color: #2e7d32;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--dark-blue);
        margin-bottom: 15px;
        border-bottom: 2px solid var(--accent-blue);
        padding-bottom: 8px;
    }
    
    /* Icon styling */
    .icon {
        vertical-align: middle;
        margin-right: 8px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #dee2e6;
    }
    
    /* File uploader */
    .css-1qrvfrg {
        background-color: var(--light-blue);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Progress bar */
    .match-progress {
        height: 20px;
        border-radius: 5px;
        background-color: #e9ecef;
        margin-top: 10px;
        margin-bottom: 20px;
        overflow: hidden;
    }
    
    .match-progress-bar {
        height: 100%;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 20px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        padding: 10px 16px;
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--main-blue) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'jd_summary' not in st.session_state:
    st.session_state['jd_summary'] = None
if 'cv_data' not in st.session_state:
    st.session_state['cv_data'] = None
if 'shortlisted' not in st.session_state:
    st.session_state['shortlisted'] = False
if 'interview_email' not in st.session_state:
    st.session_state['interview_email'] = None

# Sidebar content
with st.sidebar:
    st.image("recuLogo.png", width=80)
    st.markdown("## Settings & Tips")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    st.session_state['debug_mode'] = st.checkbox("Enable Debug Mode", st.session_state['debug_mode'])
    shortlist_threshold = st.slider("Shortlisting Threshold", 50, 95, 70, 5, 
                                    help="Candidates with match scores above this percentage will be shortlisted")
    
    # Resume tips
    st.markdown("### 📝 Resume Tips")
    st.markdown("""
    - Tailor your resume to the job description
    - Quantify your achievements with numbers
    - Use action verbs (Led, Developed, Implemented)
    - Include relevant keywords from the job posting
    - Keep your resume concise and well-formatted
    """)
    
    # About section
    st.markdown("### ℹ️ About RecruitEase")
    st.markdown("""
    RecruitEase helps you optimize your resume for ATS systems using advanced AI analysis. Upload your resume, paste the job description, and get instant feedback.
    """)
    
    # Footer
    st.markdown("<div class='footer'>© 2025 RecruitEase | v1.0</div>", unsafe_allow_html=True)

# Main content
st.markdown("<h1 class='main-header'>📝 RecruitEase: ATS Resume Analyzer</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("""
### Welcome to RecruitEase! 
Our AI-powered tool analyzes your resume against job descriptions and provides actionable feedback to increase your chances of getting past ATS systems and landing that interview.
""")
st.markdown("</div>", unsafe_allow_html=True)

# Create tabs for different functionalities
tabs = st.tabs(["📋 ATS Analysis", "📊 JD Summarizer", "👥 Candidate Management", "📅 Interview Scheduler"])

# Tab 1: ATS Analysis
with tabs[0]:
    st.markdown("<h2>📋 Upload & Analyze</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>📄 Job Description</p>", unsafe_allow_html=True)
        jd = st.text_area("Paste the job description here", height=250, placeholder="Paste the complete job description here...")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>📎 Resume Upload</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your resume (PDF format)", type="pdf", help="Please upload a PDF file only")
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
            st.info("Your resume will be analyzed against the job description. Make sure both are complete for the best results.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze My Resume")

    # Processing and Results
    if analyze_button:
        if uploaded_file is not None and jd.strip():
            with st.spinner("⏳ Analyzing your resume against the job description..."):
                # Extract text from resume
                resume_text = input_pdf_text(uploaded_file)
                
                # Get JD summary
                st.session_state['jd_summary'] = summarize_job_description(jd)
                
                # Extract CV data
                st.session_state['cv_data'] = extract_cv_data(resume_text)
                
                # Get ATS analysis from AI
                response = get_gemini_response(resume_text, jd)
                st.session_state['result'] = response
                st.session_state['analysis_complete'] = True
                
                # Check if candidate should be shortlisted
                if "JD Match" in response:
                    match_percentage = int(response['JD Match'].strip('%'))
                    st.session_state['shortlisted'] = match_percentage >= shortlist_threshold
                    
                    # Generate interview email if shortlisted
                    if st.session_state['shortlisted']:
                        st.session_state['interview_email'] = generate_interview_email(
                            st.session_state['cv_data'], 
                            st.session_state['jd_summary']
                        )
        else:
            st.error("Please upload your resume and enter a job description to proceed with the analysis.")

    # Display results if analysis is complete
    if st.session_state['analysis_complete'] and st.session_state['result']:
        response = st.session_state['result']
        
        if "error" in response:
            st.error(response["error"])
            
            # Show raw response in debug mode
            if st.session_state['debug_mode'] and "raw_response" in response:
                st.subheader("Raw AI Response")
                st.code(response["raw_response"])
                
                # Provide troubleshooting help
                st.subheader("Troubleshooting Tips")
                st.markdown("""
                - The AI response is not in valid JSON format
                - Try simplifying your resume or job description
                - Check if your API key is valid and has necessary permissions
                - Try again later as the service might be experiencing high traffic
                """)
        else:
            st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
            
            # Results overview
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                # Match percentage with progress bar
                match_percentage = int(response['JD Match'].strip('%'))
                st.markdown(f"<h2 style='text-align: center;'>Match Score</h2>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; color: {'#28a745' if match_percentage >= 70 else '#ffc107' if match_percentage >= 40 else '#dc3545'};'>{response['JD Match']}</h1>", unsafe_allow_html=True)
                
                # Progress bar
                progress_color = '#28a745' if match_percentage >= 70 else '#ffc107' if match_percentage >= 40 else '#dc3545'
                st.markdown(f"""
                <div class='match-progress'>
                    <div class='match-progress-bar' style='width: {match_percentage}%; background-color: {progress_color};'>
                        {match_percentage}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick assessment based on match percentage
                if match_percentage >= 70:
                    st.success("Great match! Your resume is well-aligned with this position.")
                elif match_percentage >= 40:
                    st.warning("Moderate match. Consider enhancing your resume with the suggested improvements.")
                else:
                    st.error("Low match. Significant improvements needed to increase your chances.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Strengths and improvement areas
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>💪 Your Strengths</p>", unsafe_allow_html=True)
                for strength in response.get('StrengthAreas', []):
                    st.markdown(f"✅ {strength}")
                    
                st.markdown("<p class='section-header' style='margin-top: 20px;'>🔧 Areas for Improvement</p>", unsafe_allow_html=True)
                for area in response.get('ImprovementAreas', []):
                    st.markdown(f"⚠️ {area}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Keywords section
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<p class='section-header'>❌ Missing Keywords</p>", unsafe_allow_html=True)
                if response.get('MissingKeywords'):
                    for keyword in response.get('MissingKeywords', []):
                        st.markdown(f"<span class='keyword-pill'>🔍 {keyword}</span>", unsafe_allow_html=True)
                else:
                    st.info("No critical missing keywords detected!")
                    
            with col2:
                st.markdown("<p class='section-header'>✅ Matched Keywords</p>", unsafe_allow_html=True)
                if response.get('MatchedKeywords'):
                    for keyword in response.get('MatchedKeywords', []):
                        st.markdown(f"<span class='keyword-pill matched-keyword'>✓ {keyword}</span>", unsafe_allow_html=True)
                else:
                    st.info("No matched keywords found.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Profile summary
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>📋 Profile Summary</p>", unsafe_allow_html=True)
            st.markdown(f"{response.get('ProfileSummary', 'No profile summary available.')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommended skills
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>🚀 Recommended Skills</p>", unsafe_allow_html=True)
            st.markdown("Consider adding these skills to your resume to increase your match rate:")
            
            skill_cols = st.columns(3)
            for i, skill in enumerate(response.get('RecommendedSkills', [])):
                with skill_cols[i % 3]:
                    st.markdown(f"<div style='background-color: #e9f5fe; padding: 10px; border-radius: 8px; margin: 5px 0;'><b>🔹 {skill}</b></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Action plan
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>📝 Your Action Plan</p>", unsafe_allow_html=True)
            st.markdown("""
            1. **Add Missing Keywords**: Include the missing keywords highlighted above where relevant in your resume.
            2. **Quantify Achievements**: Add specific metrics and numbers to demonstrate your impact.
            3. **Optimize Format**: Ensure your resume is ATS-friendly with a clean, simple format.
            4. **Tailor Your Summary**: Customize your professional summary to match this specific role.
            5. **Add Recommended Skills**: Incorporate relevant skills you possess but haven't mentioned.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # In debug mode, show the raw response
            if st.session_state['debug_mode']:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>🛠️ Debug Information</p>", unsafe_allow_html=True)
                st.json(response)
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Reset button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📝 Analyze Another Resume"):
                    st.session_state['analysis_complete'] = False
                    st.session_state['result'] = None

# Tab 2: JD Summarizer
with tabs[1]:
    st.markdown("<h2>📊 Job Description Summarizer</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This tool extracts key information from job descriptions to help you understand the requirements better. 
    Paste a job description and get a structured breakdown of skills, qualifications, and responsibilities.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input area
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📄 Job Description Input</p>", unsafe_allow_html=True)
    jd_to_summarize = st.text_area("Paste the job description here", height=200, 
                                    placeholder="Paste the complete job description here to analyze...",
                                    key="jd_summarizer_input")
    
    # Summarize button
    summarize_button = st.button("📊 Summarize Job Description")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process and display results
    if summarize_button and jd_to_summarize.strip():
        with st.spinner("⏳ Analyzing job description..."):
            jd_summary = summarize_job_description(jd_to_summarize)
            st.session_state['jd_summary'] = jd_summary
            
        if "error" in jd_summary:
            st.error(jd_summary["error"])
        else:
            # Display JD summary
            st.markdown("<h3>📑 Job Description Summary</h3>", unsafe_allow_html=True)
            
            # Job title and company overview
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h2>{jd_summary.get('JobTitle', 'Position')}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>🏢 Company Overview</p>", unsafe_allow_html=True)
            st.markdown(f"{jd_summary.get('CompanyOverview', 'No company information available.')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Skills and qualifications
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>🔧 Required Skills</p>", unsafe_allow_html=True)
                if jd_summary.get('RequiredSkills'):
                    for skill in jd_summary.get('RequiredSkills', []):
                        st.markdown(f"• {skill}")
                else:
                    st.info("No specific skills listed.")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>🎓 Required Qualifications</p>", unsafe_allow_html=True)
                st.markdown(f"**Experience:** {jd_summary.get('RequiredExperience', 'Not specified')}")
                
                if jd_summary.get('RequiredQualifications'):
                    for qual in jd_summary.get('RequiredQualifications', []):
                        st.markdown(f"• {qual}")
                else:
                    st.info("No specific qualifications listed.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Job responsibilities
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>📋 Job Responsibilities</p>", unsafe_allow_html=True)
            if jd_summary.get('JobResponsibilities'):
                for resp in jd_summary.get('JobResponsibilities', []):
                    st.markdown(f"• {resp}")
            else:
                st.info("No specific responsibilities listed.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Keyword summary
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>🔑 Key Takeaways</p>", unsafe_allow_html=True)
            st.markdown(f"{jd_summary.get('KeywordsSummary', 'No summary available.')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Debug information
            if st.session_state['debug_mode']:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>🛠️ Debug Information</p>", unsafe_allow_html=True)
                st.json(jd_summary)
                st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Candidate Management (Continued)
with tabs[2]:
    # Candidate upload section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📄 Upload Candidate Resume</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a resume (PDF, DOCX)", type=["pdf", "docx"], key="resume_uploader")
    candidate_name = st.text_input("Candidate Name", key="candidate_name")
    candidate_email = st.text_input("Candidate Email", key="candidate_email")
    
    # Job description selection for comparison
    st.markdown("<p class='section-header'>🔍 Select Job Description for Comparison</p>", unsafe_allow_html=True)
    
    # Use the previously analyzed JD if available, or let user select from saved JDs
    use_current_jd = False
    if 'jd_summary' in st.session_state and st.session_state['jd_summary']:
        use_current_jd = st.checkbox("Use currently analyzed job description", value=True)
    
    if use_current_jd and 'jd_summary' in st.session_state:
        selected_jd = st.session_state['jd_summary']
        st.info(f"Using current JD: {selected_jd.get('JobTitle', 'Unnamed Position')}")
    else:
        # Demo mode - would normally load from database
        saved_jds = {
            "Select a job": None,
            "Data Scientist": {"JobTitle": "Data Scientist", "RequiredSkills": ["Python", "Machine Learning", "SQL", "Data Visualization"]},
            "Software Engineer": {"JobTitle": "Software Engineer", "RequiredSkills": ["Java", "JavaScript", "Cloud Services", "Git"]}
        }
        selected_jd_name = st.selectbox("Select a saved job description", list(saved_jds.keys()))
        selected_jd = saved_jds[selected_jd_name]
    
    analyze_button = st.button("🔍 Analyze Candidate")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process resume and display results
    if analyze_button and uploaded_file and selected_jd:
        with st.spinner("⏳ Analyzing resume and comparing to job description..."):
            # Extract resume data
            resume_data = extract_resume_data(uploaded_file)
            
            # Calculate match score
            match_score, skill_matches, missing_skills = compare_resume_to_jd(resume_data, selected_jd)
            
            # Store in session state
            if 'candidates' not in st.session_state:
                st.session_state['candidates'] = []
                
            # Add to candidates list
            candidate_info = {
                "name": candidate_name,
                "email": candidate_email,
                "resume_data": resume_data,
                "match_score": match_score,
                "skill_matches": skill_matches,
                "missing_skills": missing_skills,
                "jd_title": selected_jd.get('JobTitle', 'Unnamed Position'),
                "shortlisted": match_score >= 80  # Auto-shortlist if score is 80% or higher
            }
            
            st.session_state['candidates'].append(candidate_info)
            st.session_state['current_candidate'] = candidate_info
            
        # Display resume analysis
        st.markdown("<h3>📋 Resume Analysis</h3>", unsafe_allow_html=True)
        
        # Candidate profile
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{candidate_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Email:</strong> {candidate_email}</p>", unsafe_allow_html=True)
        
        # Display match score with color coding
        score_color = "green" if match_score >= 80 else "orange" if match_score >= 60 else "red"
        st.markdown(f"<p><strong>Match Score:</strong> <span style='color:{score_color};font-weight:bold;'>{match_score}%</span></p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p class='section-header'>✅ Matching Skills</p>", unsafe_allow_html=True)
            if skill_matches:
                for skill in skill_matches:
                    st.markdown(f"• {skill}")
            else:
                st.info("No matching skills found.")
                
        with col2:
            st.markdown("<p class='section-header'>❌ Missing Skills</p>", unsafe_allow_html=True)
            if missing_skills:
                for skill in missing_skills:
                    st.markdown(f"• {skill}")
            else:
                st.success("No missing key skills.")
        
        # Candidate summary
        st.markdown("<p class='section-header'>📝 Candidate Summary</p>", unsafe_allow_html=True)
        st.markdown(resume_data.get('summary', 'No summary available.'))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Resume details
        exp_col, edu_col = st.columns(2)
        
        with exp_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>💼 Experience</p>", unsafe_allow_html=True)
            if resume_data.get('experience'):
                for exp in resume_data.get('experience', []):
                    st.markdown(f"**{exp.get('title')}** at {exp.get('company')}")
                    st.markdown(f"{exp.get('date')}")
                    st.markdown(f"{exp.get('description')}")
                    st.markdown("---")
            else:
                st.info("No experience data extracted.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with edu_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>🎓 Education</p>", unsafe_allow_html=True)
            if resume_data.get('education'):
                for edu in resume_data.get('education', []):
                    st.markdown(f"**{edu.get('degree')}** from {edu.get('institution')}")
                    st.markdown(f"{edu.get('date')}")
                    st.markdown("---")
            else:
                st.info("No education data extracted.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Skills section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>🔧 Skills</p>", unsafe_allow_html=True)
        if resume_data.get('skills'):
            skills_list = resume_data.get('skills', [])
            # Display skills in multiple columns
            cols = st.columns(3)
            for i, skill in enumerate(skills_list):
                cols[i % 3].markdown(f"• {skill}")
        else:
            st.info("No skills data extracted.")
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 4: Shortlisting
with tabs[3]:
    st.markdown("<h2>🏆 Candidate Shortlisting</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This section shows all analyzed candidates and helps you shortlist them based on their match score.
    You can review candidates, adjust shortlisting status, and proceed to scheduling interviews.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Candidates table
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>👥 Candidate Pool</p>", unsafe_allow_html=True)
    
    if 'candidates' in st.session_state and st.session_state['candidates']:
        candidates_df = pd.DataFrame([
            {
                "Name": c["name"],
                "Email": c["email"],
                "Job Position": c["jd_title"],
                "Match Score": f"{c['match_score']}%",
                "Status": "Shortlisted" if c["shortlisted"] else "Not Shortlisted",
                "Index": i
            } for i, c in enumerate(st.session_state['candidates'])
        ])
        
        # Color coding based on match score
        def highlight_score(val):
            score = int(val.replace('%', ''))
            if score >= 80:
                return 'background-color: #c8e6c9; color: #2e7d32'
            elif score >= 60:
                return 'background-color: #fff9c4; color: #f57f17'
            else:
                return 'background-color: #ffcdd2; color: #c62828'
        
        # Apply styling
        styled_df = candidates_df.style.applymap(
            highlight_score, subset=['Match Score']
        )
        
        # Display table
        st.dataframe(styled_df, hide_index=True)
        
        # Candidate selection and actions
        st.markdown("<p class='section-header'>🔍 Candidate Actions</p>", unsafe_allow_html=True)
        
        selected_candidate_idx = st.selectbox(
            "Select candidate to review or modify",
            options=range(len(st.session_state['candidates'])),
            format_func=lambda x: f"{st.session_state['candidates'][x]['name']} ({st.session_state['candidates'][x]['jd_title']})"
        )
        
        if selected_candidate_idx is not None:
            candidate = st.session_state['candidates'][selected_candidate_idx]
            
            col1, col2 = st.columns(2)
            with col1:
                current_status = candidate["shortlisted"]
                new_status = st.checkbox("Shortlist this candidate", value=current_status)
                
                if new_status != current_status:
                    st.session_state['candidates'][selected_candidate_idx]["shortlisted"] = new_status
                    st.success(f"Candidate {candidate['name']} {'shortlisted' if new_status else 'removed from shortlist'}")
            
            with col2:
                view_details = st.button("View Detailed Profile")
                
            if view_details:
                # Display detailed profile
                st.markdown("<h3>📋 Candidate Profile</h3>", unsafe_allow_html=True)
                
                # Profile card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<h3>{candidate['name']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Email:</strong> {candidate['email']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Match Score:</strong> {candidate['match_score']}%</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Status:</strong> {'Shortlisted' if candidate['shortlisted'] else 'Not Shortlisted'}</p>", unsafe_allow_html=True)
                
                # Show resume details similar to the analysis tab
                resume_data = candidate['resume_data']
                st.markdown("<p class='section-header'>📝 Summary</p>", unsafe_allow_html=True)
                st.markdown(resume_data.get('summary', 'No summary available.'))
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Experience and education
                exp_col, edu_col = st.columns(2)
                
                with exp_col:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<p class='section-header'>💼 Experience</p>", unsafe_allow_html=True)
                    if resume_data.get('experience'):
                        for exp in resume_data.get('experience', []):
                            st.markdown(f"**{exp.get('title')}** at {exp.get('company')}")
                            st.markdown(f"{exp.get('date')}")
                            st.markdown(f"{exp.get('description')}")
                            st.markdown("---")
                    else:
                        st.info("No experience data extracted.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with edu_col:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<p class='section-header'>🎓 Education</p>", unsafe_allow_html=True)
                    if resume_data.get('education'):
                        for edu in resume_data.get('education', []):
                            st.markdown(f"**{edu.get('degree')}** from {edu.get('institution')}")
                            st.markdown(f"{edu.get('date')}")
                            st.markdown("---")
                    else:
                        st.info("No education data extracted.")
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No candidates have been analyzed yet. Please upload and analyze resumes in the Candidate Management tab.")
    
    # Shortlisting overview
    if 'candidates' in st.session_state and st.session_state['candidates']:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>🏆 Shortlisting Overview</p>", unsafe_allow_html=True)
        
        # Count shortlisted candidates per job
        jobs = {}
        for c in st.session_state['candidates']:
            job = c['jd_title']
            if job not in jobs:
                jobs[job] = {'total': 0, 'shortlisted': 0}
            
            jobs[job]['total'] += 1
            if c['shortlisted']:
                jobs[job]['shortlisted'] += 1
        
        # Display overview
        for job, counts in jobs.items():
            st.markdown(f"**{job}:** {counts['shortlisted']} shortlisted out of {counts['total']} candidates")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 5: Interview Scheduler
with tabs[4]:
    st.markdown("<h2>📅 Interview Scheduler</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Schedule interviews with shortlisted candidates. Configure interview details and generate 
    personalized invitation emails.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Shortlisted candidates for interviews
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>👥 Shortlisted Candidates</p>", unsafe_allow_html=True)
    
    if 'candidates' in st.session_state and any(c["shortlisted"] for c in st.session_state['candidates']):
        shortlisted = [c for c in st.session_state['candidates'] if c["shortlisted"]]
        
        # Display shortlisted candidates
        shortlisted_df = pd.DataFrame([
            {
                "Name": c["name"],
                "Email": c["email"],
                "Job Position": c["jd_title"],
                "Match Score": f"{c['match_score']}%",
                "Index": i
            } for i, c in enumerate(shortlisted)
        ])
        
        st.dataframe(shortlisted_df, hide_index=True)
        
        # Select candidate for scheduling
        selected_candidate_idx = st.selectbox(
            "Select candidate to schedule interview",
            options=range(len(shortlisted)),
            format_func=lambda x: f"{shortlisted[x]['name']} ({shortlisted[x]['jd_title']})"
        )
        
        if selected_candidate_idx is not None:
            candidate = shortlisted[selected_candidate_idx]
            
            # Interview details form
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>📝 Interview Details</p>", unsafe_allow_html=True)
            
            # Interview type and format
            interview_type = st.selectbox(
                "Interview Type",
                options=["Initial Screening", "Technical Interview", "Team Interview", "Final Interview"]
            )
            
            interview_format = st.selectbox(
                "Interview Format",
                options=["Video Call", "Phone Call", "In-Person", "Technical Assessment"]
            )
            
            # Date and time options
            col1, col2 = st.columns(2)
            with col1:
                interview_date = st.date_input("Interview Date", value=datetime.now() + timedelta(days=3))
            
            with col2:
                interview_time = st.time_input("Interview Time", value=datetime.strptime("14:00", "%H:%M").time())
            
            # Alternate date/time
            st.markdown("<p>Alternative Date/Time (Optional)</p>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                alt_date = st.date_input("Alternative Date", value=datetime.now() + timedelta(days=4))
            
            with col2:
                alt_time = st.time_input("Alternative Time", value=datetime.strptime("10:00", "%H:%M").time())
            
            # Interviewers
            interviewers = st.text_input("Interviewers (comma separated names)")
            
            # Location or meeting link
            if interview_format == "In-Person":
                location = st.text_input("Interview Location")
            else:
                location = st.text_input("Meeting Link or Phone Number")
            
            # Additional notes
            notes = st.text_area("Additional Notes for Candidate", placeholder="Any special instructions or preparation details...")
            
            # Generate email preview
            if st.button("Generate Email"):
                # Create email template
                email_content = generate_interview_email(
                    candidate_name=candidate['name'],
                    job_title=candidate['jd_title'],
                    interview_type=interview_type,
                    interview_format=interview_format,
                    interview_date=interview_date,
                    interview_time=interview_time,
                    alt_date=alt_date,
                    alt_time=alt_time,
                    location=location,
                    interviewers=interviewers,
                    notes=notes
                )
                
                st.session_state['email_preview'] = email_content
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Email preview
            if 'email_preview' in st.session_state:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>📧 Email Preview</p>", unsafe_allow_html=True)
                
                st.text_area("Email Content", value=st.session_state['email_preview'], height=300)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Send Email"):
                        # In a real application, this would send the email
                        st.success(f"Interview invitation sent to {candidate['name']} at {candidate['email']}")
                with col2:
                    st.download_button(
                        "Download Email",
                        data=st.session_state['email_preview'],
                        file_name="interview_invitation.txt",
                        mime="text/plain"
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No candidates have been shortlisted yet. Please shortlist candidates in the Shortlisting tab.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Helper functions (for actual implementation)

def extract_resume_data(uploaded_file):
    """
    Extract key information from a resume file
    In a real implementation, this would use NLP to extract data
    """
    # This is a placeholder function - in a real implementation would use
    # libraries like PyPDF2, python-docx, and NLP to extract resume data
    
    # For demo purposes, return mock data
    return {
        "summary": "Experienced software engineer with 5+ years in full-stack development and ML engineering. Strong skills in Python, React, and cloud technologies.",
        "experience": [
            {
                "title": "Senior Software Engineer",
                "company": "Tech Solutions Inc.",
                "date": "2020 - Present",
                "description": "Led development of ML-powered recommendation systems. Improved system efficiency by 40%."
            },
            {
                "title": "Software Developer",
                "company": "DataCorp",
                "date": "2018 - 2020",
                "description": "Developed RESTful APIs and microservices. Worked with Python, Docker, and AWS."
            }
        ],
        "education": [
            {
                "degree": "MS in Computer Science",
                "institution": "University of Technology",
                "date": "2016 - 2018"
            },
            {
                "degree": "BS in Software Engineering",
                "institution": "Tech Institute",
                "date": "2012 - 2016"
            }
        ],
        "skills": [
            "Python", "JavaScript", "React", "Node.js", "Docker", 
            "Kubernetes", "Machine Learning", "SQL", "MongoDB",
            "AWS", "CI/CD", "Git", "Agile", "TensorFlow"
        ]
    }

def compare_resume_to_jd(resume_data, jd_data):
    """
    Compare resume data to job description and calculate match score
    """
    # This is a simplified implementation for demo purposes
    # In a real application, would use NLP, word embeddings, etc.
    
    # Extract skills from resume
    resume_skills = set([s.lower() for s in resume_data.get('skills', [])])
    
    # Extract required skills from JD
    jd_skills = set([s.lower() for s in jd_data.get('RequiredSkills', [])])
    
    # Calculate skill matches
    if jd_skills:
        matching_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills - resume_skills
        
        # Calculate match percentage
        if len(jd_skills) > 0:
            skill_match_percentage = (len(matching_skills) / len(jd_skills)) * 100
        else:
            skill_match_percentage = 100
    else:
        matching_skills = []
        missing_skills = []
        skill_match_percentage = 50  # Default if no skills are listed
    
    # In a real implementation, would also consider:
    # - Experience level match
    # - Education match
    # - Location match
    # - etc.
    
    # For demo purposes, just return the skill match
    return round(skill_match_percentage), list(matching_skills), list(missing_skills)

def summarize_job_description(jd_text):
    """
    Extract key information from a job description
    In a real implementation, this would use NLP to extract data
    """
    # This is a placeholder function - in a real implementation would use
    # an LLM or other NLP techniques to extract JD data
    
    # For demo purposes, return mock data
    return {
        "JobTitle": "Senior Data Scientist",
        "CompanyOverview": "We are a leading tech company focused on AI solutions.",
        "RequiredSkills": [
            "Python", "Machine Learning", "SQL", "TensorFlow", "PyTorch",
            "Data Visualization", "Statistical Analysis", "NLP"
        ],
        "RequiredExperience": "5+ years",
        "RequiredQualifications": [
            "Master's degree in Computer Science, Statistics, or related field",
            "Experience with deep learning frameworks",
            "Strong communication skills"
        ],
        "JobResponsibilities": [
            "Develop and implement ML models",
            "Analyze large datasets to extract insights",
            "Collaborate with cross-functional teams",
            "Present findings to stakeholders",
            "Research and implement new ML techniques"
        ],
        "KeywordsSummary": "This role requires strong skills in machine learning, Python, and data analysis. The ideal candidate will have experience with deep learning frameworks and a strong educational background in a related field."
    }

def generate_interview_email(candidate_name, job_title, interview_type, interview_format, 
                            interview_date, interview_time, alt_date, alt_time, 
                            location, interviewers, notes):
    """Generate personalized interview invitation email"""
    
    # Format dates and times
    date_str = interview_date.strftime("%A, %B %d, %Y")
    time_str = interview_time.strftime("%I:%M %p")
    
    alt_date_str = alt_date.strftime("%A, %B %d, %Y")
    alt_time_str = alt_time.strftime("%I:%M %p")
    
    # Format interviewers
    if interviewers:
        interviewers_list = [name.strip() for name in interviewers.split(",")]
        if len(interviewers_list) == 1:
            interviewers_str = interviewers
        elif len(interviewers_list) == 2:
            interviewers_str = f"{interviewers_list[0]} and {interviewers_list[1]}"
        else:
            interviewers_str = ", ".join(interviewers_list[:-1]) + f", and {interviewers_list[-1]}"
    else:
        interviewers_str = "our team"
    
    # Create email template
    email = f"""Subject: Interview Invitation: {job_title} Position at [Company Name]

Dear {candidate_name},

We are pleased to inform you that your application for the {job_title} position has been shortlisted. We would like to invite you for a {interview_type.lower()} interview to further discuss your qualifications and experience.

Interview Details:
- Position: {job_title}
- Interview Type: {interview_type}
- Format: {interview_format}
- Date: {date_str}
- Time: {time_str}
- {"Location: " + location if location else ""}

{f"Alternative Slot (if the above time doesn't work for you):\\n- Date: {alt_date_str}\\n- Time: {alt_time_str}\\n" if alt_date and alt_time else ""}

You will be interviewed by {interviewers_str}.

{f"Additional Information:\\n{notes}" if notes else ""}

Please confirm your availability for this interview by replying to this email. If the proposed time doesn't work for you, please let us know your preference from the alternative options provided.

We look forward to speaking with you and learning more about your experience and skills.

Best regards,
[Your Name]
Recruitment Team
[Company Name]
[Contact Information]
"""
    
    return email
