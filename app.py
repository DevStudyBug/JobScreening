import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import requests
from datetime import datetime, timedelta

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

# Tab 3: Candidate Management
with tabs[2]:
    st.markdown("<h2>👥 Candidate Management</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This section helps you manage candidate information and shortlisting based on resume analysis.
    View all analyzed candidates, filter by match score, and manage the recruitment pipeline.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Candidate list view
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>👥 Candidate Database</p>", unsafe_allow_html=True)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_option = st.selectbox("Filter by status", ["All", "Shortlisted", "Not qualified", "Scheduled"])
    with col2:
        sort_option = st.selectbox("Sort by", ["Match percentage (high to low)", "Match percentage (low to high)", "Name (A-Z)", "Date added (newest first)"])
    with col3:
        search_term = st.text_input("Search by name or position", "")
    
    # Apply filters and sorting
    filtered_candidates = st.session_state['candidates']
