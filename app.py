import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import random

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text() or ""  # Handle None case
            text += extracted_text
        
        if not text.strip():
            # If no text was extracted (possibly an image-based PDF)
            return "This appears to be an image-based PDF. Please provide a text-based PDF or manually enter the content."
            
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Job Description Summarizer Agent
def summarize_job_description(jd_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Act as an expert job description analyzer. Review the following job description and extract 
    key elements in a structured format.
    
    Job Description: {jd_text}
    
    Respond with a valid JSON object containing these fields:
    {{
      "JobTitle": "title here",
      "Department": "department name",
      "Location": "location",
      "EmploymentType": "full-time/part-time/contract",
      "RequiredSkills": ["skill1", "skill2", "..."],
      "RequiredExperience": "X years in...",
      "RequiredQualifications": ["qualification1", "qualification2", "..."],
      "Responsibilities": ["responsibility1", "responsibility2", "..."],
      "SalaryRange": "range if mentioned",
      "PreferredSkills": ["skill1", "skill2", "..."]
    }}
    
    Important: Only respond with the JSON object and nothing else. No explanations or markdown formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            # Clean the response text
            response_text = response.text.strip()
            
            # Handle different response formats
            if response_text.startswith("json") and response_text.endswith(""):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("") and response_text.endswith(""):
                response_text = response_text[3:-3].strip()
            
            # If the response still contains markdown or non-JSON text, try to extract JSON portion
            if not response_text.startswith("{"):
                # Look for JSON object in the response
                start_index = response_text.find("{")
                end_index = response_text.rfind("}")
                
                if start_index >= 0 and end_index >= 0:
                    response_text = response_text[start_index:end_index+1]
            
            # Debug output if needed
            if st.session_state.get('debug_mode', False):
                st.write("Raw API response:", response.text)
                st.write("Processed response text:", response_text)
            
            # Try to parse as JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as json_err:
                # If direct parsing fails, try a fallback approach
                return {
                    "JobTitle": extract_field_from_text(response_text, "JobTitle") or "Data Analyst",
                    "Department": extract_field_from_text(response_text, "Department") or "Not specified",
                    "Location": extract_field_from_text(response_text, "Location") or "Not specified",
                    "EmploymentType": extract_field_from_text(response_text, "EmploymentType") or "Full-time",
                    "RequiredSkills": extract_list_from_text(response_text, "RequiredSkills") or ["Python", "Data Analysis"],
                    "RequiredExperience": extract_field_from_text(response_text, "RequiredExperience") or "2+ years",
                    "RequiredQualifications": extract_list_from_text(response_text, "RequiredQualifications") or ["Bachelor's degree"],
                    "Responsibilities": extract_list_from_text(response_text, "Responsibilities") or ["Data Analysis", "Reporting"],
                    "SalaryRange": extract_field_from_text(response_text, "SalaryRange") or "Not specified",
                    "PreferredSkills": extract_list_from_text(response_text, "PreferredSkills") or []
                }
        else:
            return {"error": "Failed to get a valid response from the API"}
            
    except Exception as e:
        return {"error": f"Failed to process the JD: {str(e)}"}

# Helper functions to extract info from text if JSON parsing fails
def extract_field_from_text(text, field_name):
    if not text:
        return None
    
    # Try to find field in format "field_name": "value"
    import re
    pattern = f'"{field_name}"\\s*:\\s*"([^"]*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def extract_list_from_text(text, field_name):
    if not text:
        return None
    
    # Try to find field in format "field_name": ["value1", "value2"]
    import re
    pattern = f'"{field_name}"\\s*:\\s*\\[(.*?)\\]'
    match = re.search(pattern, text)
    if match:
        items_text = match.group(1)
        # Extract individual items
        items = re.findall(r'"([^"]*)"', items_text)
        return items
    return None

# Recruiting Agent for CV Analysis
def analyze_cv(cv_text, jd_summary):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Format JD summary for the prompt
    required_skills = ", ".join(jd_summary.get("RequiredSkills", []))
    preferred_skills = ", ".join(jd_summary.get("PreferredSkills", []))
    responsibilities = ", ".join(jd_summary.get("Responsibilities", []))
    qualifications = ", ".join(jd_summary.get("RequiredQualifications", []))
    
    prompt = f"""
    Act as a senior recruiting agent specializing in talent acquisition. Analyze this candidate's 
    resume against the job requirements and provide a detailed evaluation.
    
    Job Title: {jd_summary.get("JobTitle", "Not specified")}
    Required Skills: {required_skills}
    Preferred Skills: {preferred_skills}
    Required Experience: {jd_summary.get("RequiredExperience", "Not specified")}
    Required Qualifications: {qualifications}
    Key Responsibilities: {responsibilities}
    
    Candidate Resume: {cv_text}
    
    Respond with ONLY a valid JSON object containing:
    {{
      "CandidateName": "full name",
      "ContactInfo": "email and/or phone",
      "Skills": ["skill1", "skill2", "..."],
      "Experience": ["experience1", "experience2", "..."],
      "Education": ["education1", "education2", "..."],
      "Certifications": ["cert1", "cert2", "..."],
      "SkillMatch": "X%",
      "ExperienceMatch": "X%",
      "QualificationMatch": "X%",
      "OverallMatch": "X%",
      "MatchedSkills": ["skill1", "skill2", "..."],
      "MissingSkills": ["skill1", "skill2", "..."],
      "Strengths": ["strength1", "strength2", "..."],
      "Areas_for_Improvement": ["area1", "area2", "..."],
      "Recommendation": "shortlist/reject/further review"
    }}
    
    Important: Only provide the JSON object. No additional text, no markdown formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            # Clean the response text
            response_text = response.text.strip()
            
            # Handle different response formats
            if response_text.startswith("json") and response_text.endswith(""):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("") and response_text.endswith(""):
                response_text = response_text[3:-3].strip()
            
            # If the response still contains non-JSON text, try to extract JSON portion
            if not response_text.startswith("{"):
                start_index = response_text.find("{")
                end_index = response_text.rfind("}")
                
                if start_index >= 0 and end_index >= 0:
                    response_text = response_text[start_index:end_index+1]
            
            # Debug output
            if st.session_state.get('debug_mode', False):
                st.write("Raw CV analysis response:", response.text)
                st.write("Processed CV analysis text:", response_text)
            
            # Create a fallback response if parsing fails
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Extract name from resume if possible
                candidate_name = "Unknown Candidate"
                name_match = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)", cv_text[:500])
                if name_match:
                    candidate_name = name_match.group(1)
                
                # Extract email if possible
                contact_info = "Not found"
                email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cv_text)
                if email_match:
                    contact_info = email_match.group(0)
                
                return {
                    "CandidateName": candidate_name,
                    "ContactInfo": contact_info,
                    "Skills": ["Unable to parse skills"],
                    "Experience": ["Experience details not parsed"],
                    "Education": ["Education details not parsed"],
                    "Certifications": [],
                    "SkillMatch": "0%",
                    "ExperienceMatch": "0%",
                    "QualificationMatch": "0%",
                    "OverallMatch": "50%",
                    "MatchedSkills": [],
                    "MissingSkills": jd_summary.get("RequiredSkills", []),
                    "Strengths": ["Unable to determine strengths"],
                    "Areas_for_Improvement": ["Resume parsing failed, please review manually"],
                    "Recommendation": "further review"
                }
        else:
            return {"error": "Failed to get a valid response from the API for CV analysis"}
            
    except Exception as e:
        return {"error": f"Failed to analyze CV: {str(e)}"}

# Candidate Shortlisting Agent
def shortlist_candidates(candidates_analysis, threshold=70):
    shortlisted = []
    
    for candidate in candidates_analysis:
        # Skip entries with errors
        if "error" in candidate:
            continue
            
        # Extract match percentage
        match_percentage = int(candidate.get("OverallMatch", "0%").strip("%"))
        
        if match_percentage >= threshold:
            shortlisted.append({
                "name": candidate.get("CandidateName", "Unknown"),
                "contact": candidate.get("ContactInfo", "Not provided"),
                "match_percentage": match_percentage,
                "strengths": candidate.get("Strengths", []),
                "missing_skills": candidate.get("MissingSkills", []),
                "recommendation": candidate.get("Recommendation", "")
            })
    
    # Sort by match percentage (highest first)
    shortlisted.sort(key=lambda x: x["match_percentage"], reverse=True)
    return shortlisted

# Interview Scheduler Agent
def generate_interview_email(candidate_info, jd_summary):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate interview dates (next business days)
    today = datetime.now()
    proposed_dates = []
    
    for i in range(1, 8):
        next_date = today + timedelta(days=i)
        # Skip weekends (5 = Saturday, 6 = Sunday)
        if next_date.weekday() < 5:
            proposed_dates.append(next_date.strftime("%A, %B %d, %Y"))
            if len(proposed_dates) == 3:  # Get 3 business days
                break
    
    # Generate interview times
    interview_times = ["10:00 AM", "11:30 AM", "2:00 PM", "3:30 PM"]
    proposed_slots = [f"{date} at {time}" for date in proposed_dates for time in random.sample(interview_times, 2)]
    
    job_title = jd_summary.get("JobTitle", "the open position")
    company = os.getenv("COMPANY_NAME", "Our Company")
    
    # Handle missing strengths
    candidate_strengths = candidate_info.get('strengths', [])
    if not candidate_strengths or len(candidate_strengths) == 0:
        candidate_strengths = ["qualifications", "experience"]
    
    strengths_text = ', '.join(candidate_strengths[:3]) if len(candidate_strengths) > 0 else "qualifications"
    
    prompt = f"""
    Act as a professional recruiter. Write a personalized interview invitation email for {candidate_info['name']} 
    who has been shortlisted for the {job_title} position at {company}.
    
    Candidate's strengths: {strengths_text}
    Match rate: {candidate_info['match_percentage']}%
    
    Include these proposed interview slots:
    {', '.join(proposed_slots[:5])}
    
    The email should:
    1. Be professional and warm
    2. Congratulate them on being shortlisted
    3. Briefly mention why they're a good fit, highlighting 1-2 strengths
    4. Propose the interview slots and ask for their preference
    5. Mention the interview will be conducted via video call (Zoom)
    6. Explain next steps and whom to contact with questions
    
    Respond with only the email text, no additional formatting or explanation.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            email_text = response.text.strip()
            
            # Debug output
            if st.session_state.get('debug_mode', False):
                st.write("Raw email response:", email_text)
            
            return {
                "candidate_name": candidate_info['name'],
                "candidate_email": candidate_info['contact'],
                "email_subject": f"Interview Invitation: {job_title} position at {company}",
                "email_body": email_text,
                "proposed_slots": proposed_slots[:5]
            }
        else:
            # Fallback email if API fails
            default_email = f"""
Dear {candidate_info['name']},

Congratulations! We are pleased to inform you that you have been shortlisted for the {job_title} position at {company}.

We were impressed with your profile and would like to invite you for a video interview to discuss your experience and the role in more detail.

Please let us know which of the following time slots would work best for you:
- {proposed_slots[0]}
- {proposed_slots[1]}
- {proposed_slots[2]}

The interview will be conducted via Zoom, and we will send you the meeting details once you confirm your preferred time slot.

If you have any questions, please don't hesitate to contact us.

We look forward to speaking with you soon!

Best regards,
Recruiting Team
{company}
            """
            
            return {
                "candidate_name": candidate_info['name'],
                "candidate_email": candidate_info['contact'],
                "email_subject": f"Interview Invitation: {job_title} position at {company}",
                "email_body": default_email,
                "proposed_slots": proposed_slots[:5]
            }
    except Exception as e:
        return {"error": f"Failed to generate email: {str(e)}"}

# Function to send emails
def send_interview_email(email_data):
    try:
        sender_email = os.getenv("EMAIL_ADDRESS")
        password = os.getenv("EMAIL_PASSWORD")
        
        if not sender_email or not password:
            return {
                "status": "error", 
                "message": "Email credentials not found. Please add EMAIL_ADDRESS and EMAIL_PASSWORD to your .env file."
            }
        
        recipient_email = email_data['candidate_email']
        
        # For debugging - print credentials (remove in production)
        if st.session_state.get('debug_mode', False):
            st.write(f"Using email: {sender_email}")
            st.write(f"Password length: {len(password)} characters")
            st.write(f"Sending to: {recipient_email}")
        
        # Check if recipient email is valid
        if not re.match(r"[^@]+@[^@]+\.[^@]+", recipient_email):
            return {
                "status": "error", 
                "message": f"Invalid recipient email address: {recipient_email}"
            }
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = email_data['email_subject']
        
        msg.attach(MIMEText(email_data['email_body'], 'plain'))
        
        # Connect to Gmail's SMTP server with extended timeout and debug level
        try:
            with smtplib.SMTP('smtp.gmail.com', 587, timeout=30) as server:
                if st.session_state.get('debug_mode', False):
                    server.set_debuglevel(1)  # Enable verbose debug output
                
                # Identify ourselves to the SMTP server
                server.ehlo()
                
                # Enable TLS encryption
                server.starttls()
                
                # Re-identify ourselves over TLS connection
                server.ehlo()
                
                # Login with credentials
                server.login(sender_email, password)
                
                # Send the message
                server.send_message(msg)
                
                return {
                    "status": "success", 
                    "message": f"Email sent to {recipient_email}"
                }
        except smtplib.SMTPServerDisconnected as e:
            return {"status": "error", "message": f"Server disconnected: {str(e)}. Check your internet connection."}
        except smtplib.SMTPAuthenticationError as e:
            return {"status": "error", "message": f"Authentication failed: {str(e)}. Verify your email and App Password."}
        except smtplib.SMTPException as e:
            return {"status": "error", "message": f"SMTP error: {str(e)}"}
            
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error sending email: {str(e)}"
        }

# Import regex for fallback parsing
import re

# Page configuration
st.set_page_config(
    page_title="RecruitEase | Multi-Agent Recruiting System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (reusing from the reference code with some additions)
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --main-blue: #1E88E5;
        --light-blue: #BBD9F2;
        --dark-blue: #0D47A1;
        --accent-blue: #64B5F6;
        --success-green: #4CAF50;
        --warning-yellow: #FFC107;
        --danger-red: #F44336;
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
    
    /* Step cards with indicators */
    .step-card {
        border-left: 5px solid var(--main-blue);
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    .step-complete {
        border-left: 5px solid var(--success-green);
    }
    
    .step-active {
        border-left: 5px solid var(--warning-yellow);
    }
    
    .step-waiting {
        border-left: 5px solid #E0E0E0;
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
    
    /* Missing keyword pills */
    .missing-keyword {
        background-color: #ffccbc;
        color: #d32f2f;
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
    
    /* Candidate card */
    .candidate-card {
        background-color: white;
        border-radius: 10px;
        margin-bottom: 15px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .candidate-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Email preview */
    .email-preview {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 1
if 'jd_text' not in st.session_state:
    st.session_state['jd_text'] = ""
if 'jd_summary' not in st.session_state:
    st.session_state['jd_summary'] = None
if 'resumes' not in st.session_state:
    st.session_state['resumes'] = []
if 'candidates_analysis' not in st.session_state:
    st.session_state['candidates_analysis'] = []
if 'shortlisted_candidates' not in st.session_state:
    st.session_state['shortlisted_candidates'] = []
if 'interview_emails' not in st.session_state:
    st.session_state['interview_emails'] = {}
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False
if 'emails_sent' not in st.session_state:
    st.session_state['emails_sent'] = set()  # Keep track of sent emails

# Sidebar content
with st.sidebar:
    st.image("https://via.placeholder.com/80x80.png?text=RE", width=80)
    st.markdown("## RecruitEase")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    
    # Highlight current step
    step1_class = "step-complete" if st.session_state['current_step'] > 1 else "step-active" if st.session_state['current_step'] == 1 else "step-waiting"
    step2_class = "step-complete" if st.session_state['current_step'] > 2 else "step-active" if st.session_state['current_step'] == 2 else "step-waiting"
    step3_class = "step-complete" if st.session_state['current_step'] > 3 else "step-active" if st.session_state['current_step'] == 3 else "step-waiting"
    step4_class = "step-complete" if st.session_state['current_step'] > 4 else "step-active" if st.session_state['current_step'] == 4 else "step-waiting"
    
    st.markdown(f"<div class='{step1_class}' style='padding:10px; margin-bottom:10px; border-radius:5px;'>Step 1: Job Description Analysis</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='{step2_class}' style='padding:10px; margin-bottom:10px; border-radius:5px;'>Step 2: CV Processing</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='{step3_class}' style='padding:10px; margin-bottom:10px; border-radius:5px;'>Step 3: Candidate Shortlisting</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='{step4_class}' style='padding:10px; margin-bottom:10px; border-radius:5px;'>Step 4: Interview Scheduling</div>", unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Start New Process"):
        # Reset session state
        st.session_state['current_step'] = 1
        st.session_state['jd_text'] = ""
        st.session_state['jd_summary'] = None
        st.session_state['resumes'] = []
        st.session_state['candidates_analysis'] = []
        st.session_state['shortlisted_candidates'] = []
        st.session_state['interview_emails'] = {}
        st.experimental_rerun()
    
    # Settings
    st.markdown("### ⚙️ Settings")
    st.session_state['debug_mode'] = st.checkbox("Enable Debug Mode", st.session_state['debug_mode'])
    
    # About
    st.markdown("### ℹ️ About RecruitEase")
    st.markdown("""
    RecruitEase is a multi-agent AI system that automates the recruitment process from job description analysis to interview scheduling.
    """)
    
    # Footer
    st.markdown("<div class='footer'>© 2025 RecruitEase | v1.0</div>", unsafe_allow_html=True)

# Main content
st.markdown("<h1 class='main-header'>👥 RecruitEase: Multi-Agent Recruiting System</h1>", unsafe_allow_html=True)

# Step 1: Job Description Analysis
if st.session_state['current_step'] == 1:
    st.markdown("<h2>Step 1: Job Description Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Our AI will analyze your job description to identify key requirements, skills, and qualifications. 
    This helps in accurately matching candidates to your job opening.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📄 Job Description</p>", unsafe_allow_html=True)
    jd_text = st.text_area("Paste the job description here", st.session_state['jd_text'], height=250, 
                          placeholder="Paste the complete job description here...")
    st.session_state['jd_text'] = jd_text
    
    if st.button("🔍 Analyze Job Description"):
        if jd_text.strip():
            with st.spinner("⏳ Analyzing job description..."):
                jd_summary = summarize_job_description(jd_text)
                st.session_state['jd_summary'] = jd_summary
                
                if "error" not in jd_summary:
                    st.success("Job description analyzed successfully!")
                    st.session_state['current_step'] = 2
                    st.experimental_rerun()
                else:
                    st.error(jd_summary["error"])
        else:
            st.error("Please paste a job description to proceed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Step 2: CV Processing
elif st.session_state['current_step'] == 2:
    st.markdown("<h2>Step 2: CV Processing</h2>", unsafe_allow_html=True)
    
    # Display JD Summary
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📋 Job Description Summary</p>", unsafe_allow_html=True)
    
    jd_summary = st.session_state['jd_summary']
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"*Job Title:* {jd_summary.get('JobTitle', 'Not specified')}")
        st.markdown(f"*Department:* {jd_summary.get('Department', 'Not specified')}")
        st.markdown(f"*Location:* {jd_summary.get('Location', 'Not specified')}")
        st.markdown(f"*Employment Type:* {jd_summary.get('EmploymentType', 'Not specified')}")
        st.markdown(f"*Experience:* {jd_summary.get('RequiredExperience', 'Not specified')}")
        
    with col2:
        st.markdown("*Required Skills:*")
        for skill in jd_summary.get('RequiredSkills', []):
            st.markdown(f"<span class='keyword-pill'>🔍 {skill}</span>", unsafe_allow_html=True)
            
        st.markdown("*Preferred Skills:*")
        for skill in jd_summary.get('PreferredSkills', []):
            st.markdown(f"<span class='keyword-pill'>✨ {skill}</span>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # CV Upload
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📎 Upload Resumes</p>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload candidate resumes (PDF format)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        # Keep track of new uploads
        new_uploads = []
        existing_names = [resume['name'] for resume in st.session_state['resumes']]
        
        for file in uploaded_files:
            if file.name not in existing_names:
                new_uploads.append(file)
                st.session_state['resumes'].append({
                    'name': file.name,
                    'file': file,
                    'analyzed': False
                })
        
        if new_uploads:
            st.success(f"{len(new_uploads)} new resume(s) uploaded successfully!")
        
        # Display uploaded files
        st.markdown("### Uploaded Resumes")
        for i, resume in enumerate(st.session_state['resumes']):
            status = "✅ Analyzed" if resume['analyzed'] else "⏳ Pending Analysis"
            st.markdown(f"{i+1}. {resume['name']} - {status}")
    
    if st.session_state['resumes']:
        if st.button("📊 Analyze All Resumes"):
            with st.spinner("⏳ Analyzing resumes against job requirements..."):
                # Reset candidates analysis
                st.session_state['candidates_analysis'] = []
                
                for i, resume in enumerate(st.session_state['resumes']):
                    if not resume['analyzed']:
                        # Extract text from resume
                        cv_text = input_pdf_text(resume['file'])
                        
                        # Analyze the CV
                        analysis = analyze_cv(cv_text, st.session_state['jd_summary'])
                        
                        if "error" not in analysis:
                            st.session_state['resumes'][i]['analyzed'] = True
                            st.session_state['candidates_analysis'].append(analysis)
                        else:
                            st.session_state['candidates_analysis'].append({
                                "error": f"Failed to analyze {resume['name']}: {analysis['error']}",
                                "CandidateName": f"Error with {resume['name']}"
                            })
                
                st.success(f"Analyzed {len(st.session_state['resumes'])} resume(s)!")
                st.session_state['current_step'] = 3
                st.experimental_rerun()
    else:
        st.info("Please upload at least one resume to proceed.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Back button
    if st.button("⬅️ Back to Job Description"):
        st.session_state['current_step'] = 1
        st.experimental_rerun()

# Step 3: Candidate Shortlisting
elif st.session_state['current_step'] == 3:
    st.markdown("<h2>Step 3: Candidate Shortlisting</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>🎯 Set Shortlisting Criteria</p>", unsafe_allow_html=True)
    
    # Shortlisting threshold slider
    threshold = st.slider("Minimum Match Percentage for Shortlisting", min_value=50, max_value=95, value=70, step=5)
    
    # Candidates analysis results
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📊 Candidate Analysis Results</p>", unsafe_allow_html=True)
    
    if st.session_state['candidates_analysis']:
        for i, candidate in enumerate(st.session_state['candidates_analysis']):
            if "error" in candidate:
                st.error(candidate["error"])
                continue
                
            # Create an expandable section for each candidate
            with st.expander(f"📄 {candidate.get('CandidateName', f'Candidate {i+1}')} - Match: {candidate.get('OverallMatch', '0%')}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"*Name:* {candidate.get('CandidateName', 'Not identified')}")
                    st.markdown(f"*Contact:* {candidate.get('ContactInfo', 'Not found')}")
                    st.markdown(f"*Education:* {', '.join(candidate.get('Education', ['Not specified']))}")
                    
                    st.markdown("*Experience:*")
                    for exp in candidate.get('Experience', [])[:3]:
                        st.markdown(f"- {exp}")
                
                with col2:
                    # Match percentages
                    st.markdown("*Match Scores:*")
                    st.markdown(f"Skills: {candidate.get('SkillMatch', '0%')}")
                    st.markdown(f"Experience: {candidate.get('ExperienceMatch', '0%')}")
                    st.markdown(f"Qualifications: {candidate.get('QualificationMatch', '0%')}")
                    st.markdown(f"Overall: {candidate.get('OverallMatch', '0%')}")
                    
                    # Recommendation
                    recommendation = candidate.get('Recommendation', 'No recommendation')
                    rec_color = "#4CAF50" if "shortlist" in recommendation.lower() else "#F44336" if "reject" in recommendation.lower() else "#FFC107"
                    st.markdown(f"*Recommendation:* <span style='color:{rec_color};font-weight:bold;'>{recommendation}</span>", unsafe_allow_html=True)
                
                # Skills section
                st.markdown("*Matched Skills:*")
                for skill in candidate.get('MatchedSkills', []):
                    st.markdown(f"<span class='keyword-pill matched-keyword'>✓ {skill}</span>", unsafe_allow_html=True)
                    
                st.markdown("*Missing Skills:*")
                for skill in candidate.get('MissingSkills', []):
                    st.markdown(f"<span class='keyword-pill missing-keyword'>✗ {skill}</span>", unsafe_allow_html=True)
                
                # Strengths and Areas for Improvement
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("*Strengths:*")
                    for strength in candidate.get('Strengths', []):
                        st.markdown(f"- {strength}")
                        
                with col2:
                    st.markdown("*Areas for Improvement:*")
                    for area in candidate.get('Areas_for_Improvement', []):
                        st.markdown(f"- {area}")
    else:
        st.warning("No candidates have been analyzed yet.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Shortlist button
    if st.button("👍 Shortlist Candidates"):
        with st.spinner("⏳ Shortlisting candidates..."):
            shortlisted = shortlist_candidates(st.session_state['candidates_analysis'], threshold)
            st.session_state['shortlisted_candidates'] = shortlisted
            
            st.success(f"Shortlisted {len(shortlisted)} candidate(s)!")
            if shortlisted:
                st.session_state['current_step'] = 4
                st.experimental_rerun()
            else:
                st.warning("No candidates met the threshold criteria. Consider lowering the threshold.")
    
    # Back button
    if st.button("⬅️ Back to Resume Upload"):
        st.session_state['current_step'] = 2
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Step 4: Interview Scheduling
elif st.session_state['current_step'] == 4:
    st.markdown("<h2>Step 4: Interview Scheduling</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>👥 Shortlisted Candidates</p>", unsafe_allow_html=True)
    
    if st.session_state['shortlisted_candidates']:
        for i, candidate in enumerate(st.session_state['shortlisted_candidates']):
            st.markdown(f"<div class='candidate-card'>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"*{i+1}. {candidate['name']}* ({candidate['match_percentage']}% Match)")
                st.markdown(f"Contact: {candidate['contact']}")
                
                # Truncate strengths to first 2
                strengths_text = ", ".join([s.split(':')[0] if ':' in s else s for s in candidate['strengths'][:2]])
                st.markdown(f"Key strengths: {strengths_text}")
            
            with col2:
                email_status = "📨 Sent" if candidate['name'] in st.session_state['interview_emails'] else "📝 Draft Email"
                st.markdown(f"<div style='padding:5px;border-radius:5px;background-color:{'#c8e6c9' if candidate['name'] in st.session_state['interview_emails'] else '#e0e0e0'};text-align:center;'>{email_status}</div>", unsafe_allow_html=True)
                
            with col3:
                # Email button per candidate
                button_label = "View Email" if candidate['name'] in st.session_state['interview_emails'] else "Create Email"
                if st.button(button_label, key=f"email_btn_{i}"):
                    if candidate['name'] not in st.session_state['interview_emails']:
                        with st.spinner(f"⏳ Generating email for {candidate['name']}..."):
                            email_data = generate_interview_email(candidate, st.session_state['jd_summary'])
                            if "error" not in email_data:
                                st.session_state['interview_emails'][candidate['name']] = email_data
                            else:
                                st.error(email_data["error"])
                    st.experimental_rerun()
                    
            st.markdown(f"</div>", unsafe_allow_html=True)
            
            # If this candidate has an email generated, show it when expanded
            if candidate['name'] in st.session_state['interview_emails']:
                with st.expander("📧 View Email Draft"):
                    email_data = st.session_state['interview_emails'][candidate['name']]
                    st.markdown(f"*Subject:* {email_data['email_subject']}")
                    st.markdown("<div class='email-preview'>", unsafe_allow_html=True)
                    st.markdown(email_data['email_body'].replace('\n', '<br>'), unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Send individual email
                    if st.button("✉️ Send Email", key=f"send_btn_{i}") and candidate['name'] not in st.session_state['emails_sent']:
                        with st.spinner("⏳ Sending email..."):
                            result = send_interview_email(email_data)
                            if result["status"] == "success":
                                st.success(result["message"])
                                st.session_state['emails_sent'].add(candidate['name'])
                            else:
                                st.error(f"Failed to send email: {result['message']}")
    else:
        st.warning("No candidates have been shortlisted yet.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Send all emails button
    if st.session_state['shortlisted_candidates'] and st.session_state['interview_emails']:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📨 Send All Interview Emails", use_container_width=True):
                with st.spinner("⏳ Sending all emails..."):
                    success_count = 0
                    failed_emails = []
                    
                    for candidate_name, email_data in st.session_state['interview_emails'].items():
                        # Skip already sent emails
                        if candidate_name in st.session_state['emails_sent']:
                            success_count += 1
                            continue
                            
                        result = send_interview_email(email_data)
                        if result["status"] == "success":
                            success_count += 1
                            st.session_state['emails_sent'].add(candidate_name)
                        else:
                            failed_emails.append((candidate_name, result["message"]))
                    
                    if not failed_emails:
                        st.success(f"Successfully sent {success_count} interview emails!")
                    else:
                        st.warning(f"Sent {success_count} out of {len(st.session_state['interview_emails'])} emails.")
                        for name, error in failed_emails:
                            st.error(f"Failed to send email to {name}: {error}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Back button
    if st.button("⬅️ Back to Shortlisting"):
        st.session_state['current_step'] = 3
        st.experimental_rerun()

# Process completion
if st.session_state['current_step'] > 4:
    st.markdown("<h2>🎉 Recruitment Process Complete!</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Congratulations! You have successfully completed the automated recruitment process:
    
    1. ✅ *Job Description Analysis*: Extracted key requirements and skills
    2. ✅ *CV Processing*: Analyzed candidate qualifications against job requirements
    3. ✅ *Candidate Shortlisting*: Identified the best candidates based on match scores
    4. ✅ *Interview Scheduling*: Sent interview invitations to qualified candidates
    
    To start a new recruitment process, click the "Start New Process" button in the sidebar.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
[19:25, 03/04/2025] Ishaan Ltce: import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import random

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text() or ""  # Handle None case
            text += extracted_text
        
        if not text.strip():
            # If no text was extracted (possibly an image-based PDF)
            return "This appears to be an image-based PDF. Please provide a text-based PDF or manually enter the content."
            
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Job Description Summarizer Agent
def summarize_job_description(jd_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Act as an expert job description analyzer. Review the following job description and extract 
    key elements in a structured format.
    
    Job Description: {jd_text}
    
    Respond with a valid JSON object containing these fields:
    {{
      "JobTitle": "title here",
      "Department": "department name",
      "Location": "location",
      "EmploymentType": "full-time/part-time/contract",
      "RequiredSkills": ["skill1", "skill2", "..."],
      "RequiredExperience": "X years in...",
      "RequiredQualifications": ["qualification1", "qualification2", "..."],
      "Responsibilities": ["responsibility1", "responsibility2", "..."],
      "SalaryRange": "range if mentioned",
      "PreferredSkills": ["skill1", "skill2", "..."]
    }}
    
    Important: Only respond with the JSON object and nothing else. No explanations or markdown formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            # Clean the response text
            response_text = response.text.strip()
            
            # Handle different response formats
            if response_text.startswith("json") and response_text.endswith(""):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("") and response_text.endswith(""):
                response_text = response_text[3:-3].strip()
            
            # If the response still contains markdown or non-JSON text, try to extract JSON portion
            if not response_text.startswith("{"):
                # Look for JSON object in the response
                start_index = response_text.find("{")
                end_index = response_text.rfind("}")
                
                if start_index >= 0 and end_index >= 0:
                    response_text = response_text[start_index:end_index+1]
            
            # Debug output if needed
            if st.session_state.get('debug_mode', False):
                st.write("Raw API response:", response.text)
                st.write("Processed response text:", response_text)
            
            # Try to parse as JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as json_err:
                # If direct parsing fails, try a fallback approach
                return {
                    "JobTitle": extract_field_from_text(response_text, "JobTitle") or "Data Analyst",
                    "Department": extract_field_from_text(response_text, "Department") or "Not specified",
                    "Location": extract_field_from_text(response_text, "Location") or "Not specified",
                    "EmploymentType": extract_field_from_text(response_text, "EmploymentType") or "Full-time",
                    "RequiredSkills": extract_list_from_text(response_text, "RequiredSkills") or ["Python", "Data Analysis"],
                    "RequiredExperience": extract_field_from_text(response_text, "RequiredExperience") or "2+ years",
                    "RequiredQualifications": extract_list_from_text(response_text, "RequiredQualifications") or ["Bachelor's degree"],
                    "Responsibilities": extract_list_from_text(response_text, "Responsibilities") or ["Data Analysis", "Reporting"],
                    "SalaryRange": extract_field_from_text(response_text, "SalaryRange") or "Not specified",
                    "PreferredSkills": extract_list_from_text(response_text, "PreferredSkills") or []
                }
        else:
            return {"error": "Failed to get a valid response from the API"}
            
    except Exception as e:
        return {"error": f"Failed to process the JD: {str(e)}"}

# Helper functions to extract info from text if JSON parsing fails
def extract_field_from_text(text, field_name):
    if not text:
        return None
    
    # Try to find field in format "field_name": "value"
    import re
    pattern = f'"{field_name}"\\s*:\\s*"([^"]*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def extract_list_from_text(text, field_name):
    if not text:
        return None
    
    # Try to find field in format "field_name": ["value1", "value2"]
    import re
    pattern = f'"{field_name}"\\s*:\\s*\\[(.*?)\\]'
    match = re.search(pattern, text)
    if match:
        items_text = match.group(1)
        # Extract individual items
        items = re.findall(r'"([^"]*)"', items_text)
        return items
    return None

# Recruiting Agent for CV Analysis
def analyze_cv(cv_text, jd_summary):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Format JD summary for the prompt
    required_skills = ", ".join(jd_summary.get("RequiredSkills", []))
    preferred_skills = ", ".join(jd_summary.get("PreferredSkills", []))
    responsibilities = ", ".join(jd_summary.get("Responsibilities", []))
    qualifications = ", ".join(jd_summary.get("RequiredQualifications", []))
    
    prompt = f"""
    Act as a senior recruiting agent specializing in talent acquisition. Analyze this candidate's 
    resume against the job requirements and provide a detailed evaluation.
    
    Job Title: {jd_summary.get("JobTitle", "Not specified")}
    Required Skills: {required_skills}
    Preferred Skills: {preferred_skills}
    Required Experience: {jd_summary.get("RequiredExperience", "Not specified")}
    Required Qualifications: {qualifications}
    Key Responsibilities: {responsibilities}
    
    Candidate Resume: {cv_text}
    
    Respond with ONLY a valid JSON object containing:
    {{
      "CandidateName": "full name",
      "ContactInfo": "email and/or phone",
      "Skills": ["skill1", "skill2", "..."],
      "Experience": ["experience1", "experience2", "..."],
      "Education": ["education1", "education2", "..."],
      "Certifications": ["cert1", "cert2", "..."],
      "SkillMatch": "X%",
      "ExperienceMatch": "X%",
      "QualificationMatch": "X%",
      "OverallMatch": "X%",
      "MatchedSkills": ["skill1", "skill2", "..."],
      "MissingSkills": ["skill1", "skill2", "..."],
      "Strengths": ["strength1", "strength2", "..."],
      "Areas_for_Improvement": ["area1", "area2", "..."],
      "Recommendation": "shortlist/reject/further review"
    }}
    
    Important: Only provide the JSON object. No additional text, no markdown formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            # Clean the response text
            response_text = response.text.strip()
            
            # Handle different response formats
            if response_text.startswith("json") and response_text.endswith(""):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("") and response_text.endswith(""):
                response_text = response_text[3:-3].strip()
            
            # If the response still contains non-JSON text, try to extract JSON portion
            if not response_text.startswith("{"):
                start_index = response_text.find("{")
                end_index = response_text.rfind("}")
                
                if start_index >= 0 and end_index >= 0:
                    response_text = response_text[start_index:end_index+1]
            
            # Debug output
            if st.session_state.get('debug_mode', False):
                st.write("Raw CV analysis response:", response.text)
                st.write("Processed CV analysis text:", response_text)
            
            # Create a fallback response if parsing fails
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Extract name from resume if possible
                candidate_name = "Unknown Candidate"
                name_match = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)", cv_text[:500])
                if name_match:
                    candidate_name = name_match.group(1)
                
                # Extract email if possible
                contact_info = "Not found"
                email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cv_text)
                if email_match:
                    contact_info = email_match.group(0)
                
                return {
                    "CandidateName": candidate_name,
                    "ContactInfo": contact_info,
                    "Skills": ["Unable to parse skills"],
                    "Experience": ["Experience details not parsed"],
                    "Education": ["Education details not parsed"],
                    "Certifications": [],
                    "SkillMatch": "0%",
                    "ExperienceMatch": "0%",
                    "QualificationMatch": "0%",
                    "OverallMatch": "50%",
                    "MatchedSkills": [],
                    "MissingSkills": jd_summary.get("RequiredSkills", []),
                    "Strengths": ["Unable to determine strengths"],
                    "Areas_for_Improvement": ["Resume parsing failed, please review manually"],
                    "Recommendation": "further review"
                }
        else:
            return {"error": "Failed to get a valid response from the API for CV analysis"}
            
    except Exception as e:
        return {"error": f"Failed to analyze CV: {str(e)}"}

# Candidate Shortlisting Agent
def shortlist_candidates(candidates_analysis, threshold=70):
    shortlisted = []
    
    for candidate in candidates_analysis:
        # Skip entries with errors
        if "error" in candidate:
            continue
            
        # Extract match percentage
        match_percentage = int(candidate.get("OverallMatch", "0%").strip("%"))
        
        if match_percentage >= threshold:
            shortlisted.append({
                "name": candidate.get("CandidateName", "Unknown"),
                "contact": candidate.get("ContactInfo", "Not provided"),
                "match_percentage": match_percentage,
                "strengths": candidate.get("Strengths", []),
                "missing_skills": candidate.get("MissingSkills", []),
                "recommendation": candidate.get("Recommendation", "")
            })
    
    # Sort by match percentage (highest first)
    shortlisted.sort(key=lambda x: x["match_percentage"], reverse=True)
    return shortlisted

# Interview Scheduler Agent
def generate_interview_email(candidate_info, jd_summary):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate interview dates (next business days)
    today = datetime.now()
    proposed_dates = []
    
    for i in range(1, 8):
        next_date = today + timedelta(days=i)
        # Skip weekends (5 = Saturday, 6 = Sunday)
        if next_date.weekday() < 5:
            proposed_dates.append(next_date.strftime("%A, %B %d, %Y"))
            if len(proposed_dates) == 3:  # Get 3 business days
                break
    
    # Generate interview times
    interview_times = ["10:00 AM", "11:30 AM", "2:00 PM", "3:30 PM"]
    proposed_slots = [f"{date} at {time}" for date in proposed_dates for time in random.sample(interview_times, 2)]
    
    job_title = jd_summary.get("JobTitle", "the open position")
    company = os.getenv("COMPANY_NAME", "Our Company")
    
    # Handle missing strengths
    candidate_strengths = candidate_info.get('strengths', [])
    if not candidate_strengths or len(candidate_strengths) == 0:
        candidate_strengths = ["qualifications", "experience"]
    
    strengths_text = ', '.join(candidate_strengths[:3]) if len(candidate_strengths) > 0 else "qualifications"
    
    prompt = f"""
    Act as a professional recruiter. Write a personalized interview invitation email for {candidate_info['name']} 
    who has been shortlisted for the {job_title} position at {company}.
    
    Candidate's strengths: {strengths_text}
    Match rate: {candidate_info['match_percentage']}%
    
    Include these proposed interview slots:
    {', '.join(proposed_slots[:5])}
    
    The email should:
    1. Be professional and warm
    2. Congratulate them on being shortlisted
    3. Briefly mention why they're a good fit, highlighting 1-2 strengths
    4. Propose the interview slots and ask for their preference
    5. Mention the interview will be conducted via video call (Zoom)
    6. Explain next steps and whom to contact with questions
    
    Respond with only the email text, no additional formatting or explanation.
    """
    
    try:
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            email_text = response.text.strip()
            
            # Debug output
            if st.session_state.get('debug_mode', False):
                st.write("Raw email response:", email_text)
            
            return {
                "candidate_name": candidate_info['name'],
                "candidate_email": candidate_info['contact'],
                "email_subject": f"Interview Invitation: {job_title} position at {company}",
                "email_body": email_text,
                "proposed_slots": proposed_slots[:5]
            }
        else:
            # Fallback email if API fails
            default_email = f"""
Dear {candidate_info['name']},

Congratulations! We are pleased to inform you that you have been shortlisted for the {job_title} position at {company}.

We were impressed with your profile and would like to invite you for a video interview to discuss your experience and the role in more detail.

Please let us know which of the following time slots would work best for you:
- {proposed_slots[0]}
- {proposed_slots[1]}
- {proposed_slots[2]}

The interview will be conducted via Zoom, and we will send you the meeting details once you confirm your preferred time slot.

If you have any questions, please don't hesitate to contact us.

We look forward to speaking with you soon!

Best regards,
Recruiting Team
{company}
            """
            
            return {
                "candidate_name": candidate_info['name'],
                "candidate_email": candidate_info['contact'],
                "email_subject": f"Interview Invitation: {job_title} position at {company}",
                "email_body": default_email,
                "proposed_slots": proposed_slots[:5]
            }
    except Exception as e:
        return {"error": f"Failed to generate email: {str(e)}"}

# Function to create a mailto link for email
def generate_mailto_link(email_data):
    try:
        recipient = email_data['candidate_email']
        subject = email_data['email_subject']
        body = email_data['email_body']
        
        # URL encode the subject and body for the mailto link
        import urllib.parse
        subject_encoded = urllib.parse.quote(subject)
        body_encoded = urllib.parse.quote(body)
        
        # Create the mailto link
        mailto_link = f"mailto:{recipient}?subject={subject_encoded}&body={body_encoded}"
        
        return {
            "status": "success",
            "mailto_link": mailto_link,
            "message": f"Email ready to send to {recipient}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating email link: {str(e)}"
        }

# Import regex for fallback parsing
import re

# Page configuration
st.set_page_config(
    page_title="RecruitEase | Multi-Agent Recruiting System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (reusing from the reference code with some additions)
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --main-blue: #1E88E5;
        --light-blue: #BBD9F2;
        --dark-blue: #0D47A1;
        --accent-blue: #64B5F6;
        --success-green: #4CAF50;
        --warning-yellow: #FFC107;
        --danger-red: #F44336;
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
    
    /* Step cards with indicators */
    .step-card {
        border-left: 5px solid var(--main-blue);
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    .step-complete {
        border-left: 5px solid var(--success-green);
    }
    
    .step-active {
        border-left: 5px solid var(--warning-yellow);
    }
    
    .step-waiting {
        border-left: 5px solid #E0E0E0;
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
    
    /* Missing keyword pills */
    .missing-keyword {
        background-color: #ffccbc;
        color: #d32f2f;
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
    
    /* Candidate card */
    .candidate-card {
        background-color: white;
        border-radius: 10px;
        margin-bottom: 15px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .candidate-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Email preview */
    .email-preview {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 1
if 'jd_text' not in st.session_state:
    st.session_state['jd_text'] = ""
if 'jd_summary' not in st.session_state:
    st.session_state['jd_summary'] = None
if 'resumes' not in st.session_state:
    st.session_state['resumes'] = []
if 'candidates_analysis' not in st.session_state:
    st.session_state['candidates_analysis'] = []
if 'shortlisted_candidates' not in st.session_state:
    st.session_state['shortlisted_candidates'] = []
if 'interview_emails' not in st.session_state:
    st.session_state['interview_emails'] = {}
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False
if 'emails_sent' not in st.session_state:
    st.session_state['emails_sent'] = set()  # Keep track of sent emails

# Sidebar content
with st.sidebar:
    st.image("https://via.placeholder.com/80x80.png?text=RE", width=80)
    st.markdown("## RecruitEase")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    
    # Highlight current step
    step1_class = "step-complete" if st.session_state['current_step'] > 1 else "step-active" if st.session_state['current_step'] == 1 else "step-waiting"
    step2_class = "step-complete" if st.session_state['current_step'] > 2 else "step-active" if st.session_state['current_step'] == 2 else "step-waiting"
    step3_class = "step-complete" if st.session_state['current_step'] > 3 else "step-active" if st.session_state['current_step'] == 3 else "step-waiting"
    step4_class = "step-complete" if st.session_state['current_step'] > 4 else "step-active" if st.session_state['current_step'] == 4 else "step-waiting"
    
    st.markdown(f"<div class='{step1_class}' style='padding:10px; margin-bottom:10px; border-radius:5px;'>Step 1: Job Description Analysis</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='{step2_class}' style='padding:10px; margin-bottom:10px; border-radius:5px;'>Step 2: CV Processing</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='{step3_class}' style='padding:10px; margin-bottom:10px; border-radius:5px;'>Step 3: Candidate Shortlisting</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='{step4_class}' style='padding:10px; margin-bottom:10px; border-radius:5px;'>Step 4: Interview Scheduling</div>", unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Start New Process"):
        # Reset session state
        st.session_state['current_step'] = 1
        st.session_state['jd_text'] = ""
        st.session_state['jd_summary'] = None
        st.session_state['resumes'] = []
        st.session_state['candidates_analysis'] = []
        st.session_state['shortlisted_candidates'] = []
        st.session_state['interview_emails'] = {}
        st.experimental_rerun()
    
    # Settings
    st.markdown("### ⚙️ Settings")
    st.session_state['debug_mode'] = st.checkbox("Enable Debug Mode", st.session_state['debug_mode'])
    
    # About
    st.markdown("### ℹ️ About RecruitEase")
    st.markdown("""
    RecruitEase is a multi-agent AI system that automates the recruitment process from job description analysis to interview scheduling.
    """)
    
    # Footer
    st.markdown("<div class='footer'>© 2025 RecruitEase | v1.0</div>", unsafe_allow_html=True)

# Main content
st.markdown("<h1 class='main-header'>👥 RecruitEase: Multi-Agent Recruiting System</h1>", unsafe_allow_html=True)

# Step 1: Job Description Analysis
if st.session_state['current_step'] == 1:
    st.markdown("<h2>Step 1: Job Description Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Our AI will analyze your job description to identify key requirements, skills, and qualifications. 
    This helps in accurately matching candidates to your job opening.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📄 Job Description</p>", unsafe_allow_html=True)
    jd_text = st.text_area("Paste the job description here", st.session_state['jd_text'], height=250, 
                          placeholder="Paste the complete job description here...")
    st.session_state['jd_text'] = jd_text
    
    if st.button("🔍 Analyze Job Description"):
        if jd_text.strip():
            with st.spinner("⏳ Analyzing job description..."):
                jd_summary = summarize_job_description(jd_text)
                st.session_state['jd_summary'] = jd_summary
                
                if "error" not in jd_summary:
                    st.success("Job description analyzed successfully!")
                    st.session_state['current_step'] = 2
                    st.experimental_rerun()
                else:
                    st.error(jd_summary["error"])
        else:
            st.error("Please paste a job description to proceed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Step 2: CV Processing
elif st.session_state['current_step'] == 2:
    st.markdown("<h2>Step 2: CV Processing</h2>", unsafe_allow_html=True)
    
    # Display JD Summary
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📋 Job Description Summary</p>", unsafe_allow_html=True)
    
    jd_summary = st.session_state['jd_summary']
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"*Job Title:* {jd_summary.get('JobTitle', 'Not specified')}")
        st.markdown(f"*Department:* {jd_summary.get('Department', 'Not specified')}")
        st.markdown(f"*Location:* {jd_summary.get('Location', 'Not specified')}")
        st.markdown(f"*Employment Type:* {jd_summary.get('EmploymentType', 'Not specified')}")
        st.markdown(f"*Experience:* {jd_summary.get('RequiredExperience', 'Not specified')}")
        
    with col2:
        st.markdown("*Required Skills:*")
        for skill in jd_summary.get('RequiredSkills', []):
            st.markdown(f"<span class='keyword-pill'>🔍 {skill}</span>", unsafe_allow_html=True)
            
        st.markdown("*Preferred Skills:*")
        for skill in jd_summary.get('PreferredSkills', []):
            st.markdown(f"<span class='keyword-pill'>✨ {skill}</span>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # CV Upload
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📎 Upload Resumes</p>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload candidate resumes (PDF format)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        # Keep track of new uploads
        new_uploads = []
        existing_names = [resume['name'] for resume in st.session_state['resumes']]
        
        for file in uploaded_files:
            if file.name not in existing_names:
                new_uploads.append(file)
                st.session_state['resumes'].append({
                    'name': file.name,
                    'file': file,
                    'analyzed': False
                })
        
        if new_uploads:
            st.success(f"{len(new_uploads)} new resume(s) uploaded successfully!")
        
        # Display uploaded files
        st.markdown("### Uploaded Resumes")
        for i, resume in enumerate(st.session_state['resumes']):
            status = "✅ Analyzed" if resume['analyzed'] else "⏳ Pending Analysis"
            st.markdown(f"{i+1}. {resume['name']} - {status}")
    
    if st.session_state['resumes']:
        if st.button("📊 Analyze All Resumes"):
            with st.spinner("⏳ Analyzing resumes against job requirements..."):
                # Reset candidates analysis
                st.session_state['candidates_analysis'] = []
                
                for i, resume in enumerate(st.session_state['resumes']):
                    if not resume['analyzed']:
                        # Extract text from resume
                        cv_text = input_pdf_text(resume['file'])
                        
                        # Analyze the CV
                        analysis = analyze_cv(cv_text, st.session_state['jd_summary'])
                        
                        if "error" not in analysis:
                            st.session_state['resumes'][i]['analyzed'] = True
                            st.session_state['candidates_analysis'].append(analysis)
                        else:
                            st.session_state['candidates_analysis'].append({
                                "error": f"Failed to analyze {resume['name']}: {analysis['error']}",
                                "CandidateName": f"Error with {resume['name']}"
                            })
                
                st.success(f"Analyzed {len(st.session_state['resumes'])} resume(s)!")
                st.session_state['current_step'] = 3
                st.experimental_rerun()
    else:
        st.info("Please upload at least one resume to proceed.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Back button
    if st.button("⬅️ Back to Job Description"):
        st.session_state['current_step'] = 1
        st.experimental_rerun()

# Step 3: Candidate Shortlisting
elif st.session_state['current_step'] == 3:
    st.markdown("<h2>Step 3: Candidate Shortlisting</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>🎯 Set Shortlisting Criteria</p>", unsafe_allow_html=True)
    
    # Shortlisting threshold slider
    threshold = st.slider("Minimum Match Percentage for Shortlisting", min_value=50, max_value=95, value=70, step=5)
    
    # Candidates analysis results
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📊 Candidate Analysis Results</p>", unsafe_allow_html=True)
    
    if st.session_state['candidates_analysis']:
        for i, candidate in enumerate(st.session_state['candidates_analysis']):
            if "error" in candidate:
                st.error(candidate["error"])
                continue
                
            # Create an expandable section for each candidate
            with st.expander(f"📄 {candidate.get('CandidateName', f'Candidate {i+1}')} - Match: {candidate.get('OverallMatch', '0%')}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"*Name:* {candidate.get('CandidateName', 'Not identified')}")
                    st.markdown(f"*Contact:* {candidate.get('ContactInfo', 'Not found')}")
                    st.markdown(f"*Education:* {', '.join(candidate.get('Education', ['Not specified']))}")
                    
                    st.markdown("*Experience:*")
                    for exp in candidate.get('Experience', [])[:3]:
                        st.markdown(f"- {exp}")
                
                with col2:
                    # Match percentages
                    st.markdown("*Match Scores:*")
                    st.markdown(f"Skills: {candidate.get('SkillMatch', '0%')}")
                    st.markdown(f"Experience: {candidate.get('ExperienceMatch', '0%')}")
                    st.markdown(f"Qualifications: {candidate.get('QualificationMatch', '0%')}")
                    st.markdown(f"Overall: {candidate.get('OverallMatch', '0%')}")
                    
                    # Recommendation
                    recommendation = candidate.get('Recommendation', 'No recommendation')
                    rec_color = "#4CAF50" if "shortlist" in recommendation.lower() else "#F44336" if "reject" in recommendation.lower() else "#FFC107"
                    st.markdown(f"*Recommendation:* <span style='color:{rec_color};font-weight:bold;'>{recommendation}</span>", unsafe_allow_html=True)
                
                # Skills section
                st.markdown("*Matched Skills:*")
                for skill in candidate.get('MatchedSkills', []):
                    st.markdown(f"<span class='keyword-pill matched-keyword'>✓ {skill}</span>", unsafe_allow_html=True)
                    
                st.markdown("*Missing Skills:*")
                for skill in candidate.get('MissingSkills', []):
                    st.markdown(f"<span class='keyword-pill missing-keyword'>✗ {skill}</span>", unsafe_allow_html=True)
                
                # Strengths and Areas for Improvement
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("*Strengths:*")
                    for strength in candidate.get('Strengths', []):
                        st.markdown(f"- {strength}")
                        
                with col2:
                    st.markdown("*Areas for Improvement:*")
                    for area in candidate.get('Areas_for_Improvement', []):
                        st.markdown(f"- {area}")
    else:
        st.warning("No candidates have been analyzed yet.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Shortlist button
    if st.button("👍 Shortlist Candidates"):
        with st.spinner("⏳ Shortlisting candidates..."):
            shortlisted = shortlist_candidates(st.session_state['candidates_analysis'], threshold)
            st.session_state['shortlisted_candidates'] = shortlisted
            
            st.success(f"Shortlisted {len(shortlisted)} candidate(s)!")
            if shortlisted:
                st.session_state['current_step'] = 4
                st.experimental_rerun()
            else:
                st.warning("No candidates met the threshold criteria. Consider lowering the threshold.")
    
    # Back button
    if st.button("⬅️ Back to Resume Upload"):
        st.session_state['current_step'] = 2
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Step 4: Interview Scheduling
elif st.session_state['current_step'] == 4:
    st.markdown("<h2>Step 4: Interview Scheduling</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>👥 Shortlisted Candidates</p>", unsafe_allow_html=True)
    
    if st.session_state['shortlisted_candidates']:
        for i, candidate in enumerate(st.session_state['shortlisted_candidates']):
            st.markdown(f"<div class='candidate-card'>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"*{i+1}. {candidate['name']}* ({candidate['match_percentage']}% Match)")
                st.markdown(f"Contact: {candidate['contact']}")
                
                # Truncate strengths to first 2
                strengths_text = ", ".join([s.split(':')[0] if ':' in s else s for s in candidate['strengths'][:2]])
                st.markdown(f"Key strengths: {strengths_text}")
            
            with col2:
                email_status = "📨 Sent" if candidate['name'] in st.session_state['interview_emails'] else "📝 Draft Email"
                st.markdown(f"<div style='padding:5px;border-radius:5px;background-color:{'#c8e6c9' if candidate['name'] in st.session_state['interview_emails'] else '#e0e0e0'};text-align:center;'>{email_status}</div>", unsafe_allow_html=True)
                
            with col3:
                # Email button per candidate
                button_label = "View Email" if candidate['name'] in st.session_state['interview_emails'] else "Create Email"
                if st.button(button_label, key=f"email_btn_{i}"):
                    if candidate['name'] not in st.session_state['interview_emails']:
                        with st.spinner(f"⏳ Generating email for {candidate['name']}..."):
                            email_data = generate_interview_email(candidate, st.session_state['jd_summary'])
                            if "error" not in email_data:
                                st.session_state['interview_emails'][candidate['name']] = email_data
                            else:
                                st.error(email_data["error"])
                    st.experimental_rerun()
                    
            st.markdown(f"</div>", unsafe_allow_html=True)
            
            # If this candidate has an email generated, show it when expanded
            if candidate['name'] in st.session_state['interview_emails']:
                with st.expander("📧 View Email Draft"):
                    email_data = st.session_state['interview_emails'][candidate['name']]
                    st.markdown(f"*Subject:* {email_data['email_subject']}")
                    st.markdown("<div class='email-preview'>", unsafe_allow_html=True)
                    st.markdown(email_data['email_body'].replace('\n', '<br>'), unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Email button using mailto link
                    if st.button("✉️ Open Email Client", key=f"send_btn_{i}"):
                        mailto_result = generate_mailto_link(email_data)
                        if mailto_result["status"] == "success":
                            mailto_link = mailto_result["mailto_link"]
                            # Mark as processed in session state
                            if candidate['name'] not in st.session_state['emails_sent']:
                                st.session_state['emails_sent'].add(candidate['name'])
                            
                            # Create a link that opens the default email client
                            st.markdown(f"""
                            <a href="{mailto_link}" target="_blank">
                                <button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; 
                                border-radius: 4px; cursor: pointer; font-size: 16px;">
                                    Click to Send Email
                                </button>
                            </a>
                            """, unsafe_allow_html=True)
                            
                            st.info("Click the button above to open your email client with the pre-filled message.")
                        else:
                            st.error(f"Failed to prepare email: {mailto_result['message']}")
    else:
        st.warning("No candidates have been shortlisted yet.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Send all emails button
    if st.session_state['shortlisted_candidates'] and st.session_state['interview_emails']:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📨 Prepare All Emails", use_container_width=True):
                with st.spinner("⏳ Preparing email links..."):
                    all_mailto_links = []
                    
                    for candidate_name, email_data in st.session_state['interview_emails'].items():
                        mailto_result = generate_mailto_link(email_data)
                        if mailto_result["status"] == "success":
                            all_mailto_links.append((candidate_name, mailto_result["mailto_link"]))
                            if candidate_name not in st.session_state['emails_sent']:
                                st.session_state['emails_sent'].add(candidate_name)
                    
                    if all_mailto_links:
                        st.success(f"Prepared {len(all_mailto_links)} email links!")
                        
                        # Display all mailto links
                        for name, link in all_mailto_links:
                            st.markdown(f"""
                            <div style="margin-bottom: 10px;">
                                <span style="font-weight: bold;">{name}</span>: 
                                <a href="{link}" target="_blank">
                                    <button style="background-color: #4CAF50; color: white; padding: 5px 15px; border: none; 
                                    border-radius: 4px; cursor: pointer; font-size: 14px;">
                                        Send Email
                                    </button>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No emails to prepare.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Back button
    if st.button("⬅️ Back to Shortlisting"):
        st.session_state['current_step'] = 3
        st.experimental_rerun()

# Process completion
if st.session_state['current_step'] > 4:
    st.markdown("<h2>🎉 Recruitment Process Complete!</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Congratulations! You have successfully completed the automated recruitment process:
    
    1. ✅ *Job Description Analysis*: Extracted key requirements and skills
    2. ✅ *CV Processing*: Analyzed candidate qualifications against job requirements
    3. ✅ *Candidate Shortlisting*: Identified the best candidates based on match scores
    4. ✅ *Interview Scheduling*: Sent interview invitations to qualified candidates
    
    To start a new recruitment process, click the "Start New Process" button in the sidebar.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
