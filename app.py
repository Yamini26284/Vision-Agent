import streamlit as st
import json
import subprocess
import sys

st.set_page_config(page_title="AI Interview Coach", page_icon="🤖")
st.title("🤖 AI Interview Coach")

# 1. Core Settings
with st.sidebar:
    st.header("Personal Details")
    role = st.text_input("Target Role", placeholder="e.g. Frontend Developer")
    level = st.selectbox("Seniority", ["Junior", "Mid-Level", "Senior", "Lead"])
    num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=3)

# 2. Dynamic Input Method
st.header("Interview Preparation Coach")
input_method = st.radio(
    "How should Agent prepare your interview Questions?",
    ["Paste Job Description", "Upload Resume", "Let AI Choose"],
    horizontal=True
)

# --- DYNAMIC UI LOGIC STARTS HERE ---
content = ""

if input_method == "Paste Job Description":
    content = st.text_area("📄 Paste the Job Description below:", height=250)

elif input_method == "Upload Resume":
    uploaded_file = st.file_uploader("📤 Upload your Resume (PDF)", type=["pdf"])
    if uploaded_file:
        # For the hackathon, we'll label it. 
        # (Optional: Use PyPDF2 here to extract text if you have time)
        content = f"User uploaded a resume for the {role} role."
        st.success("Resume uploaded successfully!")

else:
    st.info("Our Agent will conduct a general technical interview based on your Target Role.")
    content = f"General {level} level interview for {role}."

# --- DYNAMIC UI LOGIC ENDS HERE ---

# 3. Launch Button
if st.button("🚀 Start My Interview"):
    # Validation
    if not role:
        st.error("Please fill in the Target Role and Email in the sidebar.")
    elif input_method == "Paste Job Description" and not content:
        st.error("Please paste a Job Description to proceed.")
    elif input_method == "Upload Resume" and not uploaded_file:
        st.error("Please upload your resume to proceed.")
    else:
        # Save config
        config = {
            "role": role,
            "level": level,
            "content": content,
            "num_questions": num_questions
        }
        with open("interview_config.json", "w") as f:
            json.dump(config, f)
        
        st.success("Agent is ready! Your interview room is opening...")
        subprocess.Popen([sys.executable, "main.py", "run"])