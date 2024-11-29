import os
import json
from flask import Flask, request, render_template, redirect, url_for,flash

import PyPDF2 as pdf
import google.generativeai as genai
import joblib
import re
import pandas as pd
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
app.secret_key = 'ba77ee8863d05e28047c645ee1421a85'

# Load configuration from your specified JSON file
def load_config():
    config_path = r"C:\Users\gunja\flask_job_prediction\useful-atlas-441309-v2-d2ab04cb48c5.json"
    with open(config_path) as f:
        config = json.load(f)
    return config

# Load your Gemini API key from the JSON config
genai.configure(api_key="AIzaSyDIBD78qUuCJl610CxB7vpMCAccZ7J8Oxc")

# Configuration for uploading resumes
app.config['UPLOAD_FOLDER'] = 'uploads/'
count_vectorizer = joblib.load('vectorizer.pkl')  # Update this path

# Load models globally once when the app starts
count_vectorizer = joblib.load('vectorizer.pkl')  # Load the vectorizer
model = joblib.load('model.pkl')  # Load the model

# Helper functions for Gemini integration
def get_gemini_response(resume_text, jd):
    input_prompt = f"""
    Hey, act like a skilled ATS (Applicant Tracking System) with expertise in AWS, software engineering, and data science roles.
    Evaluate the resume based on the provided job description.
    For the output:
    1. Assign the percentage match for the JD based on the resume.
    2. List missing keywords in an array.
    3. Provide a summary explaining why the candidate is eligible or not.

    Resume: {resume_text}
    Description: {jd}

    Return as:
    {{"JD Match": "<match_percentage>%", "MissingKeywords": [<missing_keywords>], "Profile Summary": "<summary>"}}
    """
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_prompt)
    return json.loads(response.text)  # parse the structured output

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Function for matching job description and resume skills
def process_resume(filepath, required_skills):
    skills = extract_skills(filepath)
    matched_skills = set(skills).intersection(set(required_skills))
    success_rate = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
    eligibility = success_rate > 0
    return eligibility, success_rate, matched_skills

def extract_skills(filepath):
    # Placeholder function for extracting skills from the resume (you can use a package or logic to extract skills)
    return []

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')  # Make sure the about.html file exists in the templates folder




@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    description = request.form['description']
    input_text = f"{title} {description}"

    # Vectorize the input text
    input_vectorized = count_vectorizer.transform([input_text])

    # Predict and get probabilities
    prediction_prob = model.predict_proba(input_vectorized)[0]
    fraud_proba = prediction_prob[1]  # Probability of fraudulent

    # Set fraud threshold
    fraud_threshold = 0.1  # Adjust if needed based on model behavior

    # Determine result based on threshold
    if fraud_proba > fraud_threshold:
        result = "This job posting is likely fraudulent."
    else:
        result = "This job posting is likely legitimate."

    # For debugging, you can add a print statement
    print(f"Prediction Probabilities: Legitimate -> {prediction_prob[0]:.2f}, Fraudulent -> {fraud_proba:.2f}")

    return render_template('result.html', result=result)


# Route for resume upload (after 'Next' button)
@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        resume = request.files.get('resume')
        required_skills = request.form.get('required_skills', '').split(',')

        if resume:
            # Ensure the uploads directory exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            filename = secure_filename(resume.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the resume file
            resume.save(filepath)

            # Extract text from the resume
            resume_text = input_pdf_text(filepath)

            # Prepare input for Gemini API
            jd = request.form.get('job_description', '')
            gemini_output = get_gemini_response(resume_text, jd)

            # Extract values from Gemini API response
            jd_match_str = gemini_output.get("JD Match", "0").replace('%', '')
            jd_match = float(jd_match_str) if jd_match_str.isdigit() else 0  # Convert to a number

            missing_keywords = gemini_output.get("MissingKeywords", [])
            profile_summary = gemini_output.get("Profile Summary", "")

            # Generate eligibility message
            eligibility_message = "Your resume is eligible for the job!" if jd_match > 50 else "Your resume is not eligible for the job!"

            # Return the response to the user
            return render_template('resume_result.html', 
                                   eligibility_message=eligibility_message, 
                                   jd_match=f"{jd_match}%",  # display as a percentage
                                   missing_keywords=missing_keywords,
                                   profile_summary=profile_summary)

    required_skills = request.args.get('required_skills', '').split(',')
    return render_template('next.html', required_skills=required_skills)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        flash('Thank you for reaching out! We will get back to you soon.')
        return redirect(url_for('contact'))  # Redirect back to contact page after form submission
    return render_template('contact.html')  # Render the contact page


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
