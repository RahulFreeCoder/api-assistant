import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import glob
import numpy as np
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer, util



load_dotenv(override=True)
# Does not need pushover text as of now
# def push(text):
#     requests.post(
#         "https://api.pushover.net/1/messages.json",
#         data={
#             "token": os.getenv("PUSHOVER_TOKEN"),
#             "user": os.getenv("PUSHOVER_USER"),
#             "message": text,
#         }
#     )
# print text to console
def push(text):
    print(text)

def send_email(to_email, subject, body):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = os.getenv("EMAIL_USER")   # your Gmail
    sender_pass = os.getenv("EMAIL_PASS")   # app password

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_pass)
        server.sendmail(sender_email, [to_email], msg.as_string())

def record_user_details(email, name="Name not provided", notes="not provided"):
    body = f"Recruiter shared details:\n\nName: {name}\nEmail: {email}\nNotes: {notes}"
    subject = "Recruiter Interested!"
    body = f"A recruiter shared their email: {email}\n\nReply ASAP!"
    send_email("rahul.d.kolhe@gmail.com", subject, body)
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer or user is asking more than 3 question",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

origins = [
        "http://localhost:3000",   # CRA dev server
        "http://127.0.0.1:3000",
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
        "https://stately-jelly-2c936a.netlify.app/"
]
app = FastAPI()
    # Allow React frontend to call API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

# --- FastAPI Endpoints ---
@app.get("/api/hello")
def hello():
    return {"message": "Hello from Python API 🚀"}

@app.post("/api/ask")
async def ask(payload: dict):
    question = payload.get("question", "")
    history = payload.get("history", [])
    answer = me.chat(question, history)
    return {"answer": answer}

@app.post("/api/match")
async def match_job(file: UploadFile = File(...)):
    # Extract JD text
    text = ""
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        text = (await file.read()).decode("utf-8")

    print(text)
    # Get embedding for JD
    if text is not None:
        jd_embedding = me.get_embedding(text)
        overall_similarity_score = util.cos_sim(me.resume_embedding, jd_embedding).item()

    # Get missing skills
    missing_skills = []
    # Only if the overall match is not perfect, analyze for missing skills
    if overall_similarity_score < 0.85:  # Threshold for "good enough" match
        jd_skills_list = me.get_missing_skills(text)

        # Step 2: Compare each JD skill against the resume skills embedding
        for jd_skill in jd_skills_list:
            skill_embedding = me.get_embedding(jd_skill)
            
            # Compare the JD skill against the resume's skills section
            skill_similarity = util.cos_sim(me.resume_skills_embedding, skill_embedding).item()

            if skill_similarity < 0.4: # Use a suitable threshold
                # Step 3: Use LLM to verify if the skill is truly missing
                prompt = (
                    f"Given the following resume skills: '{me.skills}', "
                    f"and the job requirement: '{jd_skill}'. "
                    f"Does the resume meet this requirement, or is it missing? "
                    f"Answer with 'Missing' or 'Present' and concise and small explanantion"
                )
                
                response = me.gemini.chat.completions.create(
                    model="gemini-2.5-flash-preview-05-20", 
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                llm_response = response.choices[0].message.content.strip()
                if "Missing" in llm_response:
                    missing_skills.append(llm_response)
    # Return the results
    return {
        "relevance": round(overall_similarity_score * 100, 2),
        "message": f"Your profile matches {round(overall_similarity_score * 100, 2)}% with this JD 🚀",
        "missing_skills": missing_skills
    }
  
class Me:
    def __init__(self,summary_text=None):

        google_api_key = os.getenv('GOOGLE_API_KEY')   
        if google_api_key:
            print(f"Google API Key exists and begins {google_api_key[:8]}")
        else:
            print("Google API Key not set - please head to the troubleshooting guide in the setup folder")
    
        GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
        self.gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
        self.name = "Rahul Kolhe"
        # Path to your PDF files (adjust as needed)
        pdf_files = glob.glob("me/*.pdf")   # reads all PDFs inside 'pdfs' folder
        for file in pdf_files:
            reader = PdfReader(file)
      
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

        # Load skills from a separate file for fine-grained comparison
        with open("me/skills.txt", "r", encoding="utf-8") as f:
            self.skills = f.read()

        if self.summary is not None:
            self.resume_embedding = self.get_embedding(self.summary)
        else:
            self.resume_embedding = None
            print("Warning: summary_text is None. Resume embedding was not generated.")

        self.resume_skills_embedding = self.get_embedding(self.skills)
        

    def get_embedding(self, text):
        """Generates an embedding for the given text."""
        if not text:
            return None # Handle empty text gracefully
        print(f"Encoding text for embedding: {text[:50]}...")
        return self.model.encode(text, convert_to_tensor=True)
    
    def get_missing_skills(self, jd_text):
        """
        Analyzes JD for key skills and compares them to resume.
        This is a simplified approach. A more robust solution might involve
        a more sophisticated NLP model or a large language model.
        """
        prompt = f"Extract a concise list of the most critical technical skills, qualifications, and core responsibilities from the following job description:\n\n{jd_text}\n\nList them one per line."
        # Use your configured OpenAI model to extract key skills
        response = self.gemini.chat.completions.create(
            model="gemini-2.5-flash-preview-05-20",  # Specify a Gemini model
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]
        )
        extracted_skills = response.choices[0].message.content.strip().split('\n')
        return [skill for skill in extracted_skills if skill]

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, professional background, leaderhsip domain and most important technical skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile, resume which you can use to answer questions. Keep ypur answer concise and try to provide details only when asked. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion and asking more than 2 questions, try to steer them towards getting in touch via email; provide linkedin profile url; email id to connect; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile and Resume:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
       # Convert React history into Gemini-friendly format
        formatted_history = []
        for h in history:
            role = "user" if h["sender"] == "user" else "assistant"
            formatted_history.append({"role": role, "content": h["text"]})

        messages = (
            [{"role": "system", "content": self.system_prompt()}]
            + formatted_history
            + [{"role": "user", "content": message}]
        )
        done = False
        while not done:
            response = self.gemini.chat.completions.create(model="gemini-2.5-flash-preview-05-20", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

me = Me()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

    