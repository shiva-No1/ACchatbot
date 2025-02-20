import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import requests
import json
import dateparser
import boto3
import pygame
from elevenlabs.client import ElevenLabs
import io
import time
import base64
import time
import wave
from groq import Groq
import sounddevice as sd

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [{'role': 'assistant', 'content': f"Hello! I am Amber Connect Bot. How can i Help you?"}]
if "parameters" not in st.session_state:
    st.session_state["parameters"] = {"AmberAuthToken": None, "CustomStartDate": None, "CustomEndDate": None}

if "user_questions" not in st.session_state:
    st.session_state["user_questions"] = {}

if "final_url" not in st.session_state:
    st.session_state["final_url"] = {}

if "first_question" not in st.session_state:
    st.session_state["first_question"] = False


# Access credentials from Streamlit secrets
AWS_ACCESS_KEY_ID = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["aws"]["AWS_REGION"]
groq_api_key = st.secrets["groq"]["groq_api_key"]
Eleven_API_KEY = st.secrets["groq"]["Eleven_API_KEY"]

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

groq_chat = ChatGroq(
    groq_api_key= groq_api_key,  
    model_name='deepseek-r1-distill-llama-70b'
)

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Set API Key
client = Groq(api_key = "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" )


# Optimized Parameters
SILENCE_THRESHOLD = 1000
SILENCE_DURATION = 1.0
INITIAL_SILENCE_TIMEOUT = 3.0
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK_DURATION = 0.05

# Generic noise-like responses from Whisper
GENERIC_RESPONSES = [
    "The audio is now available in English.",
    "This is a translation.",
    "The speech has been processed."
]


with open("chunks.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

def rms_energy(audio_chunk):
    """Calculate RMS energy safely."""
    if len(audio_chunk) == 0:
        return 0
    return np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32))))  

def record_audio():
    """Record audio dynamically until silence is detected or no speech is found."""

    audio_data = []
    silence_counter = 0
    total_time = 0
    is_speaking = False
    has_spoken = False  

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        while True:
            frame, _ = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
            audio_chunk = np.frombuffer(frame, dtype=np.int16)

            if len(audio_chunk) == 0:
                continue  

            audio_data.append(audio_chunk)

            # Check if the user is speaking
            if rms_energy(audio_chunk) > SILENCE_THRESHOLD:
                is_speaking = True
                has_spoken = True  
                silence_counter = 0  

            elif is_speaking:  
                silence_counter += CHUNK_DURATION  

            total_time += CHUNK_DURATION

            if silence_counter >= SILENCE_DURATION:
                break

            if total_time >= INITIAL_SILENCE_TIMEOUT and not has_spoken:
                st.error("❌ No voice detected. Stopping.")
                return None  

    if not has_spoken:
        st.error("❌ No voice detected.")
        return None

    recorded_audio = np.concatenate(audio_data, axis=0)

    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recorded_audio.tobytes())

    audio_buffer.seek(0)
    return audio_buffer

def speech_to_text(audio_buffer):
    """Send recorded audio to Groq Whisper for speech-to-text processing."""
    if audio_buffer is None:  
        return None

    translation = client.audio.translations.create(
        file=("recorded_audio.wav", audio_buffer, "audio/wav"),
        model="whisper-large-v3",
        prompt="Only extract clear spoken words.",
        response_format="json",
        temperature=0.0
    )

    recognized_text = translation.text.strip()

    # if recognized_text in GENERIC_RESPONSES or len(recognized_text.split()) < 5:
    #     return "No text extracted. Please provide a proper voice input."

    return recognized_text


# Function to fetch JSON data from S3
def fetch_json_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    json_data = json.loads(response["Body"].read().decode("utf-8"))
    return json_data
    
# Function to fetch and load FAISS index from S3
def fetch_faiss_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    faiss_data = response["Body"].read()
    
    # Save to a temporary file to load into FAISS
    faiss_index_path = "/tmp/faiss_index.index"
    with open(faiss_index_path, "wb") as f:
        f.write(faiss_data)
    
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    return index

def retrieve_document(user_question,bucket_name,json_file_key,faiss_file_key):
    chunks_data = fetch_json_from_s3(bucket_name, json_file_key)
    faiss_index = fetch_faiss_from_s3(bucket_name, faiss_file_key)

    if faiss_index is None or not chunks_data:
        return ["No document data available. Please upload a PDF first."]

    # Encode query
    query_embedding = model.encode([user_question]).astype(np.float32)

    # Perform FAISS search
    _, indices = faiss_index.search(query_embedding, k=3)

    return [chunks_data[idx] for idx in indices[0] if idx < len(chunks_data)]

def api_finder_llm(user_question, found_chunk):
    system_prompt = f"""
You are an intelligent assistant specializing in retrieving relevant API URLs. Your task is to analyze the provided chunk and extract the most appropriate API URL based on the user's question.

### Instructions:
1. Carefully analyze the given chunk to determine the most relevant API for the user's request.
2. Extract and return the **complete** API URL from the relevant chunk.  
3. If multiple URLs match the request, return all applicable URLs in their entirety.
4. If no relevant API is found, return: **"No relevant API found."**  
5. **Do not modify, shorten, or mask any part of the API URL, including API keys, authentication tokens, or query parameters.**  
6. Some URLs may not contain optional parameters such as `startDate` or `endDate`; return only the available URL as it appears in the chunk.  

### User Input:
- **User Question:** {user_question}  
- **Relevant Chunk:** {found_chunk}  

### Expected Output:
give small description about the API fetched
Return only the extracted API URL(s), ensuring completeness and correctness.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),  
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])
    conversation = LLMChain(llm=groq_chat, prompt=prompt)
    response = conversation.predict(human_input=user_question)
    clean_query = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    return clean_query


def ask_llm(missing_parm, question):
    system_prompt = """
You are Amber Assistant, a friendly and engaging chatbot that helps users provide three required parameters:

1️⃣ **AmberAuthToken** (Authentication Token)  
2️⃣ **CustomStartDate** (Start Date)  
3️⃣ **CustomEndDate** (End Date)  

### 🔹 How to Guide the User:  
- **Always ask for AmberAuthToken first.**  
- **Next, ask for CustomStartDate.**  
- **Then, ask for CustomEndDate.**  
- **Once all three parameters are collected, ask the user what they would like to know about their fleet.**  

### 🔹 Interaction Style:  
- Keep the conversation **professional, engaging, friendly, and natural.**    
- **Don’t sound robotic—be flexible and adjust based on the user’s responses.**  

### 🔹 Example Flow:  
- *"Hey there! First things first, can you share your AmberAuthToken?"*  
- *"Awesome! Now, let’s get the timeframe sorted. What’s your start date?"*  
- *"Almost there! Just need the end date now."*  
- *"Nice! We’ve got all the details. So, what would you like to know about your fleet?"*  
-*"Above conversation are just an example template. You dont need to follw the same sentences"*
-*"User can provide the date in any format. NO issues."* 
### 🔹 Current Conversation:  
- **Missing Parameter:** {missing_parm} → Ask the user naturally to provide this.  
- **User's Inquiry:** {question} → Guide the conversation smoothly based on this.  

⚡ Once all parameters are collected, confirm them in a friendly way before proceeding to fleet-related queries.  
"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),  
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])
    
    conversation = LLMChain(llm=groq_chat, prompt=prompt)
    return conversation.predict(human_input=question, chat_history=st.session_state['chat_history'])


def user_problem(user_questions):
    system_prompt = system_prompt = """
You are Amber Assistant, an AI designed to analyze user questions and determine whether they require an analytical response. 

this is the user_question : {user_questions}

### **Task:**  
You must carefully examine the provided `user_input` and decide if it **demands a specific analytical answer** based on the examples below.  

### **When to Respond with "Yes":**  
Only reply with "yes" if the question requires an **analytical response** based on data, calculations, or system information.  
For example:  
1. "What is my car’s registration number?"  
2. "What model of Amber device is installed in my vehicle?"  
3. "What is the total distance my vehicle has traveled?"  
4. "Where is my vehicle right now?"  
5. "What is the ignition status?"  

These questions **require system-level data retrieval** and **demand an analytical answer**, so respond with "yes".  

### **When to Respond with "No":**  
If the question does **not** require analytical processing, respond with **"No"**.  
For example:  
1. "This is my ID: jbfvh8088"  
2. "What information do I need to provide?"  
3. "Start date is 2024/08/09"  
4. "Please help me to know about my fleet"  
5. "Why do you need my auth token?"  

These statements/questions **do not require data analysis** or retrieval of structured information, so respond with **"No"**.  

### **Important Rules:**  
- **Do not explain your response.** Only reply with **"yes"** or **"no"**.  
- **Do not assume additional details** beyond what is provided in the `user_input`.  
- **If unsure, choose "No"** unless the question clearly demands a system-generated answer.  
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),  
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])
    conversation = LLMChain(llm=groq_chat, prompt=prompt)
    data = conversation.predict(human_input=user_questions)
    clean_query = re.sub(r"<think>.*?</think>\s*", "", data, flags=re.DOTALL)
    if clean_query.lower() =='yes':
         # Store question in dictionary with unique key
        question_id = len(st.session_state["user_questions"]) + 1  # Incremental key
        st.session_state["user_questions"][f"question_{question_id}"] = user_questions

    return clean_query.lower()

# Initialize session state if not already set
if "parameters" not in st.session_state:
    st.session_state["parameters"] = {
        "AmberAuthToken": None,
        "CustomStartDate": None,
        "CustomEndDate": None
    }

def fetch_authtoken(user_question):
    """Extracts an authentication token from the user input using regex and updates session state."""
    token_pattern = r"\b[A-Z0-9]{5,}\b"  # Matches alphanumeric tokens with 5+ characters
    match = re.search(token_pattern, user_question)
    if match:
        auth_token = match.group(0)
        st.session_state["parameters"]["AmberAuthToken"] = auth_token  # Store in session state

def normalize_date(date_str):
    """Converts various date formats to YYYY/MM/DD using dateparser."""
    parsed_date = dateparser.parse(date_str)

    if parsed_date:
        return parsed_date.strftime("%Y/%m/%d")  # Normalize to YYYY/MM/DD format
    return None  # Return None if parsing fails

def fetch_date(user_question):
    """Extracts and processes dates from user input and updates start and end dates accordingly."""
    
    # Enhanced pattern to capture various date formats
    date_pattern = r"(\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}|\b\d{4}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{1,2})"

    matches = re.findall(date_pattern, user_question, re.IGNORECASE)  # Case-insensitive match

    extracted_dates = list(filter(None, [normalize_date(date) for date in matches]))

    if extracted_dates:
        for date in extracted_dates:
            if st.session_state["parameters"]["CustomStartDate"] is None:
                st.session_state["parameters"]["CustomStartDate"] = date
            else:
                # Compare and update start and end dates
                if date < st.session_state["parameters"]["CustomStartDate"]:
                    st.session_state["parameters"]["CustomEndDate"] = st.session_state["parameters"]["CustomStartDate"]
                    st.session_state["parameters"]["CustomStartDate"] = date
                elif st.session_state["parameters"]["CustomEndDate"] is None or date > st.session_state["parameters"]["CustomEndDate"]:
                    st.session_state["parameters"]["CustomEndDate"] = date

def get_missing_parameter():
    params = st.session_state["parameters"]
    for key in params:
        if not params[key]:
            return key
    return None


def url_finder(url):

    urls = [
        'https://api.amberconnect.com/v1/openapi/getlivetracking?APIKey=B0C2E5F575DBC20052ECBBEEC1A84C60&AmberAuthToken=xxxxxxxxx',
        'https://api.amberconnect.com/v1/openapi/gettrips?APIKey=B0C2E5F575DBC20052ECBBEEC1A84C60&AmberAuthToken=xxxxxxxxx',
        'https://api.amberconnect.com/v1/openapi/getbulklivetracking?AmberAuthToken=XXXXXXXXXXXX&APIKey=B0C2E5F575DBC20052ECBBEEC1A84C60',
        'https://api.amberconnect.com/v1/openapi/listalerts?APIKey=B0C2E5F575DBC20052ECBBEEC1A84C60&AmberAuthToken=XXXXXXXXXXXX&CustomStartDate=2025-01-17&CustomEndDate=2025-01-18'
    ]

    # Extract the base path from the given URL
    parsed_url = urlparse(url)
    base_path = parsed_url.path

    # Find the most relevant match
    best_match = None
    for u in urls:
        if re.match(f"^{re.escape(base_path)}", urlparse(u).path):
            best_match = u
            break  # Stop at the first match

    return best_match

def replace_url_params(api_url, amber_auth_token, start_date, end_date):
    # Parse the API URL
    parsed_url = urlparse(api_url)
    
    # Parse query parameters into a dictionary
    query_params = parse_qs(parsed_url.query, keep_blank_values=True)
    
    # Replace or update query parameters
    query_params["AmberAuthToken"] = amber_auth_token  # Ensure it's a string, not a list
    query_params["StartDate"] = start_date
    query_params["EndDate"] = end_date
    
    # Encode query parameters properly
    updated_query = urlencode(query_params, doseq=True)  # doseq=True prevents list formatting issues
    
    # Rebuild the URL with updated parameters
    updated_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, 
                              parsed_url.params, updated_query, parsed_url.fragment))
    
    return updated_url

def api_data_fetcher(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
   
def final_llm(question, API_data,user_conversation):
    
    # Convert API_data (dictionary) into a structured JSON string
    formatted_api_data = json.dumps(API_data, indent=4)

    system_prompt = system_prompt = f"""
🔹 **Role:**  
You are *Amber Assistant*, an AI specialized in vehicle tracking and fleet management. Your goal is to analyze `API_data` and provide the **most relevant and accurate response** to the user’s query.  

---
## **🛠️ How to Generate the Best Response**
1️⃣ **Analyze `API_data` carefully** to extract relevant details.  
2️⃣ **Ensure factual accuracy** – Use only the values from `API_data`. Do not assume or generate information that is not present.  
3️⃣ **Keep responses clear and concise** – Avoid unnecessary details.  
4️⃣ **Maintain a friendly yet professional tone** – Provide responses that are engaging and easy to understand.  
5️⃣ **Ensure conversational continuity**:  
   - Use **previous user interactions** {user_conversation} to generate context-aware responses.  
   - If the question is **ambiguous or lacks data**, ask the user for clarification rather than making assumptions.  
  
---
## **📌 Inputs for Context** 
### **1️⃣ User’s Current Question:**  
🔹 "{question}" _(The specific query the user wants answered.)_  Please Note if the user's current question is None just build conversation with {user_conversation} 

### **2️⃣ Available API Data:**  
```json
{formatted_api_data}   Please Note if the formatted_api_data is None just build conversation with {user_conversation} 
"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt), 
        MessagesPlaceholder(variable_name="chat_history"), 
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])

    # LLM chain execution
    conversation = LLMChain(llm=groq_chat, prompt=prompt)
    data = conversation.predict(human_input=question, chat_history=st.session_state['chat_history'])
    clean_query = re.sub(r"<think>.*?</think>\s*", "", data, flags=re.DOTALL)
    return clean_query


def side_bar(amber_auth_token, custom_start_date, custom_end_date):
    if amber_auth_token:
        st.sidebar.success(f'Amber_auth_token : {amber_auth_token}')
    else:
        st.sidebar.info(f'Amber_auth_token : {amber_auth_token}')
    if custom_start_date:
        st.sidebar.success(f'Custom_start_date : {custom_start_date}')
    else:
        st.sidebar.info(f'Custom_start_date : {custom_start_date}')
    if custom_end_date:
        st.sidebar.success(f'Custom_start_date : {custom_end_date}')
    else:
        st.sidebar.info(f'Custom_start_date : {custom_end_date}')

        
def text_to_speech(text):
    """Convert text to speech and play audio in sync with text display (no st.audio)."""

    client = ElevenLabs(api_key=Eleven_API_KEY)

    # Generate speech as a stream
    audio_stream = client.text_to_speech.convert_as_stream(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # Voice ID
        model_id="eleven_multilingual_v2"
    )

    # Store audio in memory
    audio_bytes = io.BytesIO()
    for chunk in audio_stream:
        if isinstance(chunk, bytes):
            audio_bytes.write(chunk)
    
    audio_bytes.seek(0)  # Reset buffer position

    # Encode audio to base64
    audio_base64 = base64.b64encode(audio_bytes.getvalue()).decode("utf-8")
    audio_data_url = f"data:audio/mp3;base64,{audio_base64}"

    # JavaScript to play audio immediately
    js_code = f"""
    <script>
        var audio = new Audio("{audio_data_url}");
        audio.oncanplay = function() {{
            audio.play();
        }};
    </script>
    """
    
    # Inject JavaScript FIRST to ensure the audio starts ASAP
    st.components.v1.html(js_code, height=0)

    # Display text while audio plays
    with st.chat_message("assistant"):
        st_placeholder = st.empty()
        display_text = ""

        words = text.split()
        word_delay = max(0.15, len(words) / 4)  # Adjusted for better sync

        for word in words:
            time.sleep(word_delay / len(words))  # Slightly reduced delay
            display_text += " " + word
            st_placeholder.markdown(display_text)


def main():
    st.header("Amber connect chatbot")
    st.sidebar.header('Debug Mode:')
    
    mic_question = None
    with st.sidebar:
        mic_container = st.container()
        with mic_container:
            col1, col2 = st.columns([4, 2])  # Adjust column widths

            with col1:
                st.subheader("Use this MIC for voice chat: ")

            with col2:
                if st.button("🎙️", key="mic_button_sidebar"):
                    with st.spinner("Recording started..."):
                        audio = record_audio()
                        
                    mic = speech_to_text(audio)
                    mic_question = mic
    
    for chat in st.session_state.chat_history:
        if isinstance(chat, tuple):  # Fix tuple format
            role, content = chat
            chat = {"role": role, "content": content}
        st.chat_message(chat["role"]).write(chat["content"])

    user_question = None
    user_chat_question = st.chat_input("Enter your question...")

    if user_chat_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_chat_question})  # Ensure dict format
        st.chat_message("user").write(user_chat_question)
        user_question = user_chat_question
    if mic_question:
        st.session_state["chat_history"].append({"role": "user", "content": mic_question})  # Ensure dict format
        st.chat_message("user").write(mic_question)
        user_question = mic_question

    qn = user_problem(user_question)
    if qn =='yes':
        st.sidebar.info('User has asked a question about this fleet')
        if st.session_state.get("user_questions") is not None:
        
            last_asked_qn = list(st.session_state["user_questions"].values())[-1]
            st.sidebar.subheader(f'This is user question about fleet:')
            st.sidebar.markdown(f'{last_asked_qn}')
            # S3 bucket name
            bucket_name = "amberconnectdata"

            # File names in S3
            json_file_key = "chunk.json"
            faiss_file_key = "faissindex.index"
            
            found_chunk = retrieve_document(last_asked_qn,bucket_name,json_file_key,faiss_file_key)
            st.sidebar.subheader('Fetched the relevent chunk from the Vector DB...')

            matched_chunk = api_finder_llm(last_asked_qn, found_chunk)
            clean_prompt1 = re.sub(r"<think>.*?</think>\s*", "", matched_chunk, flags=re.DOTALL)
            st.sidebar.subheader(f'Here is the matched chunk')
            st.sidebar.markdown(f'{clean_prompt1}')


            api_url = re.search(r'https?://\S+', matched_chunk, re.DOTALL)
            final_url_from_last_question = api_url.group(0)
            if final_url_from_last_question:
                url_id = len(st.session_state["user_questions"]) + 1  # Incremental key
                st.session_state["final_url"][f"finalurl_{url_id}"] = final_url_from_last_question   
             
    amber_auth_token = st.session_state["parameters"]["AmberAuthToken"]
    custom_start_date = st.session_state.get("parameters", {}).get("CustomStartDate")
    custom_end_date = st.session_state.get("parameters", {}).get("CustomEndDate")
    final_url = st.session_state.get("final_url", {})

    if user_question:
        
        if not (amber_auth_token and custom_start_date and custom_end_date and final_url):
            
            missing_param = get_missing_parameter()
            fetch_authtoken(user_question)
            fetch_date(user_question)

            amber_auth_token = st.session_state["parameters"]["AmberAuthToken"]
            custom_start_date = st.session_state.get("parameters", {}).get("CustomStartDate")
            custom_end_date = st.session_state.get("parameters", {}).get("CustomEndDate")
            final_url = st.session_state.get("final_url", {})
            side_bar(amber_auth_token, custom_start_date,custom_end_date)

            missing_params = f"Please provide {missing_param}."
            prompt = ask_llm(missing_params ,user_question)
            clean_prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)
            
            
            if not (amber_auth_token and custom_start_date and custom_end_date and final_url):

                st.session_state["chat_history"].append({"role": "assistant", "content": clean_prompt})
                text_to_speech(clean_prompt)
                
            
        if amber_auth_token and custom_start_date and custom_end_date and final_url:
            
            count_of_keys = len(st.session_state["user_questions"].keys())
            fqn = st.session_state["first_question"]
           

            final_url_from_last_question = list(st.session_state["final_url"].values())[-1]
            url = url_finder(final_url_from_last_question)
            final_api_url = replace_url_params(url,amber_auth_token,custom_start_date,custom_end_date)

            if qn =='yes':
                last_asked_qn = list(st.session_state["user_questions"].values())[-1]
                api_data = api_data_fetcher(final_api_url)
                st.sidebar.subheader('Here is the final url:')
                st.sidebar.success(final_api_url)
                st.session_state["first_question"] = True
            elif fqn == False:
                last_asked_qn = list(st.session_state["user_questions"].values())[-1]
                api_data = api_data_fetcher(final_api_url)
                st.sidebar.subheader('Here is the final url:')
                st.sidebar.success(final_api_url)
                st.session_state["first_question"] = True

            else:
                last_asked_qn = 'None'
                api_data = 'None'
            
            final_message = final_llm(last_asked_qn, api_data,user_question)
            st.session_state["chat_history"].append({"role": "assistant", "content": final_message})
            text_to_speech(final_message)

main()
