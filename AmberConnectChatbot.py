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


# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [{'role': 'assistant', 'content': f"Hello! I am Amber Connect Bot. How can i Help you?"}]
if "parameters" not in st.session_state:
    st.session_state["parameters"] = {"AmberAuthToken": None, "CustomStartDate": None, "CustomEndDate": None}

if "user_questions" not in st.session_state:
    st.session_state["user_questions"] = {}

if "final_url" not in st.session_state:
    st.session_state["final_url"] = {}


# Access credentials from Streamlit secrets
AWS_ACCESS_KEY_ID = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["aws"]["AWS_REGION"]
groq_api_key = st.secrets["groq"]["groq_api_key"]

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

# S3 bucket name
bucket_name = "amberconnectdata"

# File names in S3
json_file_key = "chunk.json"
faiss_file_key = "faissindex.index"

# Function to fetch JSON data from S3
def fetch_json_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    json_data = json.loads(response["Body"].read().decode("utf-8"))
    return json_data
    
# Function to fetch FAISS index file as bytes
def fetch_faiss_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    faiss_data = response["Body"].read()
    return faiss_data

def retrieve_document(user_question):
    chunks_data = fetch_json_from_s3(bucket_name, json_file_key)
    faiss_index_data = fetch_faiss_from_s3(bucket_name, faiss_file_key)
    if not faiss_index or not documents:
        return ["No document data available. Please upload a PDF first."]
    query_embedding = model.encode([user_question])
    _, indices = faiss_index.search(np.array(query_embedding), k=3)
    return [documents[idx] for idx in indices[0]]

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


def main():
    st.header("Amber connect chatbot")
    st.sidebar.header('Debug Mode:')
    
    for chat in st.session_state.chat_history:
        if isinstance(chat, tuple):  # Fix tuple format
            role, content = chat
            chat = {"role": role, "content": content}
        st.chat_message(chat["role"]).write(chat["content"])

    user_question = st.chat_input("Enter your question...")
    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})  # Ensure dict format
        st.chat_message("user").write(user_question)

    qn = user_problem(user_question)
    if qn =='yes':
        st.sidebar.info('User has asked a question about this fleet')
        if st.session_state.get("user_questions") is not None:
        
            last_asked_qn = list(st.session_state["user_questions"].values())[-1]
            st.sidebar.subheader(f'This is user question about fleet:')
            st.sidebar.markdown(f'{last_asked_qn}')

            found_chunk = retrieve_document(last_asked_qn)
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
            side_bar(amber_auth_token, custom_start_date,custom_end_date)

            missing_params = f"Please provide {missing_param}."
            prompt = ask_llm(missing_params ,user_question)
            clean_prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)
            
            if not (amber_auth_token and custom_start_date and custom_end_date and final_url):
                st.session_state["chat_history"].append({"role": "assistant", "content": clean_prompt})
                st.chat_message("assistant").write(clean_prompt)
            
        if amber_auth_token and custom_start_date and custom_end_date and final_url :
            
            count_of_keys = len(st.session_state["user_questions"].keys())

            final_url_from_last_question = list(st.session_state["final_url"].values())[-1]
            url = url_finder(final_url_from_last_question)
            final_api_url = replace_url_params(url,amber_auth_token,custom_start_date,custom_end_date)

            if qn =='yes':
                last_asked_qn = list(st.session_state["user_questions"].values())[-1]
                api_data = api_data_fetcher(final_api_url)
                st.sidebar.subheader('Here is the final url:')
                st.sidebar.success(final_api_url)
            elif count_of_keys == 1:
                last_asked_qn = list(st.session_state["user_questions"].values())[-1]
                api_data = api_data_fetcher(final_api_url)
                st.sidebar.subheader('Here is the final url:')
                st.sidebar.success(final_api_url)

            else:
                last_asked_qn = 'None'
                api_data = 'None'
            
            final_message = final_llm(last_asked_qn, api_data,user_question)
            st.session_state["chat_history"].append({"role": "assistant", "content": final_message})
            st.chat_message("assistant").write(final_message)

main()
