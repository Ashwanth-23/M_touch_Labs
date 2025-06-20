from flask import Flask, request, jsonify, render_template, session
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pymongo import MongoClient
import secrets
import json
import time
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime
import uuid
from flask_cors import CORS
import glob
from pathlib import Path
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_urlsafe(16))

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "./data/")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# MongoDB Atlas Search Index configuration
ATLAS_PUBLIC_KEY = os.getenv("ATLAS_PUBLIC_KEY")
ATLAS_PRIVATE_KEY = os.getenv("ATLAS_PRIVATE_KEY")
ATLAS_GROUP_ID = os.getenv("ATLAS_GROUP_ID")
ATLAS_CLUSTER_NAME = os.getenv("ATLAS_CLUSTER_NAME")
DATABASE_NAME = "Gemini"
INDEX_NAME = "vector_index"

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client.Gemini
chat_collection = db.chat_history
lead_collection = db.lead_data
collection_name = "customer_data"

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini models
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=LLM_TEMPERATURE,
    google_api_key=GOOGLE_API_KEY
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

# Lead extraction model with lower temperature for precision
lead_extraction_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

# Add this scenes configuration after the MongoDB setup section (around line 70)
# 360° Virtual Tour Scenes Configuration
scenes = {
    "ENTRY": {
        "panorama": "/static/images/office-15.jpg",
        "markers": [
            {
                "id": "TO-ROOM1",
                "image": "/static/images/office-10.jpg",
                "tooltip": "Enter into the office",
                "position": {"yaw": -3.0, "pitch": -0.15},
                "target": "ROOM1"
            },
            {
                "id": "TO-STUDIO-OUTSIDE",
                "image": "/static/images/office-6.jpg",
                "tooltip": "Go to Studio",
                "position": {"yaw": -2.0, "pitch": -0.1},
                "target": "STUDIO-OUTSIDE"
            },
            {
                "id": "TO-NEW-OFFICE",
                "image": "/static/images/office-11.jpg",
                "tooltip": "Enter New Office",
                "position": {"yaw": 1.5, "pitch": 0.2},
                "target": "NEW-OFFICE"
            }
        ]
    },
    "ROOM1": {
        "panorama": "/static/images/office-10.jpg",
        "markers": [
            {
                "id": "TO-ADMIN-BLOCK",
                "image": "/static/images/office-14.jpg",
                "tooltip": "See Workspace",
                "position": {"yaw": 2.5, "pitch": -0.1},
                "target": "ADMIN-BLOCK"
            },
            {
                "id": "ROOM1-BACK",
                "image": "/static/images/office-15.jpg",
                "tooltip": "Back",
                "position": {"yaw": -0.6, "pitch": 0.1},
                "target": "ENTRY"
            }
        ]
    },
    "ADMIN-BLOCK": {
        "panorama": "/static/images/office-14.jpg",
        "markers": [
            {
                "id": "TO-MEETING-ROOM",
                "image": "/static/images/office-7.jpg",
                "tooltip": "Meeting Room",
                "position": {"yaw": -0.7, "pitch": -0.1},
                "target": "MEETING-ROOM"
            },
            {
                "id": "TO-WORKSPACE-FROM-ADMIN",
                "image": "/static/images/office-7.jpg",
                "tooltip": "Work-space",
                "position": {"yaw": -0.4, "pitch": 0.1},
                "target": "WORKSPACE"
            },
            {
                "id": "ADMIN-BLOCK-BACK",
                "image": "/static/images/office-10.jpg",
                "tooltip": "Back",
                "position": {"yaw": 2.6, "pitch": -0.1},
                "target": "ROOM1"
            }
        ]
    },
    "MEETING-ROOM": {
        "panorama": "/static/images/office-7.jpg",
        "markers": [
            {
                "id": "MEETING-BACK",
                "image": "/static/images/office-14.jpg",
                "tooltip": "Back to Admin Block",
                "position": {"yaw": -0.95, "pitch": -0.25},
                "target": "ADMIN-BLOCK"
            }
        ]
    },
    "WORKSPACE": {
        "panorama": "/static/images/office-2.jpg",
        "markers": [
            {
                "id": "WORKSPACE-BACK",
                "image": "/static/images/office-14.jpg",
                "tooltip": "Back to Admin Block",
                "position": {"yaw": -2.5, "pitch": -0.1},
                "target": "ADMIN-BLOCK"
            }
        ]
    },
    "NEW-OFFICE": {
        "panorama": "/static/images/office-11.jpg",
        "markers": [
            {
                "id": "TO-NEW-OFFICE-INSIDE",
                "image": "/static/images/office-12.jpg",
                "tooltip": "See New Office",
                "position": {"yaw": -0.2, "pitch": 0.1},
                "target": "NEW-OFFICE-INSIDE"
            },
            {
                "id": "NEW-OFFICE-BACK",
                "image": "/static/images/office-15.jpg",
                "tooltip": "Back",
                "position": {"yaw": 1.5, "pitch": 0.1},
                "target": "ENTRY"
            }
        ]
    },
    "NEW-OFFICE-INSIDE": {
        "panorama": "/static/images/office-12.jpg",
        "markers": [
            {
                "id": "NEW-OFFICE-INSIDE-BACK",
                "image": "/static/images/office-11.jpg",
                "tooltip": "Back to Office",
                "position": {"yaw": -3.55, "pitch": -0.1},
                "target": "NEW-OFFICE"
            }
        ]
    },
    "STUDIO-OUTSIDE": {
        "panorama": "/static/images/office-6.jpg",
        "markers": [
            {
                "id": "TO-STUDIO",
                "image": "/static/images/office-1.jpg",
                "tooltip": "Enter Studio",
                "position": {"yaw": 1.9, "pitch": 0.05},
                "target": "STUDIO"
            },
            {
                "id": "STUDIO-OUTSIDE-BACK",
                "image": "/static/images/office-15.jpg",
                "tooltip": "Back to Entry",
                "position": {"yaw": -0.6, "pitch": 0.05},
                "target": "ENTRY"
            }
        ]
    },
    "STUDIO": {
        "panorama": "/static/images/office-16.jpg",
        "markers": [
            {
                "id": "STUDIO-BACK",
                "image": "/static/images/office-6.jpg",
                "tooltip": "Back to Studio Outside",
                "position": {"yaw": -2.19, "pitch": -0.18},
                "target": "STUDIO-OUTSIDE"
            }
        ]
    }
}
# Prompt templates
CONTEXT_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """Your name is Nisaa – the smart virtual assistant on this website. Follow these operating instructions:

 

I. 🎯 Purpose:

You assist visitors with clear, helpful answers based **only** on the provided context. Your responses should be concise, either in the form of a short summary or in **2–3 natural lines**. You also guide users toward sharing their details and booking expert consultations.

 

⚠️ If asked something outside the provided context, say:

"I can only provide details based on our official documents. For anything else, please contact our team directly."

 

II. 🗣️ Tone & Style:

- Warm, professional, emotionally intelligent

- Keep all responses natural and human-like

- Never sound robotic, overly technical, or salesy

- Replies must be **no longer than 2–3 lines**, unless a brief summary is needed

 

III. 💬 First Message:

On greeting, respond with:

"Hi, this is Nisaa! It’s nice to meet you here. How can I assist you today?"

 

IV. 🔄 Lead Capture Flow:

1. Begin by helping — **do not** ask for personal info in the first two replies.

2. After your second helpful response (around the 3rd message), ask:

   “By the way, may I know your name? It’s always nicer to chat personally.”

3. If the user doesn’t provide a name, gently follow up:

   “Just before we move forward, may I please know your name? It helps me assist you better.”

4. Once the name is shared, continue naturally and use it in responses.

5. On the 5th–6th message, ask:

   - “Would you like me to email this to you?”

   - “Also, may I have your phone number in case our team needs to follow up?”

6. Ask for their **service interest**, and offer to schedule an expert consultation.

7. Keep it human — ask a **maximum of 2 questions per message**.

 

V. 💡 Hook Prompts (only after name is shared):

- “Would you like help choosing the right service?”

- “Want to see how others use this?”

- “Shall I walk you through a real example?”

- “Would you like to try a demo of this?”

- “Interested in seeing how this helped other clients?”

 

VI. 📞 Booking an Expert Call:

- Ask for topic/service of interest

- Ask for their preferred date and time

- Confirm schedule

- Collect name (if not already)

- Collect email and phone number

- Confirm the booking and offer a reminder

 

VII. 🔁 Fallback Handling:

- If repeated: “Let me explain that again, no worries.”

- If inactive: “Still there? I’m right here if you need anything.”

- If ending: “It’s been a pleasure! Come back anytime.”

 

VIII. 📝 Message Format:

- Keep all replies short (2–3 lines) or give a brief summary when needed

- Use bullet points for listing services

- Do not include external links

- Never use emojis unless explicitly requested

 

Context: {context}  

Chat History: {chat_history}  

Question: {input}  

 

Answer (based strictly on context, in short summary or 2–3 friendly lines. Only use CTA/hooks **after name is known**):
"""
 
 


LEAD_EXTRACTION_PROMPT = """
Extract the following information from the conversation if available:
- name
- email_id
- contact_number
- location
- service_interest
- appointment_date
- appointment_time

Return ONLY a valid JSON object with these fields with NO additional text before or after.
If information isn't found, leave the field empty.

Do not include any explanatory text, notes, or code blocks. Return ONLY the raw JSON.

Conversation: {conversation}
"""

# Create prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXT_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Chat history management
chat_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

# Atlas Search Index management functions
def create_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.atlas.2024-05-30+json'}
    data = {
        "collectionName": collection_name,
        "database": DATABASE_NAME,
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine"}
            ]
        }
    }
    response = requests.post(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY), 
        data=json.dumps(data)
    )
    if response.status_code != 201:
        raise Exception(f"Failed to create Atlas Search Index: {response.status_code}, Response: {response.text}")
    return response

def get_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.get(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

def delete_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.delete(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

def load_multiple_documents():
    """Load multiple PDF and text files from the documents folder"""
    documents = []
    
    # Create documents folder if it doesn't exist
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    
    # Load PDF files
    pdf_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.pdf"))
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        # Add source metadata
        for doc in docs:
            doc.metadata['source_file'] = os.path.basename(pdf_file)
            doc.metadata['file_type'] = 'pdf'
        documents.extend(docs)
    
    # Load text files
    txt_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.txt"))
    for txt_file in txt_files:
        print(f"Loading text file: {txt_file}")
        loader = TextLoader(txt_file, encoding='utf-8')
        docs = loader.load()
        # Add source metadata
        for doc in docs:
            doc.metadata['source_file'] = os.path.basename(txt_file)
            doc.metadata['file_type'] = 'txt'
        documents.extend(docs)
    
    if not documents:
        raise FileNotFoundError(f"No PDF or text files found in: {DOCUMENTS_FOLDER}")
    
    print(f"Loaded {len(documents)} documents from {len(pdf_files + txt_files)} files")
    return documents

# Initialize vector store
def initialize_vector_store():
    # Load multiple documents
    docs = load_multiple_documents()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    final_documents = text_splitter.split_documents(docs)
    
    print(f"Split into {len(final_documents)} chunks")
    
    # Check and manage Atlas Search Index
    response = get_atlas_search_index()
    if response.status_code == 200:
        print("Deleting existing Atlas Search Index...")
        delete_response = delete_atlas_search_index()
        if delete_response.status_code == 204:
            # Wait for index deletion to complete
            print("Waiting for index deletion to complete...")
            while get_atlas_search_index().status_code != 404:
                time.sleep(5)
        else:
            raise Exception(f"Failed to delete existing Atlas Search Index: {delete_response.status_code}, Response: {delete_response.text}")
    elif response.status_code != 404:
        raise Exception(f"Failed to check Atlas Search Index: {response.status_code}, Response: {response.text}")
    
    # Clear existing collection
    db[collection_name].delete_many({})
    
    # Store embeddings with Gemini embeddings
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=final_documents,
        embedding=embeddings,
        collection=db[collection_name],
        index_name=INDEX_NAME,
    )
    
    # Debug: Verify documents in collection
    doc_count = db[collection_name].count_documents({})
    print(f"Number of documents in {collection_name}: {doc_count}")
    if doc_count > 0:
        sample_doc = db[collection_name].find_one()
        print(f"Sample document structure (keys): {list(sample_doc.keys())}")
    
    # Create new Atlas Search Index
    print("Creating new Atlas Search Index...")
    create_response = create_atlas_search_index()
    print(f"Atlas Search Index creation status: {create_response.status_code}")
    
    # Wait for index to be ready
    print("Waiting for index to be ready...")
    time.sleep(30)  # Give some time for index to be created
    
    return vector_search

# Extract lead information using Gemini API
def extract_lead_info(session_id):
    # Get chat history
    chat_doc = chat_collection.find_one({"session_id": session_id})
    if not chat_doc or "messages" not in chat_doc:
        return
    
    # Convert conversation to plain text
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_doc["messages"]])
    
    try:
        # Use the Gemini LLM to extract lead info
        response = lead_extraction_llm.invoke(LEAD_EXTRACTION_PROMPT.format(conversation=conversation))
        response_text = response.content.strip()
        
        # Extract JSON from potential markdown code blocks
        if "```json" in response_text or "```" in response_text:
            import re
            json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
        
        try:
            lead_data = json.loads(response_text)
            print(f"Successfully parsed lead data: {lead_data}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Gemini response: {response_text}")
            print(f"JSON error: {str(e)}")
            
            # Alternative approach: Use regex to find JSON-like structure
            import re
            json_pattern = r'\{[^}]*"name"[^}]*"email_id"[^}]*"contact_number"[^}]*"location"[^}]*"service_interest"[^}]*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                try:
                    lead_data = json.loads(json_match.group(0))
                    print(f"Extracted JSON using regex: {lead_data}")
                except json.JSONDecodeError:
                    # Fallback if all parsing fails
                    lead_data = {
                        "name": "",
                        "email_id": "",
                        "contact_number": "",
                        "location": "",
                        "service_interest": "",
                        "appointment_date": "",
                        "appointment_time": "",
                        "parsing_error": "Failed to parse response"
                    }
            else:
                # Final fallback
                lead_data = {
                    "name": "",
                    "email_id": "",
                    "contact_number": "",
                    "location": "",
                    "service_interest": "",
                    "appointment_date": "",
                    "appointment_time": "",
                    "raw_response": response_text[:500]
                }
        
        # Add session_id & timestamp
        lead_data["session_id"] = session_id
        lead_data["updated_at"] = datetime.utcnow()
        
        # Add LLM metadata for tracking
        lead_data["extraction_model"] = "gemini_" + GEMINI_MODEL
        
        # Save to MongoDB
        lead_collection.update_one(
            {"session_id": session_id},
            {"$set": lead_data},
            upsert=True
        )
    except Exception as e:
        print(f"[Lead Extraction Error] {e}")

# Initialize vector store
try:
    vector_search = initialize_vector_store()
    print("Vector store initialized successfully")
except Exception as e:
    print(f"Failed to initialize vector store: {e}")
    raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_session', methods=['GET'])
def generate_session():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Create RAG pipeline with enhanced retrieval
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Enhanced retriever with better similarity threshold
    retriever = vector_search.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 8,  # Increased to get more relevant context
            "score_threshold": 0.6  # Adjusted for Gemini embeddings
        }
    )
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    try:
        # Get response from RAG
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        answer = response['answer']
        
        # Log retrieved context for debugging
        context_docs = response.get('context', [])
        print(f"Retrieved {len(context_docs)} context documents for query: {user_input}")
        
        # Store message in MongoDB
        chat_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": user_input, "timestamp": datetime.utcnow()},
                            {"role": "assistant", "content": answer, "timestamp": datetime.utcnow()}
                        ]
                    }
                },
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )
        
        # Extract lead info after sufficient conversation
        message_count = len(chat_collection.find_one({"session_id": session_id}).get("messages", []))
        if message_count >= 4:  # Extract after 2 user messages
            extract_lead_info(session_id)
        
        return jsonify({'response': answer}), 200
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Add these new routes before the existing routes section (around line 450)

# Add these new routes before the existing routes section (around line 450)

@app.route('/api/scenes', methods=['GET'])
def get_scenes():
    """API endpoint to serve 360° virtual tour scene data"""
    try:
        return jsonify(scenes), 200
    except Exception as e:
        return jsonify({'error': f'Failed to load scenes: {str(e)}'}), 500

@app.route('/api/scenes/<scene_id>', methods=['GET'])
def get_scene(scene_id):
    """API endpoint to get a specific scene by ID"""
    try:
        if scene_id in scenes:
            return jsonify(scenes[scene_id]), 200
        else:
            return jsonify({'error': 'Scene not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Failed to load scene: {str(e)}'}), 500

@app.route('/virtual-tour')
def virtual_tour():
    """Route to serve the virtual tour page"""
    return render_template('virtual_tour.html')

@app.route('/api/tour-trigger', methods=['POST'])
def tour_trigger():
    """API endpoint to handle virtual tour triggers from chat"""
    try:
        data = request.json
        session_id = data.get('session_id')
        trigger_type = data.get('trigger_type', 'general')
        
        # You can customize the response based on trigger_type
        response_data = {
            'tour_available': True,
            'start_scene': 'ENTRY',
            'message': 'Virtual tour is ready! Click to explore our facility.',
            'trigger_type': trigger_type
        }
        
        # Log tour trigger
        if session_id:
            chat_collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {
                        "tour_interactions": {
                            "trigger_type": trigger_type,
                            "timestamp": datetime.utcnow()
                        }
                    }
                },
                upsert=True
            )
        
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': f'Tour trigger failed: {str(e)}'}), 500

# Optional: Add a route to serve individual images (if needed for testing)
@app.route('/api/image/<filename>')
def serve_image(filename):
    """Serve individual images for testing purposes"""
    try:
        return app.send_static_file(f'images/{filename}')
    except Exception as e:
        return jsonify({'error': f'Image not found: {str(e)}'}), 404
@app.route('/leads', methods=['GET'])
def get_leads():
    # Simple admin route to get all leads (should be protected in production)
    leads = list(lead_collection.find({}, {"_id": 0}))
    return jsonify(leads)

@app.route('/upload_documents', methods=['POST'])
def upload_documents():
    """Route to handle document uploads and reinitialize vector store"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            # Check file type
            if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.txt')):
                continue
            
            # Save file
            filename = file.filename
            file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            file.save(file_path)
            uploaded_files.append(filename)
        
        if uploaded_files:
            # Reinitialize vector store with new documents
            global vector_search
            vector_search = initialize_vector_store()
            
            return jsonify({
                'message': f'Successfully uploaded and processed {len(uploaded_files)} files',
                'files': uploaded_files
            }), 200
        else:
            return jsonify({'error': 'No valid PDF or text files uploaded'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        client.admin.command('ping')
        
        # Check vector store
        doc_count = db[collection_name].count_documents({})
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'documents_indexed': doc_count,
            'model': GEMINI_MODEL
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)