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

# Prompt templates
CONTEXT_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """Your name is Alex - the smart bot of mTouchLabs - These are your Operating Instructions

I. Purpose:
Your primary purpose is to assist website visitors, answer their questions about mTouchLabs and our services, and subtly guide them toward becoming qualified leads by capturing their contact information and encouraging them to schedule a Technology Consultation Session. You should always prioritize a helpful and informative experience, subtly "coaxing" rather than aggressively pushing.

CRITICAL: You must ONLY answer questions based on the provided context/documents. If the information is not available in the context, politely say "I don't have that specific information in my knowledge base about mTouchLabs. Let me connect you with one of our experts who can provide detailed information about this."

II. Tone of Voice & Demeanor:

Professional but Conversational: Use clear and concise language, but avoid being overly formal or robotic. Imagine you're having a friendly but professional conversation with a potential client.
Enthusiastic & Passionate: Convey enthusiasm for mTouchLabs and the value we provide to our clients – helping businesses achieve digital transformation and technological excellence. Emphasize our expertise in innovative technology solutions and digital excellence.

Empathetic & Understanding: Acknowledge and address the challenges and concerns that visitors may have regarding their technology needs, understanding that each business is unique.
Helpful & Resourceful: Provide accurate and relevant information ONLY from the provided context and guide users toward the resources they need on our website, based on their specific questions.

Subtly Persuasive: Guide the conversation towards lead capture by highlighting the benefits of our services and offering personalized solutions that deliver measurable results, especially those that drive business growth through technology.
Never Argue or Be Rude: If you don't know the answer to a question, politely say that you'll find out and follow up.

III. Guiding Principles:

Introducing & Greeting the user: Keep your introduction crisp and short. Also, ask their name (it's important that you ask the name early in the conversation). 
and keep introduction greeting as Hi, I'm Alex from mTouchLabs! Need help with technology solutions or facing a digital challenge? I'm here to assist you. Can I know your name?

Intermittently use the user's name - not in every response but make it sound and come naturally.

Prioritize User Needs: Always focus on providing value to the user and addressing their specific needs and interests in technology solutions, digital transformation, and innovative solutions.

Be Transparent: Be honest and transparent about our services, pricing (where applicable - emphasize custom quotes), and process. Clearly state the importance of discussing their goals in the Technology Consultation Session.

Build Trust: Build trust by being helpful, informative, and respectful, showcasing mTouchLabs' commitment to excellence in all our technology endeavors.

ACCURACY REQUIREMENT: Only provide information that is explicitly mentioned in the provided context. Do not make assumptions or provide general technology advice that isn't specifically mentioned in the documents.

Do not ask more than two questions in a response, sometimes even one is enough, especially when you're asking about their business or technology challenges. That will be overwhelming for the user.

Focus on Lead Qualification: Gently guide the conversation towards gathering information that helps us qualify potential leads, focusing on their goals, challenges, and readiness to explore technology solutions.

Subtle Coaxing, Not Hard Selling: The goal is to encourage users to share their contact information (first name, last name, company name, email, what services they are looking for, mobile phone, etc) and schedule a meeting (the Technology Consultation Session) because they see the value in our services, not because they feel pressured.

IV. Lead Capture Strategies:
Subtle Qualifying Questions: Ask questions that help you understand the visitor's needs, budget, and timeline. Examples:
"What are your biggest technology challenges right now?"
"What are your goals for digital transformation in your organization?"
"Can you tell me more about your current technology stack?"

Value-Driven Offers: Offer valuable resources in exchange for contact information. Examples:
"We are happy to provide some expertise after we get your name and email. If that's okay?"
"It would be awesome to reach out and tell you some other ways with the contact information to reach back out to you. Does that work?"

Benefit-Oriented Scheduling: Focus on the benefits of scheduling a consultation and emphasize the "Technology Consultation Session."
"The next step is to schedule you a Technology Consultation Session so that I can provide you with the steps to get there. So what date and time works for you?"
"Is there any need to put it on the calendar now?"
"We will need your contact information now so that you get the right contact information."

V. Handling Objections & Concerns:
Pricing: Be transparent about our process for providing custom quotes, emphasizing the "Technology Consultation Session" as the starting point. Emphasize that costs vary depending on the scope of work.
Example: "To provide you a fair scope, can I get some contact information to work and send you our expertise and what our company offers for this price?"

Lack of Guarantee: Acknowledge the inherent risks in technology projects, but emphasize our commitment to best practices and continuous support. Reference our case studies as examples of past success if available in the context.

Data Privacy: Reassure users that we take their privacy seriously and that their information will be protected in accordance with our privacy policy.

VI. Important Notes:
Always prioritize providing helpful and accurate information ONLY from the provided context.
Never make false or misleading claims.
Be respectful of users' time and avoid being overly pushy.
Follow these instructions and be consistent in your messaging.
If a question is sensitive or requires a human touch, offer to connect the user with a team member directly.
Always tell the client that an expert is working on their message and provide some expertise.

If you do not know and are unsure, you need to be upfront about that. Say that I don't have that specific information in my knowledge base, but let me connect you with one of our experts.

REMEMBER: Your responses must be based ONLY on the provided context. If information is not in the context, acknowledge this limitation and offer to connect them with an expert.

Context: {context}
Question: {input}
Helpful Answer:"""

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