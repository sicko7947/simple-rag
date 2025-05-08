"""
Web Interface for the RAG system using FastAPI
"""
import os
import logging
import tempfile
import uvicorn
import psutil
import time
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Import after environment variables are loaded
from src.agent.cognitive_agent import CognitiveAgent

# Initialize FastAPI app
app = FastAPI(title="RAG System API", description="API for the RAG System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store user-specific agents
conversation_agents: Dict[str, Dict[str, Any]] = {}

# Agent lifecycle settings
AGENT_IDLE_TIMEOUT = int(os.getenv("AGENT_IDLE_TIMEOUT", "1800"))  # 30 minutes by default
AGENT_ABSOLUTE_TIMEOUT = int(os.getenv("AGENT_ABSOLUTE_TIMEOUT", "10800"))  # 3 hours by default
MAX_AGENTS = int(os.getenv("MAX_AGENTS", "100"))  # Maximum number of agents to keep in memory
MEMORY_THRESHOLD = int(os.getenv("MEMORY_THRESHOLD", "80"))  # Memory utilization percentage threshold

def cleanup_agents(force_cleanup: bool = False) -> int:
    """
    Clean up agents based on various criteria:
    - Idle timeout
    - Absolute timeout
    - Memory pressure
    - Max agents limit
    
    Returns:
        Number of agents removed
    """
    current_time = time.time()
    agents_to_remove = []
    
    # Check memory usage if monitoring is enabled
    memory_pressure = False
    try:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > MEMORY_THRESHOLD:
            memory_pressure = True
            logger.warning(f"Memory pressure detected: {memory_percent}% used, exceeding {MEMORY_THRESHOLD}% threshold")
    except Exception as e:
        logger.error(f"Error checking memory usage: {str(e)}")
    
    # First pass: identify agents to remove based on timeouts
    for cid, agent_data in conversation_agents.items():
        # Check for idle timeout
        if current_time - agent_data["last_access"] > AGENT_IDLE_TIMEOUT:
            agents_to_remove.append((cid, "idle timeout"))
            continue
            
        # Check for absolute timeout
        if current_time - agent_data["creation_time"] > AGENT_ABSOLUTE_TIMEOUT:
            agents_to_remove.append((cid, "absolute timeout"))
            continue
    
    # If we're still over the limit or under memory pressure, remove oldest agents
    if (len(conversation_agents) - len(agents_to_remove) > MAX_AGENTS) or memory_pressure or force_cleanup:
        # Sort remaining agents by last access time
        remaining_agents = [(cid, data) for cid, data in conversation_agents.items() 
                            if cid not in [cid for cid, _ in agents_to_remove]]
        remaining_agents.sort(key=lambda x: x[1]["last_access"])
        
        # Calculate how many more to remove
        extra_to_remove = 0
        if len(conversation_agents) - len(agents_to_remove) > MAX_AGENTS:
            extra_to_remove = len(conversation_agents) - len(agents_to_remove) - MAX_AGENTS
            logger.info(f"Need to remove {extra_to_remove} more agents to stay under MAX_AGENTS limit")
        
        # If there's memory pressure, remove at least 10% of agents
        if memory_pressure:
            memory_remove_count = max(1, len(conversation_agents) // 10)
            extra_to_remove = max(extra_to_remove, memory_remove_count)
            logger.info(f"Memory pressure: will remove at least {memory_remove_count} agents")
        
        # If force_cleanup is True, remove an additional 20% of agents
        if force_cleanup:
            force_remove_count = max(1, len(conversation_agents) // 5)
            extra_to_remove = max(extra_to_remove, force_remove_count)
            logger.info(f"Force cleanup: will remove at least {force_remove_count} agents")
        
        # Add oldest agents to removal list
        for i in range(min(extra_to_remove, len(remaining_agents))):
            agents_to_remove.append((remaining_agents[i][0], "resource management"))
    
    # Remove the identified agents
    for cid, reason in agents_to_remove:
        logger.info(f"Removing agent for conversation {cid} due to {reason}")
        del conversation_agents[cid]
    
    if agents_to_remove:
        logger.info(f"Cleaned up {len(agents_to_remove)} agents, {len(conversation_agents)} remaining")
        
    return len(agents_to_remove)

def get_agent_by_conversation(conversation_id: str) -> CognitiveAgent:
    """
    Get or create a CognitiveAgent for the specific conversation
    
    Args:
        conversation_id: Unique ID for the conversation
        
    Returns:
        CognitiveAgent instance for this conversation
    """
    current_time = time.time()
    
    # Run agent cleanup
    cleanup_agents()
    
    # Get or create agent for current conversation
    if conversation_id not in conversation_agents:
        # If we're at max capacity, force a cleanup
        if len(conversation_agents) >= MAX_AGENTS:
            logger.warning(f"Reached maximum agent capacity ({MAX_AGENTS}), forcing cleanup")
            if cleanup_agents(force_cleanup=True) == 0:
                # If no agents were cleaned up, remove the oldest one
                oldest_conv_id = min(conversation_agents.items(), key=lambda x: x[1]["last_access"])[0]
                logger.warning(f"Force removing oldest agent for conversation {oldest_conv_id}")
                del conversation_agents[oldest_conv_id]
        
        logger.info(f"Creating new agent for conversation {conversation_id}")
        conversation_agents[conversation_id] = {
            "agent": CognitiveAgent(),
            "last_access": current_time,
            "creation_time": current_time,
            "request_count": 1
        }
    else:
        logger.info(f"Using existing agent for conversation {conversation_id}")
        conversation_agents[conversation_id]["last_access"] = current_time
        conversation_agents[conversation_id]["request_count"] += 1
        
    return conversation_agents[conversation_id]["agent"]

def get_or_create_agent(request: Request) -> CognitiveAgent:
    """
    Get or create a CognitiveAgent based on request headers
    Maintained for backward compatibility with existing endpoints
    """
    # Extract conversation_id from headers, fallback to user_id if not present
    headers = dict(request.headers)
    
    # Check for conversation_id first
    conversation_id = headers.get('conversation_id') or headers.get('conversation-id')
    
    # If no conversation_id, fall back to user_id
    if not conversation_id:
        user_id = headers.get('user_id') or headers.get('user-id') or headers.get('User-Id') or headers.get('User-ID') or headers.get('userId')
        conversation_id = user_id if user_id else str(uuid.uuid4())
        logger.info(f"No conversation_id in headers, using user_id or generated ID: {conversation_id}")
    
    # Use the conversation-based agent manager
    return get_agent_by_conversation(conversation_id)

# Define request and response models
class MessageContent(BaseModel):
    text: str
    type: str = "text"

class Message(BaseModel):
    role: str
    content: str
    parts: List[MessageContent] = []

class ChatRequest(BaseModel):
    id: str  # Conversation ID
    messages: List[Message]

class QueryRequest(BaseModel):
    id: str  # Conversation ID
    messages: List[Message]
    filter_criteria: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict[str, Any]]
    cognitive_state: Dict[str, Any]
    success: bool

@app.get("/")
async def root(request: Request):
    """Root endpoint"""
    return {"message": "RAG System API is running"}

@app.post("/upload", response_model=Dict[str, Any])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    agent: CognitiveAgent = Depends(get_or_create_agent)
):
    """
    Upload a document to process and store
    """
    try:
        # Read the uploaded file
        file_content = await file.read()
        file_name = file.filename
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            # Process the document
            result = agent.process_document(file_name, file_content)
            
            if not result["success"]:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error processing document: {result.get('error', 'Unknown error')}"
                )
                
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=Dict[str, Any])
async def query(
    query_request: QueryRequest
):
    """
    Process a query and return a response using the frontend message format
    
    Args:
        query_request: The query request containing conversation ID and message history
    
    Returns:
        Response with query results
    """
    try:
        conversation_id = query_request.id
        print(conversation_id)
        
        # Get or create agent for this conversation
        agent = get_agent_by_conversation(conversation_id)
        
        # Extract the latest user message from the messages array
        latest_message = query_request.messages[-1]
        
        if latest_message.role != "user":
            raise HTTPException(
                status_code=400,
                detail="Last message in the request must be from 'user'"
            )
        
        # Extract query text from the message content
        query = latest_message.content
        
        # Convert message history to the format expected by the agent
        chat_history = []
        for message in query_request.messages[:-1]:  # Exclude the latest message which we're processing
            chat_history.append({
                "role": message.role,
                "content": message.content
            })
        
        # Update the agent's conversation history
        agent.conversation_history = chat_history
        
        # Process the query with the agent
        result = agent.answer_query(
            query=query,
            conversation_id=conversation_id,
            filter_criteria=query_request.filter_criteria
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query: {result.get('error', 'Unknown error')}"
            )
        
        # Return result with conversation ID
        result["id"] = conversation_id
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_agent(
    request: Request,
    agent: CognitiveAgent = Depends(get_or_create_agent)
):
    """
    Reset the agent's state
    """
    try:
        agent.reset_state()
        return {"success": True, "message": "Agent state reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_agent_state(
    request: Request,
    agent: CognitiveAgent = Depends(get_or_create_agent)
):
    """
    Get the current state of the agent
    """
    try:
        state = agent.get_agent_state()
        return state
        
    except Exception as e:
        logger.error(f"Error getting agent state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-id")
async def get_user_id(request: Request):
    """
    Get or generate a user ID for the client
    """
    headers = dict(request.headers)
    
    # Check for common variations of the user_id header (same logic as in get_or_create_agent)
    user_id = headers.get('user_id') or headers.get('user-id') or headers.get('User-Id') or headers.get('User-ID') or headers.get('userId')
    
    if not user_id:
        user_id = str(uuid.uuid4())
        logger.info(f"Generated new user_id: {user_id}")
    else:
        logger.info(f"Using existing user_id from header: {user_id}")
    
    return {"user_id": user_id}

@app.get("/system/stats")
async def get_system_stats():
    """
    Get system statistics including agent count and memory usage
    """
    try:
        # Calculate agent stats
        agent_count = len(conversation_agents)
        active_agents_last_hour = 0
        current_time = time.time()
        
        for agent_data in conversation_agents.values():
            if current_time - agent_data["last_access"] < 3600:  # Last hour
                active_agents_last_hour += 1
        
        # Get memory usage
        memory_info = {}
        try:
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            memory_info = {"error": str(e)}
        
        stats = {
            "agent_stats": {
                "total_agents": agent_count,
                "active_last_hour": active_agents_last_hour,
            },
            "memory": memory_info,
            "config": {
                "max_agents": MAX_AGENTS,
                "idle_timeout_minutes": AGENT_IDLE_TIMEOUT // 60,
                "absolute_timeout_hours": AGENT_ABSOLUTE_TIMEOUT // 3600,
                "memory_threshold": f"{MEMORY_THRESHOLD}%"
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/cleanup")
async def force_cleanup():
    """
    Force cleanup of idle agents
    """
    try:
        removed_count = cleanup_agents(force_cleanup=True)
        return {
            "success": True, 
            "removed_agents": removed_count,
            "remaining_agents": len(conversation_agents)
        }
        
    except Exception as e:
        logger.error(f"Error during forced cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(
    chat_request: ChatRequest,
    stream: bool = False
):
    """
    Process a chat request from the frontend
    
    Args:
        chat_request: The chat request containing conversation ID and messages
        stream: Whether to stream the response (for compatibility with AI SDK)
    
    Returns:
        Streaming response if stream=True, otherwise regular JSON response
    """
    try:
        conversation_id = chat_request.id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info(f"No conversation ID provided, generated new ID: {conversation_id}")
        
        # Get or create agent for this conversation
        agent = get_agent_by_conversation(conversation_id)
        
        # Extract the latest user message from the messages array
        latest_message = chat_request.messages[-1]
        
        if latest_message.role != "user":
            raise HTTPException(
                status_code=400,
                detail="Last message in the request must be from 'user'"
            )
        
        # Extract query text from the message content
        query = latest_message.content
        
        # Convert message history to the format expected by the agent
        chat_history = []
        for message in chat_request.messages[:-1]:
            chat_history.append({
                "role": message.role,
                "content": message.content
            })
        
        # Update the agent's conversation history
        agent.conversation_history = chat_history
        
        # If streaming is requested, handle with StreamingResponse
        if stream:
            # Process the query with streaming
            return StreamingResponse(
                chat_streaming_generator(agent, query, conversation_id),
                media_type="text/event-stream"
            )
        
        # Non-streaming path
        result = agent.answer_query(
            query=query,
            conversation_id=conversation_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query: {result.get('error', 'Unknown error')}"
            )
        
        # Format the response in the expected structure
        response = {
            "id": conversation_id,
            "content": result["response"],
            "role": "assistant",
            "sources": result.get("sources", []),
            "success": result["success"]
        }
        
        return response
        
    except Exception as e:
        print(e)
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def chat_streaming_generator(agent: CognitiveAgent, query: str, conversation_id: str):
    """
    Generate streaming response in the format expected by AI SDK
    
    Args:
        agent: The cognitive agent instance
        query: User query
        conversation_id: Conversation ID
        
    Yields:
        Formatted SSE messages compatible with AI SDK
    """
    message_id = str(uuid.uuid4())
    
    try:
        # Process the query first to get the full response
        result = agent.answer_query(
            query=query,
            conversation_id=conversation_id
        )
        
        if not result["success"]:
            response_text = "I encountered an error while processing your request."
        else:
            response_text = result["response"]
        
        # AI SDK expects this specific format:
        # Each SSE message needs to be prefixed with "data: " and end with "\n\n"
        # The content needs to be valid JSON with specific fields

        # 1. Send metadata with model info
        yield f"data: {{\"role\":\"assistant\",\"id\":\"{message_id}\",\"createdAt\":{int(time.time())},\"content\":\"\",\"model\":\"rag-model\"}}\n\n"
        
        # 2. Stream the response token by token
        tokens = response_text.split()
        chunk_size = 3  # Number of tokens per chunk
        
        full_text = ""
        for i in range(0, len(tokens), chunk_size):
            chunk = " ".join(tokens[i:i+chunk_size])
            if i > 0:
                chunk = " " + chunk  # Add space except for first chunk
            
            full_text += chunk
            
            # AI SDK format for text delta
            delta_data = {
                "role": "assistant",
                "id": message_id,
                "createdAt": int(time.time()),
                "content": full_text,
                "model": "rag-model"
            }
            
            # Send the delta with proper SSE format
            yield f"data: {json.dumps(delta_data)}\n\n"
            await asyncio.sleep(0.05)  # Small delay to simulate real streaming
        
        # 3. Add sources as footnotes in the final response if available
        sources = result.get("sources", [])
        if sources:
            source_text = "\n\nSources:\n"
            for i, source in enumerate(sources[:3]):  # Limit to top 3 sources
                file_name = source.get("file_name", "Unknown")
                page = source.get("page", None)
                
                source_info = f"[{i+1}] {file_name}"
                if page:
                    source_info += f", Page: {page}"
                
                source_text += source_info + "\n"
                
            # Update the final response with source information
            full_text += source_text
            
            # Send the final complete response with sources
            final_data = {
                "role": "assistant",
                "id": message_id,
                "createdAt": int(time.time()),
                "content": full_text,
                "model": "rag-model"
            }
            
            yield f"data: {json.dumps(final_data)}\n\n"
        
        # 4. Send the done message
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        
        # Send error message in the format expected by AI SDK
        error_data = {
            "role": "assistant",
            "id": message_id,
            "createdAt": int(time.time()),
            "content": f"Error generating response: {str(e)}",
            "model": "rag-model"
        }
        
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"

def run_web_app():
    """Run the FastAPI web application"""
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "127.0.0.1")
    
    logger.info(f"Starting web server on {host}:{port}")
    uvicorn.run("src.user_interface.web:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    run_web_app()