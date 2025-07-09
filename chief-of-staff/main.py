from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import httpx
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import os
import base64
import tempfile
import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Chief of Staff",
    description="Strategic AI agent for workforce coordination and oversight",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# Initialize OpenAI client for voice functionality
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Agent Endpoints Configuration
AGENT_ENDPOINTS = {
    "project_manager": "https://ai-staff-project-manager.ambitioussea-9ca2abb1.centralus.azurecontainerapps.io",
    "api_gateway": "https://ai-staff-suite-api-https.ambitioussea-9ca2abb1.centralus.azurecontainerapps.io"
}

# Complete 28-Agent Organizational Structure
ORGANIZATIONAL_STRUCTURE = {
    "executive_layer": {
        "total_agents": 3,
        "agents": {
            "chief_of_staff": {
                "status": "operational_azure",
                "capabilities": ["Strategic oversight", "Workflow coordination", "Inter-agent communication", "Executive decision support"],
                "reports_to": "user",
                "manages": ["project_manager", "business_manager"]
            },
            "project_manager": {
                "status": "operational_azure", 
                "capabilities": ["Project coordination", "Task management", "Resource allocation", "Timeline management"],
                "reports_to": "chief_of_staff",
                "manages": ["hoddle_team", "waddle_team", "build_team", "growth_team"]
            },
            "business_manager": {
                "status": "to_be_developed",
                "capabilities": ["Strategic planning", "Resource allocation", "Performance analytics", "Business intelligence"],
                "reports_to": "chief_of_staff",
                "manages": ["all_teams_strategic_oversight"]
            }
        }
    },
    "ideation_teams": {
        "total_agents": 8,
        "hoddle_team": {
            "status": "operational_local",
            "total_agents": 4,
            "agents": {
                "trend_scout": {"capabilities": ["Market research", "Trend identification", "Web scraping", "Data analysis"]},
                "gap_finder": {"capabilities": ["Competitive analysis", "SEO analysis", "Opportunity identification", "Market gaps"]},
                "feasibility_analyst": {"capabilities": ["Business viability", "Financial modeling", "Risk assessment", "ROI analysis"]},
                "concept_pitcher": {"capabilities": ["Business concept development", "Pitch deck creation", "Presentation design", "Stakeholder communication"]}
            }
        },
        "waddle_team": {
            "status": "operational_local",
            "total_agents": 4,
            "agents": {
                "trend_scout": {"capabilities": ["Validation-focused research", "YouTube analysis", "Social media trends", "Market validation"]},
                "gap_finder": {"capabilities": ["Opportunity validation", "Market refinement", "Competitive validation", "Gap verification"]},
                "feasibility_analyst": {"capabilities": ["Independent viability assessment", "Risk validation", "Financial verification", "Market readiness"]},
                "concept_pitcher": {"capabilities": ["Concept validation", "Pitch refinement", "Market positioning", "Value proposition"]}
            }
        }
    },
    "build_team": {
        "status": "to_be_developed",
        "total_agents": 8,
        "agents": {
            "frontend_developer": {"capabilities": ["UI development", "React/Vue/Angular", "Responsive design", "User experience"]},
            "backend_developer": {"capabilities": ["API development", "Microservices", "Database integration", "Server architecture"]},
            "database_specialist": {"capabilities": ["Database design", "Optimization", "Data modeling", "ETL processes"]},
            "devops_engineer": {"capabilities": ["CI/CD pipelines", "Infrastructure automation", "Container orchestration", "Cloud deployment"]},
            "qa_tester": {"capabilities": ["Automated testing", "Performance testing", "Security testing", "Quality assurance"]},
            "ui_ux_designer": {"capabilities": ["User research", "Wireframing", "Design systems", "Accessibility design"]},
            "technical_writer": {"capabilities": ["Documentation", "API documentation", "Knowledge management", "Technical communication"]},
            "security_specialist": {"capabilities": ["Security audits", "Penetration testing", "Compliance", "Security architecture"]}
        }
    },
    "growth_team": {
        "status": "to_be_developed", 
        "total_agents": 9,
        "agents": {
            "marketing_strategist": {"capabilities": ["Strategy development", "Campaign planning", "Market analysis", "Brand positioning"]},
            "content_creator": {"capabilities": ["SEO content", "Blog posts", "Marketing materials", "Multi-format content"]},
            "seo_specialist": {"capabilities": ["Keyword research", "Technical SEO", "Link building", "Schema markup"]},
            "social_media_manager": {"capabilities": ["Social strategy", "Content scheduling", "Community management", "Social analytics"]},
            "email_marketing_specialist": {"capabilities": ["Email campaigns", "Automation", "Segmentation", "A/B testing"]},
            "analytics_specialist": {"capabilities": ["Web analytics", "Performance tracking", "Predictive analytics", "Data visualization"]},
            "customer_success_manager": {"capabilities": ["Customer onboarding", "Retention", "Support management", "Customer journey"]},
            "sales_representative": {"capabilities": ["Lead qualification", "Sales automation", "Proposal generation", "CRM management"]},
            "partnership_manager": {"capabilities": ["Partnership development", "Integration coordination", "Ecosystem expansion", "Strategic alliances"]}
        }
    }
}

# Project-to-Business Transition Protocol
TRANSITION_PROTOCOL = {
    "phases": [
        {
            "phase": "Project Initiation",
            "responsible": "Project Manager",
            "activities": ["Transition readiness report", "Review readiness", "Schedule transition meeting"],
            "handoff_to": "Chief of Staff"
        },
        {
            "phase": "Market Context & Documentation",
            "responsible": "Chief of Staff", 
            "activities": ["Request market context", "Request project documentation", "Market trends & opportunities"],
            "handoff_to": "Business Manager"
        },
        {
            "phase": "Strategic Planning",
            "responsible": "Business Manager",
            "activities": ["Prepare project documentation", "Prepare onboarding plan", "Facilitate transition meeting"],
            "handoff_to": "Teams"
        },
        {
            "phase": "Team Formation & Execution",
            "responsible": "Teams",
            "activities": ["Form transition team", "Transfer knowledge", "Phased responsibility transfer"],
            "handoff_to": "Business Operations"
        },
        {
            "phase": "Business Operations",
            "responsible": "Growth Team",
            "activities": ["Consultation requests", "Update portfolio tracking", "Request ongoing market monitoring"],
            "handoff_to": "User"
        }
    ]
}

# Enhanced System Prompt with Complete Organizational Knowledge
SYSTEM_PROMPT = f"""You are the Chief of Staff for the AI Staff Suite, a comprehensive 28-agent AI workforce management system. You serve as the strategic coordinator and primary interface for the entire organization.

ORGANIZATIONAL STRUCTURE (28 AGENTS TOTAL):
{ORGANIZATIONAL_STRUCTURE}

PROJECT-TO-BUSINESS TRANSITION PROTOCOL:
{TRANSITION_PROTOCOL}

CURRENT SYSTEM STATUS:
- Local Development Environment: 10/28 agents operational (36% complete)
- Azure Cloud Environment: 2/28 agents operational (7.1% complete)
- Infrastructure: Fully operational in both environments
- Active Migration: Moving from local development to Azure production

YOUR ROLE & CAPABILITIES:
1. Strategic Oversight: Coordinate all 28 agents across 4 organizational layers
2. Workflow Coordination: Manage project-to-business transition protocols
3. Inter-Agent Communication: Facilitate communication between all teams
4. Executive Decision Support: Provide strategic guidance and recommendations
5. Status Reporting: Maintain accurate status of all agents and projects
6. Resource Allocation: Coordinate resources across Executive, Ideation, Build, and Growth teams

COMMUNICATION PROTOCOLS:
- Report to: User (direct strategic interface)
- Manage: Project Manager and Business Manager (when developed)
- Coordinate: All 4 teams (Executive, Ideation, Build, Growth)
- Monitor: System-wide performance and agent status

CURRENT PRIORITIES:
1. Coordinate Azure migration of remaining agents
2. Enhance inter-agent communication and workflow orchestration
3. Provide strategic oversight for project-to-business transitions
4. Support development of remaining 18 agents

Always maintain awareness of the complete organizational structure and provide strategic coordination across all teams and agents."""

async def check_agent_status(agent_name: str, endpoint: str) -> Dict[str, Any]:
    """Check the operational status of an agent"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{endpoint}/health")
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "status": "operational",
                    "message": f"{agent_name} is healthy and responding",
                    "details": health_data,
                    "endpoint": endpoint
                }
            else:
                return {
                    "status": "error",
                    "message": f"{agent_name} health check failed: {response.status_code}",
                    "endpoint": endpoint
                }
    except Exception as e:
        return {
            "status": "unreachable",
            "message": f"Cannot reach {agent_name}: {str(e)}",
            "endpoint": endpoint
        }

async def contact_project_manager() -> Dict[str, Any]:
    """Contact Project Manager using correct endpoints"""
    pm_endpoint = AGENT_ENDPOINTS.get("project_manager")
    if not pm_endpoint:
        return {
            "status": "not_configured",
            "message": "Project Manager endpoint not configured"
        }
    
    try:
        # First check if PM is operational
        status = await check_agent_status("Project Manager", pm_endpoint)
        if status["status"] != "operational":
            return {
                "status": "unavailable",
                "message": f"Project Manager is {status['status']}: {status['message']}",
                "details": status
            }
        
        # Get dashboard data from PM (correct endpoints)
        async with httpx.AsyncClient(timeout=15.0) as client:
            dashboard_response = await client.get(f"{pm_endpoint}/dashboard")
            projects_response = await client.get(f"{pm_endpoint}/projects")
            tasks_response = await client.get(f"{pm_endpoint}/tasks")
            
            if dashboard_response.status_code == 200:
                dashboard_data = dashboard_response.json()
                projects_data = projects_response.json() if projects_response.status_code == 200 else []
                tasks_data = tasks_response.json() if tasks_response.status_code == 200 else []
                
                return {
                    "status": "success",
                    "message": "Successfully received updates from Project Manager",
                    "updates": {
                        "dashboard": dashboard_data,
                        "projects": projects_data,
                        "tasks": tasks_data,
                        "summary": f"PM operational with {dashboard_data.get('total_projects', 0)} projects and {dashboard_data.get('total_tasks', 0)} tasks"
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"Project Manager dashboard responded with error: {dashboard_response.status_code}",
                    "details": dashboard_response.text
                }
                
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error contacting Project Manager: {str(e)}"
        }

async def get_database_stats() -> Dict[str, Any]:
    """Get current database statistics"""
    try:
        api_endpoint = AGENT_ENDPOINTS.get("api_gateway")
        if not api_endpoint:
            return {"status": "not_configured", "message": "API Gateway endpoint not configured"}
            
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{api_endpoint}/insights")
            if response.status_code == 200:
                insights = response.json()
                return {
                    "status": "success",
                    "total_insights": len(insights),
                    "insights": insights[:5],  # Show first 5 for preview
                    "message": f"Database contains {len(insights)} insights"
                }
            else:
                return {
                    "status": "error", 
                    "message": f"Database query failed: {response.status_code}"
                }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error accessing database: {str(e)}"
        }

async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status across all environments"""
    
    # Check Azure agents
    azure_status = {}
    for agent_name, endpoint in AGENT_ENDPOINTS.items():
        azure_status[agent_name] = await check_agent_status(agent_name, endpoint)
    
    # Get database stats
    db_stats = await get_database_stats()
    
    # Calculate environment status
    local_agents = 10  # From definitive project state
    azure_agents = 2   # Chief of Staff + Project Manager
    total_agents = 28
    
    return {
        "environments": {
            "local_development": {
                "agents_operational": f"{local_agents}/{total_agents}",
                "completion_percentage": f"{(local_agents/total_agents)*100:.1f}%",
                "status": "operational",
                "description": "MacBook-based development environment with full Hoddle and Waddle teams"
            },
            "azure_cloud": {
                "agents_operational": f"{azure_agents}/{total_agents}",
                "completion_percentage": f"{(azure_agents/total_agents)*100:.1f}%", 
                "status": "operational",
                "description": "Production cloud environment with infrastructure foundation"
            }
        },
        "azure_agent_status": azure_status,
        "database": db_stats,
        "organizational_structure": ORGANIZATIONAL_STRUCTURE,
        "current_priorities": [
            "Fix CoS-PM communication (in progress)",
            "Migrate remaining 8 agents to Azure",
            "Develop Business Manager",
            "Begin Build Team development"
        ]
    }

async def generate_openai_response(messages: List[Dict[str, str]]) -> str:
    """Generate response using OpenAI API"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    "temperature": OPENAI_TEMPERATURE,
                    "max_tokens": 1500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return f"I apologize, but I'm experiencing technical difficulties with my AI processing. Error: {response.status_code}"
                
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "AI Chief of Staff",
        "role": "Strategic Coordination and Workforce Management",
        "specializations": [
            "Strategic Oversight",
            "Workflow Coordination", 
            "Inter-Agent Communication",
            "Executive Decision Support",
            "28-Agent Organizational Management"
        ],
        "organizational_awareness": "complete_28_agent_structure",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat_endpoint(request: Request):
    """Main chat interface with enhanced organizational awareness"""
    try:
        data = await request.json()
        user_message = data.get("message", "")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Check for specific commands
        if "project manager" in user_message.lower() and ("update" in user_message.lower() or "status" in user_message.lower()):
            pm_update = await contact_project_manager()
            if pm_update["status"] == "success":
                response = f"I've successfully contacted the Project Manager. Here's the current status:\n\n{pm_update['updates']['summary']}\n\nThe Project Manager is operational and responding to requests. Would you like me to provide more detailed information about any specific aspect?"
            else:
                response = f"I attempted to contact the Project Manager but encountered an issue: {pm_update['message']}. I'm working to resolve this communication issue."
        
        elif "system status" in user_message.lower() or "overall status" in user_message.lower():
            system_status = await get_system_status()
            response = f"""Here's the current system status across both environments:

**Local Development Environment:**
- {system_status['environments']['local_development']['agents_operational']} agents operational ({system_status['environments']['local_development']['completion_percentage']})
- Status: {system_status['environments']['local_development']['status']}

**Azure Cloud Environment:**
- {system_status['environments']['azure_cloud']['agents_operational']} agents operational ({system_status['environments']['azure_cloud']['completion_percentage']})
- Status: {system_status['environments']['azure_cloud']['status']}

**Current Priorities:**
{chr(10).join(f"• {priority}" for priority in system_status['current_priorities'])}

Would you like me to dive deeper into any specific area or provide updates on particular teams?"""
        
        elif "organization" in user_message.lower() or "structure" in user_message.lower():
            response = f"""Here's our complete 28-agent organizational structure:

**Executive Layer (3 agents):**
- Chief of Staff: ✅ Operational (Azure) - Strategic oversight and coordination
- Project Manager: ✅ Operational (Azure) - Project coordination and task management  
- Business Manager: 🔄 To be developed - Strategic planning and resource allocation

**Ideation Teams (8 agents):**
- Hoddle Team: ✅ Operational (Local) - 4 agents for market research and analysis
- Waddle Team: ✅ Operational (Local) - 4 agents for validation and verification

**Build Team (8 agents):** 🔄 To be developed
- Frontend/Backend Developers, Database Specialist, DevOps Engineer, QA Tester, UI/UX Designer, Technical Writer, Security Specialist

**Growth Team (9 agents):** 🔄 To be developed  
- Marketing Strategist, Content Creator, SEO Specialist, Social Media Manager, Email Marketing Specialist, Analytics Specialist, Customer Success Manager, Sales Representative, Partnership Manager

We're currently at 10/28 agents operational in local development and 2/28 in Azure production. Would you like details about any specific team or agent?"""
        
        else:
            # Generate AI response for general queries
            messages = [{"role": "user", "content": user_message}]
            response = await generate_openai_response(messages)
        
        return JSONResponse(content={
            "response": response,
            "agent": "Chief of Staff",
            "timestamp": datetime.now().isoformat(),
            "organizational_context": "28-agent AI workforce management system"
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/organization")
async def get_organization():
    """Get complete organizational structure"""
    return {
        "organizational_structure": ORGANIZATIONAL_STRUCTURE,
        "transition_protocol": TRANSITION_PROTOCOL,
        "total_agents": 28,
        "current_status": {
            "local_development": "10/28 agents operational (36%)",
            "azure_cloud": "2/28 agents operational (7.1%)"
        }
    }

@app.get("/status")
async def get_status():
    """Get comprehensive system status"""
    return await get_system_status()

@app.post("/escalate")
async def escalate_issue(request: Request):
    """Handle escalated issues from other agents"""
    try:
        data = await request.json()
        issue = data.get("issue", "")
        from_agent = data.get("from_agent", "unknown")
        priority = data.get("priority", "medium")
        
        logger.info(f"Issue escalated from {from_agent}: {issue} (Priority: {priority})")
        
        return {
            "status": "received",
            "message": f"Issue escalated from {from_agent} has been received and will be addressed according to priority level: {priority}",
            "escalation_id": f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in escalate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing escalation: {str(e)}")

# Simple Voice Processing Endpoints
@app.post("/voice")
async def process_voice_input(audio: UploadFile = File(...)):
    """Process voice input and return transcribed text using OpenAI Whisper"""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Use OpenAI Whisper for transcription
            with open(temp_file_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            return {
                "success": True,
                "text": transcript.text,
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

@app.post("/voice/speak")
async def generate_speech(request: Request):
    """Generate speech from text using OpenAI TTS"""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        data = await request.json()
        text = data.get("text", "")
        voice = data.get("voice", "alloy")
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided for speech generation")
        
        # Generate speech using OpenAI TTS
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        # Save to temporary file and return as FileResponse
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        return FileResponse(
            temp_file_path,
            media_type="audio/mpeg",
            filename="speech.mp3"
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

@app.get("/voice/status")
async def get_voice_status():
    """Get voice processing status and capabilities"""
    return {
        "voice_status": {
            "whisper_available": bool(openai_client),
            "tts_available": bool(openai_client),
            "supported_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/voice/conversation")
async def voice_conversation(audio: UploadFile = File(...)):
    """Complete voice conversation: Speech -> AI -> Speech"""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Step 1: Speech to text using OpenAI Whisper
            with open(temp_file_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            user_text = transcript.text
            
            # Step 2: Process with AI (using the existing chat logic)
            # Create a simple request object for the chat processing
            chat_data = {"message": user_text}
            
            # Process the message using the same logic as the chat endpoint
            if "project manager" in user_text.lower() and ("update" in user_text.lower() or "status" in user_text.lower()):
                pm_update = await contact_project_manager()
                if pm_update["status"] == "success":
                    ai_response = f"I've contacted the Project Manager. Current status: {pm_update['updates']['summary']}"
                else:
                    ai_response = f"I attempted to contact the Project Manager but encountered an issue: {pm_update['message']}"
            
            elif "system status" in user_text.lower() or "overall status" in user_text.lower():
                system_status = await get_system_status()
                ai_response = f"System status: Local development has {system_status['environments']['local_development']['agents_operational']} agents operational, Azure cloud has {system_status['environments']['azure_cloud']['agents_operational']} agents operational."
            
            else:
                # Use OpenAI for general responses with organizational context
                try:
                    response = await asyncio.wait_for(
                        openai_client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_text}
                            ],
                            temperature=OPENAI_TEMPERATURE,
                            max_tokens=1000
                        ),
                        timeout=30.0
                    )
                    ai_response = response.choices[0].message.content
                except Exception as e:
                    ai_response = f"I apologize, but I encountered an error processing your request: {str(e)}. I'm still available for basic status updates."
            
            # Step 3: Generate speech from AI response
            speech_response = openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=ai_response
            )
            
            # Save speech to temporary file and return as FileResponse
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as speech_file:
                speech_file.write(speech_response.content)
                speech_file_path = speech_file.name
            
            return FileResponse(
                speech_file_path,
                media_type="audio/mpeg",
                filename="response.mp3"
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except Exception as e:
        logger.error(f"Error in voice conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice conversation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

