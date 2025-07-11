from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import asyncio
import logging
import json
from datetime import datetime
import os
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO )
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Chief of Staff", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PM_BASE_URL = "https://ai-staff-project-manager.ambitioussea-9ca2abb1.centralus.azurecontainerapps.io"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY" )
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.7

# Agent endpoints for health checks
AGENT_ENDPOINTS = {
    "Project Manager": f"{PM_BASE_URL}/health"
}

# System prompt for enhanced organizational awareness
SYSTEM_PROMPT = """You are the AI Chief of Staff for a 28-agent AI workforce management system. You coordinate between teams and provide strategic oversight.

CRITICAL: When providing information about projects, tasks, or operational data, you MUST use the actual data provided in the context. DO NOT make assumptions or use placeholder numbers.

Your role includes:
- Strategic coordination across all 28 agents
- Executive decision support and workforce management
- Real-time operational oversight and reporting
- Inter-agent communication and task delegation
- Resource allocation and priority management

Current organizational structure:
- Executive Layer: Chief of Staff (you), Project Manager, Business Manager
- Ideation Teams: Hoddle Team (4 agents), Waddle Team (4 agents) 
- Build Team: 8 agents for complete development stack
- Growth Team: 9 agents for marketing and growth

When asked about project or task information, always reference the actual data from the Project Manager's response. If no projects exist, clearly state there are 0 projects. If projects exist, provide the exact count and details from the PM data.

Be professional, strategic, and always use accurate real-time data in your responses."""

async def check_agent_status(agent_name: str, endpoint: str) -> Dict[str, Any]:
    """Check the status of a specific agent"""
    try:
        async with httpx.AsyncClient(timeout=10.0 ) as client:
            response = await client.get(endpoint)
            if response.status_code == 200:
                return {
                    "agent": agent_name,
                    "status": "operational",
                    "endpoint": endpoint,
                    "response_time": response.elapsed.total_seconds() if response.elapsed else 0
                }
            else:
                return {
                    "agent": agent_name,
                    "status": "error",
                    "endpoint": endpoint,
                    "error": f"HTTP {response.status_code}"
                }
    except Exception as e:
        return {
            "agent": agent_name,
            "status": "error", 
            "endpoint": endpoint,
            "error": str(e)
        }

async def get_database_stats() -> Dict[str, Any]:
    """Get current database statistics"""
    try:
        api_endpoint = AGENT_ENDPOINTS.get("api_gateway")
        if not api_endpoint:
            return {"status": "not_configured", "message": "API Gateway endpoint not configured"}
            
        async with httpx.AsyncClient(timeout=10.0 ) as client:
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

async def contact_project_manager() -> Dict[str, Any]:
    """Contact the Project Manager and get comprehensive updates"""
    try:
        # First check if PM is operational
        status = await check_agent_status("Project Manager", PM_BASE_URL + "/health")
        if status["status"] != "operational":
            return {
                "status": "unavailable",
                "message": f"Project Manager is {status['status']}: {status.get('error', 'Unknown error')}",
                "details": status
            }

        # Get dashboard data from PM (correct endpoints)
        async with httpx.AsyncClient(timeout=15.0 ) as client:
            dashboard_response = await client.get(f"{PM_BASE_URL}/dashboard")
            projects_response = await client.get(f"{PM_BASE_URL}/projects")
            tasks_response = await client.get(f"{PM_BASE_URL}/tasks")

            if dashboard_response.status_code == 200:
                dashboard_data = dashboard_response.json()
                projects_data = projects_response.json() if projects_response.status_code == 200 else {"projects": []}
                tasks_data = tasks_response.json() if tasks_response.status_code == 200 else {"tasks": []}

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
        logger.error(f"Error contacting Project Manager: {str(e)}")
        return {
            "status": "error",
            "message": f"Error contacting Project Manager: {str(e)}"
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
                "description": "MacBook-based development environment with full agent suite"
            },
            "azure_cloud": {
                "agents_operational": f"{azure_agents}/{total_agents}",
                "completion_percentage": f"{(azure_agents/total_agents)*100:.1f}%", 
                "status": "operational",
                "description": "Production cloud environment with executive agents"
            }
        },
        "azure_agent_status": azure_status,
        "database": db_stats,
        "organizational_structure": {
            "executive_layer": "Chief of Staff ✅ Operational (Azure), Project Manager ✅ Operational (Azure), Business Manager 📋 To be developed",
            "ideation_teams": "Hoddle Team ✅ Operational (Local), Waddle Team ✅ Operational (Local)",
            "build_team": "8 agents 📋 To be developed", 
            "growth_team": "9 agents 📋 To be developed"
        },
        "transition_protocol": {
            "current_phase": "Executive team migration and integration",
            "next_milestone": "Ideation teams Azure migration",
            "completion_status": f"We're currently at {local_agents}/28 agents operational in local development and {azure_agents}/28 in Azure"
        },
        "current_priorities": [
            "CoS-PM communication optimization",
            "Dashboard enhancement and agent controls", 
            "Hoddle and Waddle teams Azure migration",
            "Agent job description updates"
        ]
    }

async def generate_openai_response(messages: List[Dict[str, str]]) -> str:
    """Generate AI response for general queries"""
    try:
        async with httpx.AsyncClient(timeout=30.0 ) as client:
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
                return "I apologize, but I'm experiencing technical difficulties. Error code: " + str(response.status_code)

    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return "I apologize, but I'm experiencing technical difficulties. Error: " + str(e)

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
        if "project manager" in user_message.lower() and ("update" in user_message.lower() or "status" in user_message.lower() or "projects" in user_message.lower() or "managing" in user_message.lower()):
            pm_update = await contact_project_manager()
            if pm_update["status"] == "success":
                # Extract actual project data from PM response
                pm_data = pm_update["updates"]
                dashboard_data = pm_data.get("dashboard", {})
                projects_data = pm_data.get("projects", {})
                
                # Get actual project count from PM data
                total_projects = dashboard_data.get("total_projects", 0)
                active_projects = dashboard_data.get("active_projects", 0)
                project_list = projects_data.get("projects", [])
                
                # Create response with actual PM data
                if total_projects == 0:
                    response = f"I've contacted the Project Manager. Current status: There are currently {total_projects} projects being managed. The Project Manager is operational and ready to handle new project assignments."
                else:
                    response = f"I've contacted the Project Manager. Current status: There are currently {total_projects} projects being managed ({active_projects} active). "
                    if project_list:
                        response += f"The projects include: {', '.join([p.get('name', 'Unnamed') for p in project_list[:5]])}."
                    response += " The Project Manager is operational and managing the current workload."
                
                return JSONResponse(content={
                    "response": response,
                    "agent": "Chief of Staff",
                    "timestamp": datetime.now().isoformat(),
                    "pm_data": {
                        "total_projects": total_projects,
                        "active_projects": active_projects,
                        "project_count": len(project_list)
                    }
                })
            else:
                response = f"I attempted to contact the Project Manager but encountered an issue: {pm_update['message']}"
                return JSONResponse(content={
                    "response": response,
                    "agent": "Chief of Staff", 
                    "timestamp": datetime.now().isoformat(),
                    "error": pm_update
                })

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
{chr(10).join(f"- {priority}" for priority in system_status['current_priorities'])}

Would you like me to dive deeper into any specific area or provide updates on particular agents or projects?"""

        elif "organization" in user_message.lower() or "structure" in user_message.lower():
            response = f"""Here's our complete 28-agent organizational structure:

**Executive Layer (3 agents):**
- Chief of Staff: ✅ Operational (Azure) - Strategic oversight and coordination
- Project Manager: ✅ Operational (Azure) - Project coordination and task management  
- Business Manager: 📋 To be developed - Strategic planning and resource allocation

**Ideation Teams (8 agents):**
- Hoddle Team: ✅ Operational (Local) - 4 agents for market research and analysis
- Waddle Team: ✅ Operational (Local) - 4 agents for validation and verification

**Build Team (8 agents):** 📋 To be developed
- Frontend/Backend Developers, Database Specialist, DevOps Engineer, QA Tester, Security Specialist, API Developer, Mobile Developer

**Growth Team (9 agents):** 📋 To be developed  
- Marketing Strategist, Content Creator, SEO Specialist, Social Media Manager, Email Marketing Specialist, Analytics Expert, Customer Success Manager, Sales Development Rep, Partnership Manager

We're currently at 10/28 agents operational in local development and 2/28 in Azure cloud environment. The next phase focuses on migrating the Ideation teams to Azure."""

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

@app.get("/status")
async def get_status():
    """Get comprehensive system status"""
    return await get_system_status()

@app.get("/organization")
async def get_organization():
    """Get complete organizational structure"""
    system_status = await get_system_status()
    return {
        "total_agents": 28,
        "operational_agents": {
            "local_development": 10,
            "azure_cloud": 2
        },
        "structure": system_status["organizational_structure"],
        "transition_status": system_status["transition_protocol"]
    }

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
            "message": f"Issue escalated from {from_agent} has been received and logged",
            "escalation_id": f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in escalate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
