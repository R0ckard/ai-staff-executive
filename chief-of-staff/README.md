# Chief of Staff - AI Agent

Strategic AI agent for workforce coordination and executive oversight.

## 🎯 **Capabilities**

### **Strategic Oversight**
- Complete 28-agent organizational structure awareness
- Cross-team coordination and communication
- Executive decision support and strategic guidance
- Portfolio management and resource allocation

### **Enhanced Features**
- **Fixed PM Communication**: Uses correct Project Manager endpoints
- **Email Reporting System**: Automated reports to dave@quantai.com.au
- **Organizational Awareness**: Complete knowledge of all teams and capabilities
- **Voice Interface**: Preserved voice processing capabilities

### **Communication Protocols**
- **Project Manager Integration**: `/dashboard`, `/projects`, `/tasks` endpoints
- **Inter-Agent Messaging**: Secure communication with all 28 agents
- **Escalation Protocols**: Structured issue handling and resolution
- **Transition Management**: Project-to-business handoff leadership

## 🏗️ **Organizational Structure**

### **Executive Layer (Reports to User)**
- Chief of Staff (this agent)
- Project Manager
- Business Manager

### **Ideation Teams (8 agents)**
- Hoddle Team (4 agents): Market research and analysis
- Waddle Team (4 agents): Validation and verification

### **Build Team (8 agents)**
- Complete development stack from frontend to security

### **Growth Team (9 agents)**
- Complete marketing and growth capabilities

## 📧 **Email Reporting Schedule**

- **Executive Summary Digest**: Daily at 7:00 AM
- **Resource Allocation Brief**: Every 48 hours at 7:30 AM  
- **Portfolio Health Score**: Weekly Sunday at 8:00 AM
- **Transition Status Snapshots**: On active handoffs
- **Critical Alerts**: Instantaneous triggers

## 🚀 **Deployment**

Deployed to Azure Container Apps with:
- **Endpoint**: https://ai-staff-chief-of-staff.ambitioussea-9ca2abb1.centralus.azurecontainerapps.io
- **Health Check**: `/health` endpoint
- **Chat Interface**: https://proud-pebble-0b6812a10.2.azurestaticapps.net/index_fixed.html

## 🔧 **API Endpoints**

- `GET /health` - Health check
- `POST /chat` - Main chat interface
- `GET /dashboard` - System dashboard data
- `GET /organization` - Complete organizational structure
- `GET /status` - Enhanced system status
- `POST /escalate` - Strategic decision support

---

**Version**: 2.0.0 (Enhanced with organizational awareness)  
**Last Updated**: July 9, 2025  
**Status**: Ready for deployment

