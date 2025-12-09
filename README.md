# 🔧 Maintenance.Agentic.AI

**An Autonomous AI-Powered Industrial Maintenance Assistant**

A comprehensive AI maintenance system powered by Google Gemini, designed for universal industrial equipment monitoring, fault diagnosis, and predictive maintenance planning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-orange.svg)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Key Features

### 🤖 AI-Powered Diagnostics
- **Gemini 1.5 Flash Integration**: Advanced AI reasoning for maintenance analysis
- **Real-time Fault Diagnosis**: Instant equipment troubleshooting
- **Root Cause Analysis**: Deep investigation of equipment failures
- **Confidence Scoring**: Reliability assessment for every diagnosis

### 📊 Comprehensive Monitoring
- **Sensor Data Analysis**: Temperature, vibration, pressure, humidity, sound level
- **Anomaly Detection**: Automatic identification of abnormal operating conditions
- **Visual Inspection**: Computer vision for equipment defect detection
- **Predictive Analytics**: AI-driven failure prediction

### 🏭 Universal Equipment Support
- **Rotating Equipment**: Motors, pumps, compressors, turbines, fans
- **Static Equipment**: Heat exchangers, vessels, piping systems
- **Control Systems**: PLCs, sensors, valves, actuators
- **HVAC Systems**: Chillers, boilers, air handlers, cooling towers
- **Power Systems**: Generators, transformers, switchgear

### 📈 Intelligent Planning
- **Automated Scheduling**: AI-optimized maintenance calendars
- **Resource Optimization**: Minimize downtime and costs
- **Task Prioritization**: Risk-based maintenance planning
- **Historical Analysis**: Learn from past maintenance data

### 💬 Natural Language Interface
- **Interactive Chat**: Ask questions in plain English
- **Expert Guidance**: Get professional maintenance advice
- **Safety Protocols**: Automated safety procedure generation
- **Documentation**: Instant access to equipment knowledge

---

## 🏗️ System Architecture


┌─────────────────────────────────────────────────────────────┐
│ User Interface (CLI) │
│ • Sensor Input • Chat • Equipment Management • Reports │
└───────────────────────────┬─────────────────────────────────┘
│
┌───────────────────────────▼─────────────────────────────────┐
│ Intelligent Maintenance Assistant │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Sensor │ │ Vision │ │ Diagnostic │ │
│ │ Processor │ │ Processor │ │ Engine │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
│
┌───────────────────────────▼─────────────────────────────────┐
│ Google Gemini 1.5 Flash │
│ Advanced AI Reasoning & Knowledge Base │
└───────────────────────────┬─────────────────────────────────┘
│
┌───────────────────────────▼─────────────────────────────────┐
│ Knowledge Base (SQLite Database) │
│ • Equipment Registry • Maintenance History • Fault DB │
└─────────────────────────────────────────────────────────────┘



## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://ai.google.dev))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhishekkumawat001/maintenance.agentic.ai.git
   cd maintenance.agentic.ai
   ```

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   # Copy environment template
   cp .env.example .env

   # Edit .env and add your Gemini API key
   GEMINI_API_KEY=your_api_key_here


4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

5. **Run the application**
   ```bash
   python agentic_ai_refactored.py
   ```

6. **Project Structure**

[maintenance.agentic.ai](http://_vscodecontentref_/0)
├── [agentic_ai_refactored.py](http://_vscodecontentref_/1)    
├── maintenance.db              # SQLite database (equipment & history)
├── [requirements.txt](http://_vscodecontentref_/2)            
├── .env                        # Environment variables (API keys)
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── [README.md](http://_vscodecontentref_/3)                 
├── pump_images/               # Sample equipment images
├── uploads/                   # User document uploads
└── visual_uploads/            # User image uploads
```

## 💡 Usage Examples

### Basic Maintenance Query
```python
# Run the main application
streamlit run agentic_ai_refactored.py

# In the web interface:
# 1. Enter: "My pump is making unusual noise"
# 2. Upload equipment image (optional)
# 3. Get AI-powered diagnostics and solutions
```

### Document Analysis
```python
# Upload maintenance manual or report
# System will:
# 1. Parse the document
# 2. Extract relevant maintenance procedures
# 3. Provide contextualized recommendations
```

### Visual Equipment Inspection
```python
# Upload equipment images
# AI will:
# 1. Analyze visual defects
# 2. Identify potential issues
# 3. Recommend maintenance actions
```

## 🎯 Use Cases

- **🏭 Manufacturing Plants**: Equipment monitoring and predictive maintenance
- **🔌 Power Plants**: Critical infrastructure maintenance planning
- **🚗 Automotive**: Vehicle maintenance diagnostics
- **✈️ Aviation**: Aircraft maintenance compliance
- **🏥 Healthcare**: Medical equipment servicing
- **🏢 Facilities Management**: Building systems maintenance

## 🛠️ API Integration

The system supports multiple AI providers:

- **Google Gemini Pro**: Advanced reasoning and analysis
- **Groq**: Fast inference for real-time responses
- **HuggingFace**: Open-source model ecosystem
- **LLaMA**: Meta's large language models
- **Custom Models**: Extensible architecture for new providers

## 📊 Performance

- **Response Time**: < 3 seconds for text queries
- **Image Analysis**: < 5 seconds for visual diagnostics
- **Document Processing**: Depends on file size
- **Concurrent Users**: Supports multiple simultaneous sessions

## 🔒 Security & Privacy

- **API Key Protection**: Environment variable storage
- **Data Privacy**: No persistent storage of sensitive data
- **Secure Communication**: HTTPS for all external API calls
- **Input Validation**: Comprehensive input sanitization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Contact**: abhishekkumawat001@gmail.com

## 🙏 Acknowledgments

- Google AI for Gemini Pro API
- Groq for fast inference capabilities
- HuggingFace for open-source model ecosystem
- Streamlit for the amazing web framework
- The open-source community for inspiration and tools

---

**Made with ❤️ by Abhishek Kumawat**