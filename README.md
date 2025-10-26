# 🔧 Maintenance.Agentic.AI

**An Autonomous AI Agent for Industrial Maintenance Management**

Maintenance.Agentic.AI is a comprehensive AI-powered virtual assistant designed to revolutionize industrial equipment maintenance through intelligent analysis, predictive diagnostics, and automated maintenance planning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Features

- **🤖 Multi-Model AI Support**: Integration with Gemini Pro, Groq, LLaMA, and HuggingFace models
- **📊 Visual Equipment Analysis**: Image-based equipment diagnostics and fault detection
- **📄 Document Processing**: Parse maintenance manuals, reports, and technical documentation
- **🔍 Predictive Maintenance**: AI-powered maintenance scheduling and failure prediction
- **⚡ Real-time Diagnostics**: Instant equipment troubleshooting and repair guidance
- **🛡️ Safety Protocol Generation**: Automated safety checklists and procedures
- **📱 Web Interface**: User-friendly Streamlit-based dashboard

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   AI Processing   │───▶│   Maintenance   │
│                 │    │                  │    │    Response     │
│ • Text Queries  │    │ • Gemini Pro     │    │ • Diagnostics   │
│ • Images        │    │ • Groq LLaMA     │    │ • Instructions  │
│ • Documents     │    │ • HuggingFace    │    │ • Safety Guides │
│ • Files         │    │ • Custom Models  │    │ • Schedules     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- API keys for AI services (Gemini, Groq, etc.)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhishekkumawat001/maintenance.agentic.ai.git
   cd maintenance.agentic.ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit google-generativeai groq requests pillow pyyaml python-dotenv aiohttp PyPDF2 python-docx opencv-python numpy
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run agentic_ai_refactored.py
   ```

## 🔧 Available Versions

| Version | File | Description | AI Models |
|---------|------|-------------|-----------|
| **Production** | `agentic_ai_refactored.py` | Main production version | Gemini Pro |
| **Free APIs** | `agentaiwithfreeapis.py` | Cost-effective version | Groq + Gemini |
| **Enhanced** | `agenticaiv2.py` | Multi-modal analysis | Gemini Vision |
| **LLaMA** | `agenticaiv3llama.py` | Open-source models | LLaMA variants |
| **Latest** | `agenticaiwithnewfreemodels.py` | Newest free models | Multiple providers |

## 📁 Project Structure

```
maintenance.agentic.ai/
├── 🐍 Core Applications
│   ├── agentic_ai_refactored.py      # Production version
│   ├── agentaiwithfreeapis.py        # Free APIs version
│   ├── agenticaiv2.py                # Enhanced version
│   ├── agenticaiv3llama.py           # LLaMA version
│   └── agenticaiwithnewfreemodels.py # Latest models
├── 🎨 Interface
│   └── newupdatedui.html             # Web UI template
├── 📊 Data Directories
│   ├── pump_images/                  # Sample equipment images
│   ├── uploads/                      # User document uploads
│   └── visual_uploads/               # User image uploads
├── ⚙️ Configuration
│   ├── .env                          # Environment variables
│   ├── .gitignore                    # Git ignore rules
│   └── README.md                     # This file
```

## 🔑 Environment Setup

Create a `.env` file with your API keys:

```env
# AI Service API Keys
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_google_gemini_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Optional: Additional configurations
DEBUG_MODE=false
MAX_FILE_SIZE=10MB
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

- **Issues**: [GitHub Issues](https://github.com/abhishekkumawat001/maintenance.agentic.ai/issues)
- **Documentation**: [Wiki](https://github.com/abhishekkumawat001/maintenance.agentic.ai/wiki)
- **Contact**: abhishekkumawat001@gmail.com

## 🙏 Acknowledgments

- Google AI for Gemini Pro API
- Groq for fast inference capabilities
- HuggingFace for open-source model ecosystem
- Streamlit for the amazing web framework
- The open-source community for inspiration and tools

---

**Made with ❤️ by Abhishek Kumawat**

*Revolutionizing industrial maintenance through AI innovation*