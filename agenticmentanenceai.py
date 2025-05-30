import streamlit as st
import requests
import json
import base64
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import io
from PIL import Image

class GroqLLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def chat_completion(self, messages: List[Dict], model: str = "mixtral-8x7b-32768"):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"

class MaintenanceAgent:
    def __init__(self, llm_client: GroqLLMClient):
        self.llm_client = llm_client
        self.conversation_history = []
        
    def analyze_machine_data(self, machine_type: str, symptoms: str, 
                           images: List = None, documents: List = None) -> str:
        
        system_prompt = """You are an expert industrial maintenance AI assistant. 
        Analyze machine issues and provide detailed diagnostics, troubleshooting steps, 
        and maintenance recommendations. Be specific and prioritize safety."""
        
        user_message = f"""
        Machine Type: {machine_type}
        Reported Symptoms: {symptoms}
        
        Please provide:
        1. Possible root causes
        2. Immediate safety concerns
        3. Diagnostic steps
        4. Recommended actions
        5. Preventive measures
        """
        
        if images:
            user_message += f"\nVisual inspection data: {len(images)} images uploaded"
        if documents:
            user_message += f"\nTechnical documents: {len(documents)} files provided"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.llm_client.chat_completion(messages)
    
    def process_uploaded_files(self, uploaded_files: List) -> Dict:
        processed_data = {"images": [], "documents": []}
        
        for file in uploaded_files:
            if file.type.startswith('image/'):
                image = Image.open(file)
                processed_data["images"].append({
                    "name": file.name,
                    "size": image.size,
                    "format": image.format
                })
            elif file.type in ['application/pdf', 'text/plain']:
                processed_data["documents"].append({
                    "name": file.name,
                    "type": file.type,
                    "size": len(file.getvalue())
                })
        
        return processed_data

def main():
    st.set_page_config(
        page_title="Industrial Maintenance AI Assistant",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Industrial Maintenance AI Assistant")
    st.markdown("### AI-powered machine diagnostics and maintenance support")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        groq_api_key = st.text_input("Enter Groq API Key", type="password")
        
        if not groq_api_key:
            st.warning("Please enter your Groq API key to continue")
            st.stop()
    
    # Initialize the maintenance agent
    llm_client = GroqLLMClient(groq_api_key)
    agent = MaintenanceAgent(llm_client)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Machine Information")
        
        machine_type = st.selectbox(
            "Machine Type",
            ["Conveyor Belt", "Hydraulic Press", "CNC Machine", "Pump", 
             "Compressor", "Motor", "Generator", "Other"]
        )
        
        if machine_type == "Other":
            machine_type = st.text_input("Specify machine type")
        
        symptoms = st.text_area(
            "Describe the issue or symptoms",
            placeholder="e.g., Strange noise, vibration, overheating, error codes..."
        )
        
        # File upload section
        st.header("Upload Supporting Files")
        uploaded_files = st.file_uploader(
            "Upload images, manuals, or technical documents",
            type=['png', 'jpg', 'jpeg', 'pdf', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            processed_files = agent.process_uploaded_files(uploaded_files)
            st.success(f"Uploaded: {len(processed_files['images'])} images, {len(processed_files['documents'])} documents")
    
    with col2:
        st.header("Quick Actions")
        
        emergency_mode = st.checkbox("🚨 Emergency Mode")
        if emergency_mode:
            st.error("Emergency protocols activated")
        
        maintenance_type = st.radio(
            "Maintenance Type",
            ["Reactive", "Preventive", "Predictive"]
        )
        
        priority_level = st.select_slider(
            "Priority Level",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium"
        )
    
    # Analysis button
    if st.button("🔍 Analyze & Diagnose", type="primary"):
        if not symptoms.strip():
            st.error("Please describe the symptoms or issue")
        else:
            with st.spinner("Analyzing machine condition..."):
                # Perform analysis
                analysis = agent.analyze_machine_data(
                    machine_type, symptoms, 
                    processed_files.get("images", []) if uploaded_files else None,
                    processed_files.get("documents", []) if uploaded_files else None
                )
                
                st.header("🔍 Diagnostic Results")
                st.markdown(analysis)
                
                # Additional insights
                if emergency_mode:
                    st.error("⚠️ EMERGENCY: Follow safety protocols immediately")
                
                # Generate maintenance log entry
                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "machine_type": machine_type,
                    "symptoms": symptoms,
                    "priority": priority_level,
                    "maintenance_type": maintenance_type,
                    "files_uploaded": len(uploaded_files) if uploaded_files else 0
                }
                
                st.header("📋 Maintenance Log Entry")
                st.json(log_entry)
    
    # Chat interface
    st.header("💬 Ask Follow-up Questions")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about maintenance procedures, safety protocols, or troubleshooting..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create context-aware response
                context_messages = [
                    {"role": "system", "content": "You are a maintenance expert. Provide helpful, safe, and practical advice."},
                    {"role": "user", "content": f"Machine: {machine_type}, Context: {symptoms}, Question: {prompt}"}
                ]
                
                response = llm_client.chat_completion(context_messages)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer
    st.markdown("---")
    st.markdown("⚠️ **Safety Notice**: Always follow your organization's safety protocols. This AI assistant provides guidance but cannot replace qualified technicians for critical maintenance tasks.")

if __name__ == "__main__":
    main()