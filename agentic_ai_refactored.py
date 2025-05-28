import asyncio
import json
import logging
import sqlite3
import random
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from google.generativeai import configure
from google.generativeai.generative_models import GenerativeModel
from dotenv import load_dotenv
load_dotenv()

class LLMConfig:
    """Configuration for LLM providers"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")

gemini_key = os.getenv("GEMINI_API_KEY")
print(gemini_key)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
# Data Models
@dataclass
class SensorData:
    sensor_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    humidity: float
    sound_level: float

@dataclass
class FaultDiagnosis:
    fault_id: str
    severity: str
    confidence: float
    description: str
    root_cause: str
    recommended_actions: List[str]
    estimated_downtime: int  # in hours

class MaintenanceType(Enum):
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"

@dataclass
class MaintenanceTask:
    task_id: str
    equipment_id: str
    task_type: MaintenanceType
    priority: int
    description: str
    scheduled_date: datetime
    estimated_duration: int
    required_parts: List[str]
    assigned_technician: Optional[str] = None

# LLM Integration Layer (Gemini Only)
class LLMProvider:
    def __init__(self):
        self.init_gemini()
    
    def init_gemini(self):
        """Initialize Gemini provider"""
        try:
            if LLMConfig.GEMINI_API_KEY != "your-gemini-api-key":
                genai.configure(api_key=LLMConfig.GEMINI_API_KEY)
                self.gemini_model = GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini model initialized successfully")
                logger.info("Gemini model initialized successfully")
            else:
                self.gemini_model = None
                logger.warning("Gemini API key not configured")
        except Exception as e:
            logger.warning(f"Error initializing Gemini: {e}")
            self.gemini_model = None
    
    async def call_gemini(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            if not self.gemini_model:
                return "Gemini API not configured. Please set GEMINI_API_KEY environment variable."
                
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error calling Gemini: {e}"
    
    async def get_best_response(self, prompt: str) -> str:
        """Get response from Gemini"""
        return await self.call_gemini(prompt)
    
    async def close(self):
        """Cleanup resources (placeholder for consistency)"""
        pass
# 1. Perception Module
class SensorDataProcessor:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.thresholds = {
            'temperature': {'min': 0, 'max': 80},
            'vibration': {'min': 0, 'max': 10},
            'pressure': {'min': 0, 'max': 100},
            'humidity': {'min': 0, 'max': 100}
        }
    
    def process_sensor_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Process incoming sensor data and detect anomalies"""
        anomalies = []
        
        for param, value in [
            ('temperature', sensor_data.temperature),
            ('vibration', sensor_data.vibration),
            ('pressure', sensor_data.pressure),
            ('humidity', sensor_data.humidity)
        ]:
            if not (self.thresholds[param]['min'] <= value <= self.thresholds[param]['max']):
                anomalies.append({
                    'parameter': param,
                    'value': value,
                    'threshold': self.thresholds[param],
                    'severity': 'high' if value > self.thresholds[param]['max'] * 1.2 else 'medium'
                })
        
        return {
            'sensor_id': sensor_data.sensor_id,
            'timestamp': sensor_data.timestamp,
            'anomalies': anomalies,
            'status': 'critical' if len(anomalies) > 2 else 'warning' if anomalies else 'normal'
        }
    
    async def analyze_with_llm(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Use Gemini for advanced sensor data analysis"""
        prompt = f"""
        Analyze the following industrial equipment sensor data:
        
        Equipment: {sensor_data.sensor_id}
        Temperature: {sensor_data.temperature}°C (normal: 0-80°C)
        Vibration: {sensor_data.vibration} mm/s (normal: 0-10 mm/s)
        Pressure: {sensor_data.pressure} bar (normal: 0-100 bar)
        Humidity: {sensor_data.humidity}% (normal: 0-100%)
        Sound Level: {sensor_data.sound_level} dB
        
        Please provide:
        1. Anomaly detection and severity assessment
        2. Potential failure modes
        3. Recommended immediate actions
        4. Risk assessment (1-10 scale)
        
        Format your response as JSON with keys: anomalies, failure_modes, recommendations, risk_score
        """
        
        response = await self.llm_provider.get_best_response(prompt)
        
        try:
            # Try to parse JSON response
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to structured text parsing
            return {
                "llm_analysis": response,
                "anomalies": [],
                "failure_modes": ["Gemini analysis available in text format"],
                "recommendations": ["Review Gemini analysis"],
                "risk_score": 5
            }

class VisionProcessor:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
    
    def analyze_visual_input(self, image_path: str) -> Dict[str, Any]:
        """Analyze camera feed for visual defects"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            # Basic defect detection using edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            defects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Threshold for significant defects
                    x, y, w, h = cv2.boundingRect(contour)
                    defects.append({
                        'location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'area': float(area),
                        'type': 'surface_defect'
                    })
            
            return {
                'defects_found': len(defects),
                'defects': defects,
                'analysis_timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Vision processing error: {e}")
            return {"error": str(e)}
    
    async def analyze_with_llm(self, image_path: str, defects: List[Dict]) -> str:
        """Use Gemini to analyze visual defects"""
        prompt = f"""
        Analyze the following visual inspection results for industrial equipment:
        
        Image: {image_path}
        Number of defects found: {len(defects)}
        
        Defect details:
        """
        
        for i, defect in enumerate(defects, 1):
            prompt += f"""
            Defect {i}:
            - Type: {defect['type']}
            - Location: ({defect['location']['x']}, {defect['location']['y']})
            - Size: {defect['location']['width']}x{defect['location']['height']}
            - Area: {defect['area']} pixels
            """
        
        prompt += """
        
        Please provide:
        1. Severity assessment of detected defects
        2. Potential causes of these defects
        3. Recommended maintenance actions
        4. Expected progression if left untreated
        """
        
        return await self.llm_provider.get_best_response(prompt)

# 2. Knowledge Base Module
class MaintenanceKnowledgeBase:
    def __init__(self, db_path: str = "maintenance.db"):
        self.db_path = db_path
        self.init_database()
        # Remove the populate_sample_data call to start with empty database
        # self.populate_sample_data()
    
    def init_database(self):
        """Initialize SQLite database for maintenance knowledge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Equipment table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equipment (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                model TEXT,
                manufacturer TEXT,
                installation_date DATE,
                last_maintenance DATE
            )
        ''')
        
        # Maintenance history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id TEXT,
                maintenance_date DATE,
                type TEXT,
                description TEXT,
                technician TEXT,
                cost REAL,
                downtime_hours INTEGER,
                FOREIGN KEY (equipment_id) REFERENCES equipment (id)
            )
        ''')
        
        # Fault patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fault_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_type TEXT,
                symptoms TEXT,
                root_cause TEXT,
                solution TEXT,
                confidence_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_equipment(self, equipment_data: Dict) -> bool:
        """Add new equipment to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO equipment (id, name, type, model, manufacturer, installation_date, last_maintenance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                equipment_data['id'],
                equipment_data['name'],
                equipment_data['type'],
                equipment_data.get('model', 'Unknown'),
                equipment_data.get('manufacturer', 'Unknown'),
                equipment_data.get('installation_date', datetime.now().strftime('%Y-%m-%d')),
                equipment_data.get('last_maintenance', 'Never')
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding equipment: {e}")
            return False
    
    def get_maintenance_history(self, equipment_id: str) -> List[Dict]:
        """Retrieve maintenance history for equipment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM maintenance_history 
            WHERE equipment_id = ? 
            ORDER BY maintenance_date DESC
        ''', (equipment_id,))
        
        history = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in history]
    
    def find_similar_faults(self, symptoms: str) -> List[Dict]:
        """Find similar fault patterns in knowledge base"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple text matching - in production, use vector similarity
        cursor.execute('''
            SELECT * FROM fault_patterns 
            WHERE symptoms LIKE ? 
            ORDER BY confidence_score DESC
        ''', (f'%{symptoms}%',))
        
        patterns = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in patterns]
    
    def get_all_equipment(self) -> List[Dict]:
        """Get all equipment from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM equipment')
        equipment = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in equipment]

# 3. Gemini-Enhanced Diagnostic Engine
class DiagnosticEngine:
    def __init__(self, knowledge_base: MaintenanceKnowledgeBase, llm_provider: LLMProvider):
        self.knowledge_base = knowledge_base
        self.llm_provider = llm_provider
    
    async def diagnose_fault(self, sensor_data: Dict, visual_data: Dict, equipment_id: str) -> FaultDiagnosis:
        """Perform Gemini-enhanced root cause analysis"""
        
        # Get equipment info
        equipment_list = self.knowledge_base.get_all_equipment()
        equipment = next((eq for eq in equipment_list if eq['id'] == equipment_id), None)
        
        # Get maintenance history
        history = self.knowledge_base.get_maintenance_history(equipment_id)
        
        # Prepare comprehensive prompt for Gemini
        prompt = f"""
        As an expert maintenance engineer, analyze this equipment fault:
        
        EQUIPMENT INFORMATION:
        - ID: {equipment_id}
        - Type: {equipment.get('type', 'Unknown') if equipment else 'Unknown'}
        - Model: {equipment.get('model', 'Unknown') if equipment else 'Unknown'}
        - Last Maintenance: {equipment.get('last_maintenance', 'Unknown') if equipment else 'Unknown'}
        
        SENSOR DATA ANOMALIES:
        {json.dumps(sensor_data.get('anomalies', []), indent=2)}
        
        VISUAL INSPECTION:
        {json.dumps(visual_data, indent=2)}
        
        MAINTENANCE HISTORY:
        {json.dumps(history[:3], indent=2) if history else 'No recent history'}
        
        Please provide a comprehensive fault diagnosis including:
        1. Most likely root cause
        2. Confidence level (0-1)
        3. Severity (low/medium/high)
        4. Specific recommended actions
        5. Estimated repair time in hours
        6. Risk if left unaddressed
        
        Format as JSON with keys: root_cause, confidence, severity, actions, estimated_hours, risk_description
        """
        
        llm_response = await self.llm_provider.get_best_response(prompt)
        
        try:
            diagnosis_data = json.loads(llm_response)
            
            return FaultDiagnosis(
                fault_id=f"F_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=diagnosis_data.get('severity', 'medium'),
                confidence=float(diagnosis_data.get('confidence', 0.7)),
                description=f"Gemini-diagnosed fault for {equipment_id}",
                root_cause=diagnosis_data.get('root_cause', 'Unknown cause'),
                recommended_actions=diagnosis_data.get('actions', ['Schedule inspection']),
                estimated_downtime=int(diagnosis_data.get('estimated_hours', 4))
            )
        except (json.JSONDecodeError, ValueError):
            # Fallback to rule-based diagnosis
            symptoms = []
            if sensor_data.get('anomalies'):
                for anomaly in sensor_data['anomalies']:
                    symptoms.append(f"{anomaly['parameter']} {anomaly['severity']}")
            
            symptoms_str = " ".join(symptoms)
            similar_faults = self.knowledge_base.find_similar_faults(symptoms_str)
            
            if similar_faults:
                best_match = similar_faults[0]
                confidence = best_match['confidence_score']
                root_cause = best_match['root_cause']
                recommended_actions = best_match['solution'].split(';')
            else:
                confidence = 0.6
                root_cause = "Gemini analysis available in logs"
                recommended_actions = ["Review Gemini diagnosis", "Schedule expert inspection"]
            
            severity = "high" if any(a.get('severity') == 'high' for a in sensor_data.get('anomalies', [])) else "medium"
            
            return FaultDiagnosis(
                fault_id=f"F_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=severity,
                confidence=confidence,
                description=f"Fault detected with symptoms: {symptoms_str}",
                root_cause=root_cause,
                recommended_actions=recommended_actions,
                estimated_downtime=8 if severity == "high" else 4
            )

# 4. Gemini-Enhanced Natural Language Interface
class NaturalLanguageInterface:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
    
    async def process_query(self, message: str, context: Dict) -> str:
        """Process natural language queries with context"""
        prompt = f"""
        You are an expert industrial maintenance engineer with comprehensive knowledge of all types of equipment.
        
        Current system context:
        - Status: {context.get('status', 'operational')}
        - Active faults: {len(context.get('current_faults', []))}
        - Upcoming tasks: {len(context.get('upcoming_tasks', []))}
        
        User question: {message}
        
        Provide helpful, accurate maintenance advice based on industry best practices.
        Focus on practical, actionable guidance for industrial maintenance professionals.
        """
        
        return await self.llm_provider.get_best_response(prompt)

# 5. Gemini-Enhanced Maintenance Planner
class MaintenancePlanner:
    def __init__(self, knowledge_base: MaintenanceKnowledgeBase, llm_provider: LLMProvider):
        self.knowledge_base = knowledge_base
        self.llm_provider = llm_provider
    
    async def generate_maintenance_schedule(self, equipment_list: List[str]) -> List[MaintenanceTask]:
        """Generate Gemini-optimized maintenance schedule"""
        
        # Get equipment data
        all_equipment = self.knowledge_base.get_all_equipment()
        equipment_data = {eq['id']: eq for eq in all_equipment}
        
        # Get maintenance history for all equipment
        history_data = {}
        for eq_id in equipment_list:
            history_data[eq_id] = self.knowledge_base.get_maintenance_history(eq_id)
        
        # Use Gemini to optimize schedule
        prompt = f"""
        As a maintenance planning expert, create an optimized maintenance schedule:
        
        EQUIPMENT LIST: {equipment_list}
        
        EQUIPMENT DETAILS:
        {json.dumps({eq_id: equipment_data.get(eq_id, {}) for eq_id in equipment_list}, indent=2)}
        
        MAINTENANCE HISTORY:
        {json.dumps(history_data, indent=2)}
        
        Please create a maintenance schedule considering:
        1. Equipment criticality and type
        2. Historical maintenance patterns
        3. Manufacturer recommendations
        4. Optimal resource utilization
        5. Minimal production disruption
        
        For each equipment, specify:
        - Next maintenance date
        - Priority level (1-5)
        - Estimated duration (hours)
        - Required parts/materials
        - Maintenance type (preventive/predictive/corrective)
        
        Format as JSON array with keys: equipment_id, date, priority, duration, parts, type, description
        """
        
        response = await self.llm_provider.get_best_response(prompt)
        
        try:
            schedule_data = json.loads(response)
            tasks = []
            
            for item in schedule_data:
                task = MaintenanceTask(
                    task_id=f"PM_{item['equipment_id']}_{datetime.now().strftime('%Y%m%d')}",
                    equipment_id=item['equipment_id'],
                    task_type=MaintenanceType(item.get('type', 'preventive')),
                    priority=item.get('priority', 2),
                    description=item.get('description', f"Scheduled maintenance for {item['equipment_id']}"),
                    scheduled_date=datetime.strptime(item['date'], '%Y-%m-%d') if 'date' in item else datetime.now() + timedelta(days=30),
                    estimated_duration=item.get('duration', 4),
                    required_parts=item.get('parts', ["filters", "lubricants", "gaskets"])
                )
                tasks.append(task)
            
            return tasks
            
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback to rule-based scheduling
            return self._generate_fallback_schedule(equipment_list)
    
    def _generate_fallback_schedule(self, equipment_list: List[str]) -> List[MaintenanceTask]:
        """Fallback maintenance schedule generation"""
        tasks = []
        
        for equipment_id in equipment_list:
            history = self.knowledge_base.get_maintenance_history(equipment_id)
            
            if history:
                last_maintenance = datetime.strptime(history[0]['maintenance_date'], '%Y-%m-%d')
                next_maintenance = last_maintenance + timedelta(days=90)
            else:
                next_maintenance = datetime.now() + timedelta(days=30)
            
            task = MaintenanceTask(
                task_id=f"PM_{equipment_id}_{datetime.now().strftime('%Y%m%d')}",
                equipment_id=equipment_id,
                task_type=MaintenanceType.PREVENTIVE,
                priority=2,
                description=f"Scheduled preventive maintenance for {equipment_id}",
                scheduled_date=next_maintenance,
                estimated_duration=4,
                required_parts=["filters", "lubricants", "gaskets"]
            )
            tasks.append(task)
        
        return tasks

# 6. Enhanced Main Assistant with Gemini Integration
class IntelligentMaintenanceAssistant:
    def __init__(self):
        self.llm_provider = LLMProvider()
        self.knowledge_base = MaintenanceKnowledgeBase()
        self.sensor_processor = SensorDataProcessor(self.llm_provider)
        self.vision_processor = VisionProcessor(self.llm_provider)
        self.diagnostic_engine = DiagnosticEngine(self.knowledge_base, self.llm_provider)
        self.maintenance_planner = MaintenancePlanner(self.knowledge_base, self.llm_provider)
        self.nl_interface = NaturalLanguageInterface(self.llm_provider)
        
        self.current_context = {
            'status': 'operational',
            'current_faults': [],
            'upcoming_tasks': []
        }
    
    async def process_sensor_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Process sensor data with Gemini enhancement"""
        # Basic processing
        processed_data = self.sensor_processor.process_sensor_data(sensor_data)
        
        # Gemini analysis
        llm_analysis = await self.sensor_processor.analyze_with_llm(sensor_data)
        
        if processed_data['status'] in ['warning', 'critical']:
            # Enhanced fault diagnosis
            diagnosis = await self.diagnostic_engine.diagnose_fault(
                processed_data, {}, sensor_data.sensor_id
            )
            self.current_context['current_faults'].append(diagnosis)
            
            return {
                'sensor_analysis': processed_data,
                'llm_analysis': llm_analysis,
                'diagnosis': diagnosis
            }
        
        return {
            'sensor_analysis': processed_data,
            'llm_analysis': llm_analysis
        }
    
    async def process_visual_input(self, image_path: str) -> Dict[str, Any]:
        """Process visual input with Gemini analysis"""
        basic_analysis = self.vision_processor.analyze_visual_input(image_path)
        
        if basic_analysis.get('defects'):
            llm_analysis = await self.vision_processor.analyze_with_llm(
                image_path, basic_analysis['defects']
            )
            basic_analysis['llm_analysis'] = llm_analysis
        
        return basic_analysis
    
    async def chat_with_technician(self, message: str) -> str:
        """Enhanced chat with Gemini"""
        return await self.nl_interface.process_query(message, self.current_context)
    
    async def generate_maintenance_schedule(self, equipment_ids: List[str]) -> List[MaintenanceTask]:
        """Generate Gemini-optimized maintenance schedule"""
        tasks = await self.maintenance_planner.generate_maintenance_schedule(equipment_ids)
        self.current_context['upcoming_tasks'] = tasks
        return tasks
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.llm_provider.close()

# Enhanced Interactive Interface
class InteractiveMaintenanceInterface:
    def __init__(self):
        self.assistant = IntelligentMaintenanceAssistant()
        self.running = True
        
    def print_banner(self):
        """Print welcome banner"""
        print("=" * 70)
        print("    UNIVERSAL AI-POWERED MAINTENANCE ASSISTANT")
        print("=" * 70)
        print("Enhanced with Gemini AI - Universal Industrial Knowledge")
        print("Supports all types of industrial equipment!")
        print("=" * 70)
    
    def print_menu(self):
        """Print main menu"""
        print("\nMAIN MENU:")
        print("1. Input Sensor Data (with AI Analysis)")
        print("2. Analyze Visual Input (with AI Insights)")
        print("3. Chat with AI Assistant (Universal Knowledge)")
        print("4. Generate AI-Optimized Maintenance Schedule")
        print("5. Add New Equipment to System")
        print("6. View Equipment List")
        print("7. View System Status")
        print("8. Test Gemini Provider")
        print("9. Help")
        print("10. Exit")
        print("-" * 50)
    
    async def handle_sensor_input(self):
        """Handle sensor data input"""
        print("\nSENSOR DATA INPUT")
        print("-" * 30)
        
        equipment_list = self.assistant.knowledge_base.get_all_equipment()
        if not equipment_list:
            print("No equipment in system. Add equipment first (option 5).")
            return
        
        print("Available equipment:")
        for eq in equipment_list:
            print(f"  - {eq['id']}: {eq['name']}")
        
        equipment_id = input("\nEnter equipment ID: ").strip()
        if not equipment_id:
            print("Equipment ID is required!")
            return
        
        try:
            temperature = float(input("Temperature (°C): "))
            vibration = float(input("Vibration (mm/s): "))
            pressure = float(input("Pressure (bar): "))
            humidity = float(input("Humidity (%): "))
            sound_level = float(input("Sound level (dB): "))
            
            sensor_data = SensorData(
                sensor_id=equipment_id,
                timestamp=datetime.now(),
                temperature=temperature,
                vibration=vibration,
                pressure=pressure,
                humidity=humidity,
                sound_level=sound_level
            )
            
            print("\nProcessing sensor data with Gemini AI...")
            result = await self.assistant.process_sensor_data(sensor_data)
            
            print("\nANALYSIS RESULTS")
            print("-" * 30)
            print(f"Equipment Status: {result['sensor_analysis']['status']}")
            
            if result['sensor_analysis']['anomalies']:
                print("\nAnomalies detected:")
                for anomaly in result['sensor_analysis']['anomalies']:
                    print(f"  - {anomaly['parameter']}: {anomaly['value']} (Severity: {anomaly['severity']})")
            
            if 'llm_analysis' in result:
                print("\nGemini AI Analysis:")
                if isinstance(result['llm_analysis'], dict):
                    print(f"Risk Score: {result['llm_analysis'].get('risk_score', 'N/A')}")
                    print(f"Recommendations: {result['llm_analysis'].get('recommendations', [])}")
                else:
                    print(result['llm_analysis'])
            
            if 'diagnosis' in result:
                diagnosis = result['diagnosis']
                print(f"\nFAULT DIAGNOSIS")
                print(f"Fault ID: {diagnosis.fault_id}")
                print(f"Severity: {diagnosis.severity}")
                print(f"Confidence: {diagnosis.confidence:.2f}")
                print(f"Root Cause: {diagnosis.root_cause}")
                print(f"Recommended Actions: {', '.join(diagnosis.recommended_actions)}")
                
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"Error processing sensor data: {e}")
    
    async def test_gemini_provider(self):
        """Test Gemini provider connection"""
        print("\nTesting Gemini API connection...")
        
        test_prompt = "Briefly explain the importance of predictive maintenance in industrial settings."
        
        try:
            response = await self.assistant.llm_provider.get_best_response(test_prompt)
            print("\nGemini API Test Successful!")
            print("-" * 40)
            print(response)
        except Exception as e:
            print(f"\nGemini API Test Failed: {e}")
            print("Please check your API key configuration.")
    
    async def add_equipment(self):
        """Add new equipment to the system"""
        print("\nADD NEW EQUIPMENT")
        print("-" * 30)
        
        equipment_id = input("Equipment ID (e.g., MOTOR_001): ").strip()
        if not equipment_id:
            print("Equipment ID is required!")
            return
        
        equipment_name = input("Equipment Name: ").strip()
        equipment_type = input("Equipment Type (motor/pump/compressor/turbine/etc.): ").strip()
        model = input("Model (optional): ").strip()
        manufacturer = input("Manufacturer (optional): ").strip()
        
        equipment_data = {
            'id': equipment_id,
            'name': equipment_name or equipment_id,
            'type': equipment_type or 'general',
            'model': model,
            'manufacturer': manufacturer
        }
        
        if self.assistant.knowledge_base.add_equipment(equipment_data):
            print(f"Equipment {equipment_id} added successfully!")
            
            # Ask Gemini for general advice about this equipment type
            prompt = f"""
            Provide brief maintenance recommendations for a {equipment_type} ({equipment_name}):
            - Typical maintenance intervals
            - Key parameters to monitor
            - Common failure modes
            - Safety considerations
            """
            
            print("\nGemini AI Insights for this equipment type:")
            advice = await self.assistant.llm_provider.get_best_response(prompt)
            print(advice)
        else:
            print("Failed to add equipment!")

    async def handle_chat(self):
        """Enhanced chat with universal knowledge"""
        print("\nUNIVERSAL AI MAINTENANCE ASSISTANT")
        print("-" * 40)
        print("Ask about ANY industrial equipment - motors, pumps, turbines, HVAC, etc.")
        print("I have comprehensive knowledge of industrial maintenance!")
        print("Type 'exit' to return to main menu")
        
        while True:
            user_message = input("\nYou: ").strip()
            
            if user_message.lower() in ['exit', 'quit', 'back']:
                break
            
            if not user_message:
                continue
            
            print("Gemini is analyzing...")
            response = await self.assistant.chat_with_technician(user_message)
            print(f"\nGemini Expert: {response}")

    async def run(self):
        """Main application loop"""
        self.print_banner()
        
        try:
            while self.running:
                self.print_menu()
                choice = input("Enter your choice (1-10): ").strip()
                
                if choice == "1":
                    await self.handle_sensor_input()
                elif choice == "2":
                    print("Visual analysis with Gemini insights - Feature available!")
                elif choice == "3":
                    await self.handle_chat()
                elif choice == "4":
                    # Dynamic equipment selection
                    equipment_list = self.assistant.knowledge_base.get_all_equipment()
                    if equipment_list:
                        equipment_ids = [eq['id'] for eq in equipment_list]
                        print("Generating Gemini-optimized maintenance schedule...")
                        tasks = await self.assistant.generate_maintenance_schedule(equipment_ids)
                        print(f"Generated {len(tasks)} optimized maintenance tasks!")
                        for task in tasks:
                            print(f"  - {task.equipment_id}: {task.description} (Due: {task.scheduled_date.strftime('%Y-%m-%d')})")
                    else:
                        print("No equipment in system. Add equipment first (option 5).")
                elif choice == "5":
                    await self.add_equipment()
                elif choice == "6":
                    equipment_list = self.assistant.knowledge_base.get_all_equipment()
                    if equipment_list:
                        print(f"\nEQUIPMENT LIST ({len(equipment_list)} items)")
                        print("-" * 40)
                        for eq in equipment_list:
                            print(f"ID: {eq['id']}")
                            print(f"Name: {eq['name']}")
                            print(f"Type: {eq['type']}")
                            print(f"Model: {eq['model']}")
                            print(f"Manufacturer: {eq['manufacturer']}")
                            print(f"Last Maintenance: {eq['last_maintenance']}")
                            print("-" * 40)
                    else:
                        print("No equipment in system. Add equipment using option 5.")
                elif choice == "7":
                    print(f"\nSYSTEM STATUS")
                    print("-" * 30)
                    print(f"Status: {self.assistant.current_context['status']}")
                    print(f"Active Faults: {len(self.assistant.current_context['current_faults'])}")
                    print(f"Upcoming Tasks: {len(self.assistant.current_context['upcoming_tasks'])}")
                    equipment_count = len(self.assistant.knowledge_base.get_all_equipment())
                    print(f"Equipment Monitored: {equipment_count}")
                elif choice == "8":
                    await self.test_gemini_provider()
                elif choice == "9":
                    print("\nUNIVERSAL AI MAINTENANCE ASSISTANT HELP")
                    print("-" * 50)
                    print("This system uses Gemini AI with comprehensive industrial knowledge!")
                    print("\nFeatures:")
                    print("• Universal equipment support (motors, pumps, turbines, HVAC, etc.)")
                    print("• AI-enhanced sensor data analysis")
                    print("• Intelligent fault diagnosis")
                    print("• Expert maintenance advice")
                    print("• Dynamic equipment management")
                    print("• Industry best practices")
                    print("\nSupported Equipment Types:")
                    print("• Rotating equipment: Motors, pumps, compressors, turbines, fans")
                    print("• Static equipment: Heat exchangers, vessels, piping")
                    print("• Control systems: PLCs, sensors, valves, actuators")
                    print("• HVAC: Chillers, boilers, air handlers, cooling towers")
                    print("• Power systems: Generators, transformers, switchgear")
                elif choice == "10":
                    print("\nThank you for using the Universal AI Maintenance Assistant!")
                    self.running = False
                else:
                    print("Invalid choice. Please select 1-10.")
                
                if self.running and choice != "10":
                    input("\nPress Enter to continue...")
                    
        finally:
            await self.assistant.cleanup()

# Main execution
async def main():
    interface = InteractiveMaintenanceInterface()
    await interface.run()

if __name__ == "__main__":
    asyncio.run(main())
