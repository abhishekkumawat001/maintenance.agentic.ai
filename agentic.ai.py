import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sqlite3
import numpy as np

# Optional imports with fallbacks
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV not available. Visual processing will be disabled.")
    OPENCV_AVAILABLE = False

try:
    from transformers.pipelines import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not available. Some AI features will be disabled.")
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI not available. Advanced NLP features will be disabled.")
    OPENAI_AVAILABLE = False

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

# 1. Perception Module
class SensorDataProcessor:
    def __init__(self):
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

class VisionProcessor:
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                from transformers.pipelines import pipeline
                self.image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
            except Exception as e:
                logger.warning(f"Could not load image classifier: {e}")
                self.image_classifier = None
        else:
            self.image_classifier = None
    
    def analyze_visual_input(self, image_path: str) -> Dict[str, Any]:
        """Analyze camera feed for visual defects"""
        if not OPENCV_AVAILABLE:
            return {"error": "OpenCV not available"}
        
        try:
            import cv2  # Ensure cv2 is imported in this scope
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
                        'location': {'x': x, 'y': y, 'width': w, 'height': h},
                        'area': area,
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

# 2. Knowledge Base Module
class MaintenanceKnowledgeBase:
    def __init__(self, db_path: str = "maintenance.db"):
        self.db_path = db_path
        self.init_database()
        self.populate_sample_data()
    
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
    
    def populate_sample_data(self):
        """Populate database with sample data for demo"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM fault_patterns")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
        
        # Insert sample fault patterns
        sample_patterns = [
            ("motor", "temperature high vibration high", "bearing_failure", "Replace bearings;Check alignment;Lubricate", 0.85),
            ("pump", "pressure medium noise", "impeller_wear", "Inspect impeller;Check clearances;Replace if worn", 0.75),
            ("compressor", "vibration high temperature high", "mechanical_failure", "Stop operation;Inspect internals;Schedule overhaul", 0.90),
            ("motor", "temperature high", "cooling_system_failure", "Check cooling fans;Clean air filters;Check coolant", 0.70)
        ]
        
        cursor.executemany('''
            INSERT INTO fault_patterns (equipment_type, symptoms, root_cause, solution, confidence_score)
            VALUES (?, ?, ?, ?, ?)
        ''', sample_patterns)
        
        # Insert sample equipment
        sample_equipment = [
            ("MOTOR_001", "Main Drive Motor", "motor", "IE3-180M", "Siemens", "2020-01-15", "2024-10-15"),
            ("PUMP_002", "Cooling Water Pump", "pump", "NK100-200", "Grundfos", "2020-03-20", "2024-09-10"),
            ("COMPRESSOR_003", "Air Compressor", "compressor", "GA22", "Atlas Copco", "2020-06-10", "2024-11-01")
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO equipment (id, name, type, model, manufacturer, installation_date, last_maintenance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', sample_equipment)
        
        conn.commit()
        conn.close()
        logger.info("Sample data populated successfully")
    
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
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in history]
    
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
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in patterns]

# 3. Reasoning/Planning Agent
class DiagnosticEngine:
    def __init__(self, knowledge_base: MaintenanceKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.fault_tree_analyzer = FaultTreeAnalyzer()
    
    def diagnose_fault(self, sensor_data: Dict, visual_data: Dict, equipment_id: str) -> FaultDiagnosis:
        """Perform root cause analysis and generate diagnosis"""
        symptoms = []
        
        # Analyze sensor anomalies
        if sensor_data.get('anomalies'):
            for anomaly in sensor_data['anomalies']:
                symptoms.append(f"{anomaly['parameter']} {anomaly['severity']}")
        
        # Analyze visual defects
        if visual_data.get('defects'):
            symptoms.append(f"visual_defects_{len(visual_data['defects'])}")
        
        # Query knowledge base for similar patterns
        symptoms_str = " ".join(symptoms)
        similar_faults = self.knowledge_base.find_similar_faults(symptoms_str)
        
        if similar_faults:
            best_match = similar_faults[0]
            confidence = best_match['confidence_score']
            root_cause = best_match['root_cause']
            recommended_actions = best_match['solution'].split(';')
        else:
            # Fallback to rule-based diagnosis
            confidence = 0.6
            root_cause = "Unknown - requires expert analysis"
            recommended_actions = ["Schedule inspection", "Check equipment manual"]
        
        # Determine severity
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

class FaultTreeAnalyzer:
    def __init__(self):
        self.fault_trees = {
            'motor_failure': {
                'root_causes': ['bearing_wear', 'overheating', 'electrical_fault'],
                'symptoms': ['high_vibration', 'temperature_rise', 'noise']
            },
            'pump_failure': {
                'root_causes': ['seal_leak', 'impeller_wear', 'cavitation'],
                'symptoms': ['pressure_drop', 'flow_reduction', 'noise']
            }
        }
    
    def analyze(self, symptoms: List[str]) -> Dict[str, float]:
        """Analyze fault tree and return probability scores"""
        probabilities = {}
        for fault_type, fault_data in self.fault_trees.items():
            matching_symptoms = set(symptoms) & set(fault_data['symptoms'])
            probability = len(matching_symptoms) / len(fault_data['symptoms'])
            probabilities[fault_type] = probability
        
        return probabilities

class MaintenancePlanner:
    def __init__(self, knowledge_base: MaintenanceKnowledgeBase):
        self.knowledge_base = knowledge_base
    
    def generate_maintenance_schedule(self, equipment_list: List[str]) -> List[MaintenanceTask]:
        """Generate preventive maintenance schedule"""
        tasks = []
        
        for equipment_id in equipment_list:
            history = self.knowledge_base.get_maintenance_history(equipment_id)
            
            # Calculate next maintenance date based on history
            if history:
                last_maintenance = datetime.strptime(history[0]['maintenance_date'], '%Y-%m-%d')
                next_maintenance = last_maintenance + timedelta(days=90)  # 3-month cycle
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

# 4. Action Layer
class MaintenanceRecommendationEngine:
    def __init__(self):
        self.action_templates = {
            'high_temperature': [
                "Check cooling system",
                "Inspect air filters", 
                "Verify ventilation",
                "Check thermal sensors"
            ],
            'high_vibration': [
                "Check bearing alignment",
                "Inspect mounting bolts",
                "Balance rotating components",
                "Check for wear"
            ],
            'pressure_anomaly': [
                "Check seals and gaskets",
                "Inspect pressure sensors",
                "Verify system pressure",
                "Check for leaks"
            ]
        }
    
    def generate_recommendations(self, diagnosis: FaultDiagnosis) -> Dict[str, Any]:
        """Generate detailed maintenance recommendations"""
        recommendations = {
            'immediate_actions': diagnosis.recommended_actions,
            'required_tools': ["multimeter", "vibration_analyzer", "thermal_camera"],
            'safety_precautions': ["Lockout/Tagout", "PPE required", "Confined space entry"],
            'estimated_cost': self._estimate_cost(diagnosis),
            'priority_level': diagnosis.severity
        }
        
        return recommendations
    
    def _estimate_cost(self, diagnosis: FaultDiagnosis) -> float:
        """Estimate maintenance cost based on diagnosis"""
        base_cost = 500.0
        severity_multiplier = {'low': 1.0, 'medium': 1.5, 'high': 3.0}
        
        return base_cost * severity_multiplier.get(diagnosis.severity, 1.0)

# 5. User Interface Module
class NaturalLanguageInterface:
    def __init__(self):
        self.conversation_history = []
    
    async def process_query(self, user_input: str, context: Dict) -> str:
        """Process natural language queries from technicians"""
        user_input_lower = user_input.lower()
        
        if "status" in user_input_lower:
            return self._get_system_status(context)
        elif "fault" in user_input_lower or "problem" in user_input_lower:
            return self._get_fault_information(context)
        elif "maintenance" in user_input_lower:
            return self._get_maintenance_schedule(context)
        elif "help" in user_input_lower:
            return self._get_help_message()
        else:
            return "I understand you're asking about equipment maintenance. Could you please specify if you need status information, fault diagnosis, or maintenance scheduling?"
    
    def _get_system_status(self, context: Dict) -> str:
        """Get current system status"""
        return f"System Status: {context.get('status', 'Unknown')}\nLast Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _get_fault_information(self, context: Dict) -> str:
        """Get fault information"""
        if context.get('current_faults'):
            faults = context['current_faults']
            return f"Active Faults: {len(faults)}\nMost Critical: {faults[0].description if faults else 'None'}"
        return "No active faults detected."
    
    def _get_maintenance_schedule(self, context: Dict) -> str:
        """Get maintenance schedule"""
        if context.get('upcoming_tasks'):
            tasks = context['upcoming_tasks']
            next_task = tasks[0] if tasks else None
            if next_task:
                return f"Next Maintenance: {next_task.description}\nScheduled: {next_task.scheduled_date.strftime('%Y-%m-%d')}"
        return "No upcoming maintenance scheduled."
    
    def _get_help_message(self) -> str:
        """Get help message"""
        return """
Available Commands:
- "What's the system status?" - Get current system status
- "Are there any faults?" - Get current fault information  
- "When is the next maintenance?" - Get maintenance schedule
- "Help" - Show this message
        """

# 6. Feedback & Learning Loop
class LearningEngine:
    def __init__(self, knowledge_base: MaintenanceKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.feedback_data = []
    
    def capture_feedback(self, diagnosis_id: str, actual_outcome: str, technician_notes: str):
        """Capture feedback from technicians"""
        feedback = {
            'diagnosis_id': diagnosis_id,
            'timestamp': datetime.now(),
            'actual_outcome': actual_outcome,
            'technician_notes': technician_notes,
            'success': 'resolved' in actual_outcome.lower()
        }
        self.feedback_data.append(feedback)
        logger.info(f"Feedback captured for diagnosis {diagnosis_id}")
    
    def update_models(self):
        """Update prediction models based on feedback"""
        if len(self.feedback_data) >= 10:  # Minimum feedback for update
            success_rate = sum(1 for f in self.feedback_data if f['success']) / len(self.feedback_data)
            logger.info(f"Current model success rate: {success_rate:.2%}")
            
            # In production, retrain ML models here
            self.feedback_data.clear()

# Main Intelligent Maintenance Assistant
class IntelligentMaintenanceAssistant:
    def __init__(self):
        self.knowledge_base = MaintenanceKnowledgeBase()
        self.sensor_processor = SensorDataProcessor()
        self.vision_processor = VisionProcessor()
        self.diagnostic_engine = DiagnosticEngine(self.knowledge_base)
        self.maintenance_planner = MaintenancePlanner(self.knowledge_base)
        self.recommendation_engine = MaintenanceRecommendationEngine()
        self.nl_interface = NaturalLanguageInterface()
        self.learning_engine = LearningEngine(self.knowledge_base)
        
        self.current_context = {
            'status': 'operational',
            'current_faults': [],
            'upcoming_tasks': []
        }
    
    async def process_sensor_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Process incoming sensor data"""
        processed_data = self.sensor_processor.process_sensor_data(sensor_data)
        
        if processed_data['status'] in ['warning', 'critical']:
            # Trigger fault diagnosis
            diagnosis = self.diagnostic_engine.diagnose_fault(
                processed_data, {}, sensor_data.sensor_id
            )
            self.current_context['current_faults'].append(diagnosis)
            
            # Generate recommendations
            recommendations = self.recommendation_engine.generate_recommendations(diagnosis)
            
            return {
                'sensor_analysis': processed_data,
                'diagnosis': diagnosis,
                'recommendations': recommendations
            }
        
        return {'sensor_analysis': processed_data}
    
    async def process_visual_input(self, image_path: str) -> Dict[str, Any]:
        """Process visual input from cameras"""
        return self.vision_processor.analyze_visual_input(image_path)
    
    async def chat_with_technician(self, message: str) -> str:
        """Handle natural language interaction"""
        return await self.nl_interface.process_query(message, self.current_context)
    
    async def generate_maintenance_schedule(self, equipment_ids: List[str]) -> List[MaintenanceTask]:
        """Generate maintenance schedule"""
        tasks = self.maintenance_planner.generate_maintenance_schedule(equipment_ids)
        self.current_context['upcoming_tasks'] = tasks
        return tasks
    
    def provide_feedback(self, diagnosis_id: str, outcome: str, notes: str):
        """Provide feedback for continuous learning"""
        self.learning_engine.capture_feedback(diagnosis_id, outcome, notes)

# Demo/Testing Functions
async def demo_maintenance_assistant():
    """Demo function to test the maintenance assistant"""
    assistant = IntelligentMaintenanceAssistant()
    
    print("=== Intelligent Maintenance Assistant Demo ===")
    print("Dependencies loaded:")
    print(f"- OpenCV: {'✓' if OPENCV_AVAILABLE else '✗'}")
    print(f"- Transformers: {'✓' if TRANSFORMERS_AVAILABLE else '✗'}")
    print(f"- OpenAI: {'✓' if OPENAI_AVAILABLE else '✗'}")
    
    # Test sensor data processing
    print("\n1. Processing sensor data...")
    sensor_data = SensorData(
        sensor_id="MOTOR_001",
        timestamp=datetime.now(),
        temperature=85.0,  # High temperature
        vibration=12.0,    # High vibration
        pressure=45.0,
        humidity=65.0,
        sound_level=80.0
    )
    
    result = await assistant.process_sensor_data(sensor_data)
    if result.get('diagnosis'):
        print(f"Fault diagnosed: {result['diagnosis'].description}")
        print(f"Severity: {result['diagnosis'].severity}")
        print(f"Root cause: {result['diagnosis'].root_cause}")
        print(f"Recommendations: {result['diagnosis'].recommended_actions}")
        print(f"Estimated cost: ${result['recommendations']['estimated_cost']}")
    
    # Test natural language interface
    print("\n2. Testing natural language interface...")
    queries = [
        "What's the system status?",
        "Are there any faults?",
        "When is the next maintenance?",
        "Help"
    ]
    
    for query in queries:
        response = await assistant.chat_with_technician(query)
        print(f"Q: {query}")
        print(f"A: {response}\n")
    
    # Test maintenance scheduling
    print("3. Generating maintenance schedule...")
    equipment_ids = ["MOTOR_001", "PUMP_002", "COMPRESSOR_003"]
    tasks = await assistant.generate_maintenance_schedule(equipment_ids)
    
    for task in tasks:
        print(f"Task: {task.description}")
        print(f"Scheduled: {task.scheduled_date.strftime('%Y-%m-%d')}")
        print(f"Duration: {task.estimated_duration} hours")
        print(f"Parts needed: {', '.join(task.required_parts)}\n")

if __name__ == "__main__":
    # Run the demo
    print("Starting Intelligent Maintenance Assistant...")
    asyncio.run(demo_maintenance_assistant())