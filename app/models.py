"""
Data models for the Maintenance AI system.
Dataclasses for sensor data, fault diagnosis, maintenance tasks, and documents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


@dataclass
class SensorData:
    """Raw sensor reading from equipment"""
    sensor_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    humidity: float
    sound_level: float


@dataclass
class FaultDiagnosis:
    """Result of fault diagnosis analysis"""
    fault_id: str
    severity: str
    confidence: float
    description: str
    root_cause: str
    recommended_actions: List[str]
    estimated_downtime: int  # hours


class MaintenanceType(Enum):
    """Types of maintenance activities"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"


@dataclass
class MaintenanceTask:
    """A scheduled maintenance activity"""
    task_id: str
    equipment_id: str
    task_type: MaintenanceType
    priority: int
    description: str
    scheduled_date: datetime
    estimated_duration: int  # hours
    required_parts: List[str]
    assigned_technician: Optional[str] = None


@dataclass
class DocumentChunk:
    """A chunk of text from an ingested document"""
    chunk_id: str
    document_name: str
    content: str
    metadata: Dict = field(default_factory=dict)
    chunk_index: int = 0


@dataclass
class ChatMessage:
    """A single message in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[Dict] = field(default_factory=list)  # RAG source docs
