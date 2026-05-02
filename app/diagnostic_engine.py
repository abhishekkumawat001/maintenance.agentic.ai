"""
Diagnostic Engine — Gemini-enhanced root cause analysis.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict

from app.knowledge_base import MaintenanceKnowledgeBase
from app.llm_provider import LLMProvider
from app.models import FaultDiagnosis

logger = logging.getLogger(__name__)


class DiagnosticEngine:
    """Performs fault diagnosis using knowledge base + Gemini analysis."""

    def __init__(self, knowledge_base: MaintenanceKnowledgeBase, llm_provider: LLMProvider):
        self.knowledge_base = knowledge_base
        self.llm_provider = llm_provider

    async def diagnose_fault(self, sensor_data: Dict, visual_data: Dict,
                              equipment_id: str) -> FaultDiagnosis:
        """Perform Gemini-enhanced root cause analysis."""

        # Get equipment info
        equipment_list = self.knowledge_base.get_all_equipment()
        equipment = next((eq for eq in equipment_list if eq['id'] == equipment_id), None)

        # Get maintenance history
        history = self.knowledge_base.get_maintenance_history(equipment_id)

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
        {json.dumps(visual_data, indent=2, default=str)}

        MAINTENANCE HISTORY:
        {json.dumps(history[:3], indent=2, default=str) if history else 'No recent history'}

        Provide a comprehensive fault diagnosis including:
        1. Most likely root cause
        2. Confidence level (0-1)
        3. Severity (low/medium/high)
        4. Specific recommended actions (as a list)
        5. Estimated repair time in hours
        6. Risk if left unaddressed

        Format as JSON with keys: root_cause, confidence, severity, actions, estimated_hours, risk_description
        """

        llm_response = await self.llm_provider.generate(prompt)

        try:
            diagnosis_data = json.loads(llm_response)
            return FaultDiagnosis(
                fault_id=f"F_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=diagnosis_data.get('severity', 'medium'),
                confidence=float(diagnosis_data.get('confidence', 0.7)),
                description=f"AI-diagnosed fault for {equipment_id}",
                root_cause=diagnosis_data.get('root_cause', 'Unknown cause'),
                recommended_actions=diagnosis_data.get('actions', ['Schedule inspection']),
                estimated_downtime=int(diagnosis_data.get('estimated_hours', 4))
            )
        except (json.JSONDecodeError, ValueError):
            return self._fallback_diagnosis(sensor_data, equipment_id)

    def _fallback_diagnosis(self, sensor_data: Dict, equipment_id: str) -> FaultDiagnosis:
        """Rule-based fallback when LLM JSON parsing fails."""
        symptoms = []
        for anomaly in sensor_data.get('anomalies', []):
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
            root_cause = "Unable to determine — schedule expert inspection"
            recommended_actions = ["Schedule expert inspection", "Review sensor trends"]

        severity = "high" if any(
            a.get('severity') == 'high' for a in sensor_data.get('anomalies', [])
        ) else "medium"

        return FaultDiagnosis(
            fault_id=f"F_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            severity=severity,
            confidence=confidence,
            description=f"Fault detected with symptoms: {symptoms_str}",
            root_cause=root_cause,
            recommended_actions=recommended_actions,
            estimated_downtime=8 if severity == "high" else 4
        )
