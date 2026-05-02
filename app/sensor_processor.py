"""
Sensor Data Processor — Anomaly detection and LLM-enhanced analysis.
"""

import json
import logging
from typing import Any, Dict, List

from app.config import SensorThresholds
from app.llm_provider import LLMProvider
from app.models import SensorData

logger = logging.getLogger(__name__)


class SensorDataProcessor:
    """Processes sensor readings, detects anomalies, and provides LLM analysis."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.thresholds = SensorThresholds.THRESHOLDS

    def process_sensor_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Process incoming sensor data and detect anomalies."""
        anomalies = []

        for param, value in [
            ('temperature', sensor_data.temperature),
            ('vibration', sensor_data.vibration),
            ('pressure', sensor_data.pressure),
            ('humidity', sensor_data.humidity)
        ]:
            limits = self.thresholds[param]
            if not (limits['min'] <= value <= limits['max']):
                anomalies.append({
                    'parameter': param,
                    'value': value,
                    'threshold': limits,
                    'severity': 'high' if value > limits['max'] * 1.2 else 'medium'
                })

        return {
            'sensor_id': sensor_data.sensor_id,
            'timestamp': sensor_data.timestamp,
            'anomalies': anomalies,
            'status': 'critical' if len(anomalies) > 2 else 'warning' if anomalies else 'normal'
        }

    async def analyze_with_llm(self, sensor_data: SensorData) -> Dict[str, Any]:
        """Use Gemini for advanced sensor data analysis."""
        prompt = f"""
        Analyze the following industrial equipment sensor data:

        Equipment: {sensor_data.sensor_id}
        Temperature: {sensor_data.temperature}°C (normal: 0-80°C)
        Vibration: {sensor_data.vibration} mm/s (normal: 0-10 mm/s)
        Pressure: {sensor_data.pressure} bar (normal: 0-100 bar)
        Humidity: {sensor_data.humidity}% (normal: 0-100%)
        Sound Level: {sensor_data.sound_level} dB

        Provide:
        1. Anomaly detection and severity assessment
        2. Potential failure modes
        3. Recommended immediate actions
        4. Risk assessment (1-10 scale)

        Format as JSON with keys: anomalies, failure_modes, recommendations, risk_score
        """

        response = await self.llm_provider.generate(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "llm_analysis": response,
                "anomalies": [],
                "failure_modes": ["Analysis available in text format"],
                "recommendations": ["Review analysis above"],
                "risk_score": 5
            }
