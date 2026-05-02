"""
Maintenance Planner — Gemini-optimized maintenance scheduling.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List

from app.knowledge_base import MaintenanceKnowledgeBase
from app.llm_provider import LLMProvider
from app.models import MaintenanceTask, MaintenanceType

logger = logging.getLogger(__name__)


class MaintenancePlanner:
    """Generates AI-optimized maintenance schedules."""

    def __init__(self, knowledge_base: MaintenanceKnowledgeBase, llm_provider: LLMProvider):
        self.knowledge_base = knowledge_base
        self.llm_provider = llm_provider

    async def generate_maintenance_schedule(self, equipment_list: List[str]) -> List[MaintenanceTask]:
        """Generate Gemini-optimized maintenance schedule."""

        all_equipment = self.knowledge_base.get_all_equipment()
        equipment_data = {eq['id']: eq for eq in all_equipment}

        history_data = {}
        for eq_id in equipment_list:
            history_data[eq_id] = self.knowledge_base.get_maintenance_history(eq_id)

        prompt = f"""
        As a maintenance planning expert, create an optimized maintenance schedule:

        EQUIPMENT LIST: {equipment_list}

        EQUIPMENT DETAILS:
        {json.dumps({eq_id: equipment_data.get(eq_id, {}) for eq_id in equipment_list}, indent=2, default=str)}

        MAINTENANCE HISTORY:
        {json.dumps(history_data, indent=2, default=str)}

        Create a maintenance schedule considering:
        1. Equipment criticality and type
        2. Historical maintenance patterns
        3. Manufacturer recommendations
        4. Optimal resource utilization
        5. Minimal production disruption

        For each equipment, specify:
        - Next maintenance date (YYYY-MM-DD format)
        - Priority level (1-5)
        - Estimated duration (hours)
        - Required parts/materials
        - Maintenance type (preventive/predictive/corrective)

        Format as JSON array with keys: equipment_id, date, priority, duration, parts, type, description
        """

        response = await self.llm_provider.generate(prompt)

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
                    scheduled_date=(
                        datetime.strptime(item['date'], '%Y-%m-%d')
                        if 'date' in item
                        else datetime.now() + timedelta(days=30)
                    ),
                    estimated_duration=item.get('duration', 4),
                    required_parts=item.get('parts', ["filters", "lubricants", "gaskets"])
                )
                tasks.append(task)
            return tasks
        except (json.JSONDecodeError, KeyError, ValueError):
            return self._generate_fallback_schedule(equipment_list)

    def _generate_fallback_schedule(self, equipment_list: List[str]) -> List[MaintenanceTask]:
        """Fallback rule-based scheduling."""
        tasks = []
        for equipment_id in equipment_list:
            history = self.knowledge_base.get_maintenance_history(equipment_id)
            if history:
                last_date = datetime.strptime(history[0]['maintenance_date'], '%Y-%m-%d')
                next_date = last_date + timedelta(days=90)
            else:
                next_date = datetime.now() + timedelta(days=30)

            tasks.append(MaintenanceTask(
                task_id=f"PM_{equipment_id}_{datetime.now().strftime('%Y%m%d')}",
                equipment_id=equipment_id,
                task_type=MaintenanceType.PREVENTIVE,
                priority=2,
                description=f"Scheduled preventive maintenance for {equipment_id}",
                scheduled_date=next_date,
                estimated_duration=4,
                required_parts=["filters", "lubricants", "gaskets"]
            ))
        return tasks
