"""
Vision Processor — Visual defect detection with OpenCV and Gemini multimodal.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np

from app.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Analyzes equipment images for defects using CV and Gemini vision."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def analyze_visual_input(self, image_path: str) -> Dict[str, Any]:
        """Analyze camera feed for visual defects using OpenCV edge detection."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            defects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    defects.append({
                        'location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'area': float(area),
                        'type': 'surface_defect'
                    })

            return {
                'defects_found': len(defects),
                'defects': defects,
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("Vision processing error: %s", e)
            return {"error": str(e)}

    async def analyze_with_llm(self, image_path: str, defects: List[Dict]) -> str:
        """Use Gemini to analyze visual defects (text-based)."""
        prompt = "Analyze the following visual inspection results for industrial equipment:\n\n"
        prompt += f"Image: {image_path}\nDefects found: {len(defects)}\n\n"

        for i, defect in enumerate(defects, 1):
            prompt += (
                f"Defect {i}: Type={defect['type']}, "
                f"Location=({defect['location']['x']}, {defect['location']['y']}), "
                f"Size={defect['location']['width']}x{defect['location']['height']}, "
                f"Area={defect['area']} px\n"
            )

        prompt += """
        Provide:
        1. Severity assessment of detected defects
        2. Potential causes
        3. Recommended maintenance actions
        4. Expected progression if untreated
        """

        return await self.llm_provider.generate(prompt)

    async def analyze_with_gemini_vision(self, image_path: str) -> str:
        """Send actual image to Gemini multimodal API for analysis."""
        try:
            from PIL import Image
            image = Image.open(image_path)
            return await self.llm_provider.generate_with_image(
                "Analyze this industrial equipment image. Identify any visible defects, "
                "wear patterns, corrosion, leaks, or maintenance concerns. "
                "Provide severity assessment and recommended actions.",
                image
            )
        except Exception as e:
            logger.error("Gemini vision analysis error: %s", e)
            return f"Vision analysis error: {e}"
