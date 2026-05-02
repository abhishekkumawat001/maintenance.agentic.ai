"""
LLM Provider — Gemini API integration with async support and retry logic.
"""

import asyncio
import logging
import time
from typing import Optional

import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel

from app.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMProvider:
    """Manages Gemini API connection with retry logic and async support."""

    def __init__(self):
        self.gemini_model: Optional[GenerativeModel] = None
        self._init_gemini()

    def _init_gemini(self):
        """Initialize Gemini provider"""
        try:
            if LLMConfig.GEMINI_API_KEY:
                genai.configure(api_key=LLMConfig.GEMINI_API_KEY)
                self.gemini_model = GenerativeModel(LLMConfig.GEMINI_MODEL)
                logger.info("Gemini model initialized successfully (%s)", LLMConfig.GEMINI_MODEL)
            else:
                logger.warning("GEMINI_API_KEY not set — LLM features disabled")
        except Exception as e:
            logger.warning("Error initializing Gemini: %s", e)
            self.gemini_model = None

    async def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call Gemini API with retry logic and exponential backoff.
        Uses asyncio.to_thread to avoid blocking the event loop.
        """
        if not self.gemini_model:
            return "⚠️ Gemini API not configured. Set GEMINI_API_KEY in your .env file."

        last_error = None
        for attempt in range(max_retries):
            try:
                # Run synchronous Gemini call in a thread to keep async non-blocking
                response = await asyncio.to_thread(
                    self.gemini_model.generate_content, prompt
                )
                return response.text
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "Gemini API error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, max_retries, e, wait_time
                )
                await asyncio.sleep(wait_time)

        logger.error("Gemini API failed after %d attempts: %s", max_retries, last_error)
        return f"Error calling Gemini after {max_retries} attempts: {last_error}"

    async def generate_with_image(self, prompt: str, image) -> str:
        """Call Gemini with multimodal input (text + image)."""
        if not self.gemini_model:
            return "⚠️ Gemini API not configured."
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content, [prompt, image]
            )
            return response.text
        except Exception as e:
            logger.error("Gemini multimodal error: %s", e)
            return f"Error with multimodal analysis: {e}"

    async def close(self):
        """Cleanup resources"""
        pass
