import os
import logging
from openai import OpenAI
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class DoubaoClient():
    def __init__(self, api_key: str = None):
        self.api_key=os.environ.get("ARK_API_KEY")
        self.model="doubao-1.5-vision-pro-250328"
        self.api_url="https://ark.cn-beijing.volces.com/api/v3"
        self.client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )
        logging.info("DoubaoClient initialized.")

    def chat_completion(self, messages: list, **kwargs) -> Optional[str]:
        """Specific chat completion method for Doubao using OpenAI client."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logging.error("Failed to get content from Doubao API response.")
                return None
        except Exception as e:
            logging.error(f"Doubao API call failed: {e}")
            return None
