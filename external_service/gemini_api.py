from google import genai
from google.genai import types
from typing import Tuple

from external_service.base_api import BaseAPIClient
from utils.config import GEMINI_CREDENTIALS, GEMINI_MODEL, GEMINI_THINKING_BUDGET, GOOGLE_PROJECT_ID, GOOGLE_LOCATION
from utils.constants import MESSAGES
from utils.exceptions import APIError


class GeminiAPIClient(BaseAPIClient):
    def __init__(self):
        super().__init__(GEMINI_CREDENTIALS, GEMINI_MODEL)
        self.client = None

    def initialize(self) -> bool:
        try:
            # Vertex AI使用時の必須環境変数チェック
            if not GOOGLE_PROJECT_ID:
                raise APIError(MESSAGES["GOOGLE_PROJECT_ID_MISSING"])

            if not GOOGLE_LOCATION:
                raise APIError(MESSAGES["GOOGLE_LOCATION_MISSING"])

            # Vertex AI Clientの初期化
            self.client = genai.Client(
                vertexai=True,
                project=GOOGLE_PROJECT_ID,
                location=GOOGLE_LOCATION,
            )
            return True

        except APIError:
            # APIErrorはそのまま再raise
            raise
        except Exception as e:
            raise APIError(MESSAGES["VERTEX_AI_INIT_ERROR"].format(error=str(e)))

    def _generate_content(self, prompt: str, model_name: str) -> Tuple[str, int, int]:
        try:
            if GEMINI_THINKING_BUDGET:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=GEMINI_THINKING_BUDGET
                        )
                    )
                )
            else:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

            if hasattr(response, 'text'):
                summary_text = response.text
            else:
                summary_text = str(response)

            input_tokens = 0
            output_tokens = 0

            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

            return summary_text, input_tokens, output_tokens

        except Exception as e:
            raise APIError(MESSAGES["VERTEX_AI_API_ERROR"].format(error=str(e)))
