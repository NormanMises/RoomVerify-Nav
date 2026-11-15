from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        self._client = OpenAI(api_key=api_key, base_url=api_base)

    @classmethod
    def from_env(cls) -> Optional["OpenAIClient"]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        api_base = os.environ.get("OPENAI_API_BASE", "").strip()

        if not api_key or not api_base:
            return None

        return cls(api_key=api_key, api_base=api_base)

    def call_openai_api(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        model = os.environ.get("OPENAI_MODEL")
        temperature = float(os.environ.get("OPENAI_TEMPERATURE"))
        enable_thinking = os.environ.get("OPENAI_ENABLE_THINKING", "0").strip() == "1"

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "extra_body": {
                "enable_thinking": True if enable_thinking else False,
                "thinking": {
                    "type": "enabled" if enable_thinking else "disabled",
                },
            }
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content

        if not isinstance(content, str):
            raise RuntimeError("missing choices[0].message.content")

        data = json.loads(content)

        if not isinstance(data, dict):
            raise RuntimeError(f"expected JSON object, got: {type(data)}")

        return data

    def call_openai_vision_api(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: np.ndarray,
        json_mode: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        model_name = model or os.environ.get("OPENAI_VISION_MODEL", "").strip()
        temperature = float(os.environ.get("OPENAI_TEMPERATURE"))
        enable_thinking = os.environ.get("OPENAI_ENABLE_THINKING", "0").strip() == "1"

        success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("failed to encode image for vision API")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        image_url = f"data:image/png;base64,{image_b64}"

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "temperature": temperature,
            "extra_body": {
                "enable_thinking": True if enable_thinking else False,
                "thinking": {
                    "type": "enabled" if enable_thinking else "disabled",
                },
            },
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content

        if not isinstance(content, str):
            raise RuntimeError("missing choices[0].message.content")

        if not json_mode:
            return content

        data = json.loads(content)
        if not isinstance(data, dict):
            raise RuntimeError(f"expected JSON object, got: {type(data)}")
        return data


# def main() -> None:
#     import argparse
#     import time
#     from icecream import ic

#     parser = argparse.ArgumentParser(description="Test OpenAI /v1/chat/completions JSON mode.")
#     parser.add_argument(
#         "--system",
#         help="System prompt (should force pure JSON output).",
#     )
#     parser.add_argument(
#         "--user",
#         help="User prompt.",
#     )
#     args = parser.parse_args()

#     client = OpenAIClient.from_env()
#     if client is None:
#         raise RuntimeError("Missing OPENAI_API_KEY / OPENAI_API_BASE in environment.")

#     result = client.call_openai_api(system_prompt=args.system, user_prompt=args.user)


# if __name__ == "__main__":
#     main()
