import os
import json
from openai import OpenAI


class ImageChecker:
    def __init__(self):
        # Read OPENAI_API_KEY from environment
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self.client = OpenAI(api_key=key)

        # Your existing prompt for detecting image‐analysis requests:
        self.system_prompt = """You are an advanced visual request analyzer. Detect if the user needs:
1. Examination/analysis ("yes") - including help requests to look at something
2. Any other request ("no")

Respond with JSON containing:
{
  "type": "yes" | "no",
  "reason": "clear explanation"
}

Examples of YES requests:
- "Can you check what's in this picture?"
- "Help me analyze these test results"
- "What's wrong with this X-ray?"
- "Take a look at this strange mark"
- "Could you examine this photo?"
- "I need help identifying something in this image"

Examples of NO requests:
- "Explain quantum physics"
- "How to make lasagna?"
- "What's the weather forecast?"
- "Tell me a joke"
- "Recommend exercise routines"

Response template:
User: "Can you look at this rash?" → 
{
  "type": "yes",
  "reason": "Medical image analysis request"
}

User: "What's in this document?" → 
{
  "type": "yes",
  "reason": "Request to examine document image"
}

User: "How do batteries work?" → 
{
  "type": "no",
  "reason": "General science question"
}"""

    def classify(self, text: str):
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            return {"type": "no", "reason": f"classification error: {e}"}
