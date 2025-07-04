import google.generativeai as genai

prompt = f"""
A medical image classification model processed a chest X-ray and returned the following prediction:
- Pneumonia: 87%
- Normal: 13%

Write a short, friendly summary of what this means for a non-technical user. Suggest what they should do next in a responsible tone. Avoid giving a definitive diagnosis.
"""

genai.configure(api_key="YOUR_GEMINI_API_KEY")

model = genai.GenerativeModel("gemini-pro")

response = model.generate_content(prompt)
summary = response.text

