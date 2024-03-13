import os

# Gemini
import google.generativeai as genai
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("Who is the ceo of Meta")

print(response)