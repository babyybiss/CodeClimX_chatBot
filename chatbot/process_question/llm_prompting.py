import os
import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configure Gemini API key and model
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')


# openAI API backup GPT
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)


def backup_response_openAI(prompt):
    print("================BACKUP RESPONSE GENERATOR GPT-4-TURBO================")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": "You are a machine that only returns the given test of the task and no adittional comments or text. "}, 
                      {"role": "user", "content": prompt}],
        )
        print(f"\nGPT - ATTEMPT #1 RESPONSE : {response}\n")
        
        raw_content = response.choices[0].message.content.strip()
        print(f"\nGPT = ATTEMPT #1 RAW CONTENT : {raw_content}")
        return raw_content
    
    except Exception as e:
        print(f"BACKUP GPT-4-turbo error : {e}")
        return None
    
    
def sendPrompt(prompt):
    
    response = model.generate_content(prompt)
    print("printing resposne: ",response)
    if response is not None and hasattr(response, 'text') and response.candidates[0].content.parts:
        print(f"================GEMINI RESULT : \n {response.text}================")
        return response.text
    else:
        print("================The response did not contain text. Sending to GPT-4-Turbo================")
        return backup_response_openAI(prompt)