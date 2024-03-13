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
    print("================BACKUP RESPONSE GENERATOR================")
    
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
    
    
def generate_result(search_result, text):
    prompt = (
        f"Based on the following document: \n {search_result}"
        "\nHow would you answer the following question: "
        f"\n {text}"
        "\n - You must answer in all text format with no new lines."
        "\n - You must replace any kind of special characters with the actual meaning of that character except for punctuation marks"
        "\n - You must respond in Korean language"
    )

    try:
        response = model.generate_content(prompt)
        if response is not None and hasattr(response, 'text'):
            print(f"GEMINI RESULT : \n {response.text}")
            return response.text
        else:
            print("The response did not contain text.")
            return backup_response_openAI(prompt)
        
    except Exception as e:
        print(f"\nprocess_response ERROR : {e}")
        if response is not None and hasattr(response, 'candidate'):
            print(response.candidate.safety_ratings)
        return None
