import os
import google.generativeai as genai

# Configure the API key and model
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

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
            return None
    except Exception as e:
        print(f"\nprocess_response ERROR : {e}")
        if response is not None and hasattr(response, 'candidate'):
            print(response.candidate.safety_ratings)
        return None
