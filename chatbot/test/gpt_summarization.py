from openai import OpenAI
import os
#from langchain_openai import OpenAI
import json
import random
from nltk.tokenize import sent_tokenize
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from dotenv import load_dotenv


from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
#  Load Model and Tokenize
SUM_tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")


directory_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/raw_data/00_harvard.json'

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


def generate_documentation(original_text, openai_api_key):

    # Construct the prompt for the API
    prompt = (
        "제공된 텍스트를 요약을 최소화하여, 사람의 말이 아닌 것처럼 문서화된 형태로 한글로 번역해줘. "
        "문서화된 형태는 최소 600자이며, 직접적인 음성 언어의 특성을 제거하고, 더 공식적이고 구조화된 형식의 텍스트로 재구성 시켜줘."
        "그리고 꼭 한 문장의 텍스트 형태로 보내줘!!!"
        f"실제 데이터:\n{original_text}\n"
    )

    # Request the OpenAI API to process the text
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": "You are a machine that only returns the given test of the task and no adittional comments or text. "}, 
                      {"role": "user", "content": prompt}],
        )
        print(f"ATTEMPT #1 RESPONSE : {response}\n")
        raw_content = response.choices[0].message.content.strip()
        print(f"ATTEMPT #1 RAW CONTENT : {raw_content}")
        # replace any new lines
        cleaned_text = raw_content.replace('\n', ' ')


        try:
            prompt1 = (
                    "해당 텍스트에 대해 최소 50자의 요약본 보내줘 "
                    "꼭 한 문장의 텍스트 형태로 보내줘!!!"
                    f"실제 데이터:\n{original_text}\n"
            )

            response1 = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": "You are a machine that only returns the given test of the task and no adittional comments or text. "}, 
                      {"role": "user", "content": prompt1}],
            )
            print(f"SUMMARIZATION RESPONSE : {response1}\n")
            raw_content1 = response.choices[0].message.content.strip()
            print(f"SUMMARIZATION RAW CONTENT : {raw_content1}")
            
            # replace any new lines
            cleaned_text1 = raw_content1.replace('\n', ' ')
        except Exception as e:
            print("An error occurred during summarization:", str(e))

        print("PART 3")
        try:
            result = {
                #'id': base_filename,
                'text': cleaned_text,
                'summarization': cleaned_text1
            }
            print(f"RESULT : {result}")
            return result
        
        except json.JSONDecodeError:
            print("Failed to decode JSON response. Requesting documentation to GPT again.")
        
    except Exception as e:
        print(f'Failed to process response with GPT: {str(e)}')
        return [{'error': f'Processing failed: {str(e)}'}]
    
def process_files():
    
    # Load the JSON file
    with open('/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/raw_data/00_harvard.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # Process each entry in the "data" list
        for item in data['data']:
            original_text = item['text']
            # Use the OpenAI API to generate new text. Replace with your actual API call.
            generated_content = generate_documentation(original_text, openai_api_key)
            
            if isinstance(generated_content, dict) and 'text' in generated_content:
                # Replace the old text with the new text
                item['text'] = generated_content['text']
                item['summarization'] = generated_content['summarization']
                
                print(f"Original text: {original_text}\n Replacing new text: {generated_content['text']}\n")
                print(f"\n Result : {item}")
                
            else:
                print(f'Unexpected result from generate_documentation: {generated_content}')

    return data



def parse_stringified_json(data):
    for key, value in data.items():
        parts = value["parts"]
        for i, part in enumerate(parts):
            try:
                # Attempt to parse the stringified JSON part
                parts[i] = json.loads(part)
            except json.JSONDecodeError:
                # Part is not a stringified JSON, leave it as is
                continue



# Usage
all_documentations = process_files()

# Specify your output file path
output_file_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/completed_data/gpt_summarization.json'

# Save the documentation_object to a JSON file
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(all_documentations, f, ensure_ascii=False, indent=4)

print(f"Documentation saved to {output_file_path}")
