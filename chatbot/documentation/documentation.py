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

load_dotenv()
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
#  Load Model and Tokenize
SUM_tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")


directory_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/raw_data/00_harvard.json'

openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)


limit = 384

def chunker(context: str):
    print("=====================CHUNKER FUNCTION=====================")
    chunks = []
    all_contexts = context.split('.')
    chunk = []

    for sentence in all_contexts:
        if len('.'.join(chunk + [sentence]).strip()) + 1 > limit:
            if chunk:
                chunks.append('.'.join(chunk).strip() + '.')
            chunk = [sentence]
        else:
            chunk.append(sentence)

    if chunk:
        chunks.append('.'.join(chunk).strip() + '.')

    return chunks




def generate_documentation(splitted_content):
    print("=====================GENERATE DOCUMENT FUNCTION=====================")
    # Construct the prompt for the API
    
    prompt = (
        "This is STT data of a lecture\n"
        "Document it in a form that does not seem like human speech and DO NOT summarize!\n" 
        "The documented form should be at least 1000 characters, removing the characteristics of direct spoken language, and restructure it into a more formal and organized format of text.\n"
        "Ensure the text avoids vague references such as 'it', 'this', or 'etc.', and instead directly states the subject matter."
        "And make sure to send it in the form of one sentence of text!!!\n"
        f"Provided text:\n{splitted_content}\n"
    )

    # Request the OpenAI API to process the text
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": "You are a machine that only returns the given test of the task and no adittional comments or text. "}, 
                      {"role": "user", "content": prompt}],
        )
        print(f"\nGPT - ATTEMPT #1 RESPONSE : {response}\n")
        
        raw_content = response.choices[0].message.content.strip()
        print(f"\nGPT = ATTEMPT #1 RAW CONTENT : {raw_content}")
        
        # replace any new lines
        cleaned_text = raw_content.replace('\n', ' ')

        splitted_text = chunker(cleaned_text)
        
        print(f"\nSPLITTED CONTENT : {splitted_text}\n")
        
        return splitted_text
    
    except Exception as e:
        print(f'Failed to process response with GPT: {str(e)}')
        return [{'error': f'Processing failed: {str(e)}'}]
    
def generate_id(item):
    print("=====================GENERATE ID FUNCTION=====================")
    # Generate a random ID for the document
    random_number = random.randint(1000, 9999)
    
    item_id = item['id']
    
    id = f"{item_id}_Part{random_number}"
    return id


def process_files():
    print("=====================PROCESS FILES FUNCTION=====================")
    with open('/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/raw_data/00_harvard.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        final_result = {'data': []} 

        # Process each entry in the "data" list
        for item in data['data']:
            original_text = item['text']

            generated_content = generate_documentation(original_text)

            unique_id = generate_id(item)
            
            for doc in generated_content:
                object = {
                    'id' : unique_id,
                    'text': doc
                    }
                final_result['data'].append(object)  # Append the new object to the list

    print(f"\n Final result : {final_result}")
            
    return final_result

# Usage
all_documentations = process_files()

# Specify your output file path
output_file_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/completed_data/00_harvard.json'

# Save the documentation_object to a JSON file
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(all_documentations, f, ensure_ascii=False, indent=4)

print(f"Documentation saved to {output_file_path}")
