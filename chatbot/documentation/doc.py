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


directory_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/raw_data/before_doc.txt'

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




def generate_documentation(texts):
    print("=====================GENERATE DOCUMENT FUNCTION=====================")
    # Construct the prompt for the API
    
    sys_prompt='''1. A machine that organizes documents into paragraphs with less than 384 characters each, maintaining a timestamp for each paragraph.
2. Respond in English.
3.Ensure the text avoids vague references such as 'it', 'this', or 'etc.', and instead directly states the subject matter."
4. Output format: [{"timestamp": "01:02:12", "paragraph": "part of summary"}]'''

    # Request the OpenAI API to process the text
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": sys_prompt}, 
                {"role": "user", "content": texts,}]
            
        )
        print(f"\nGPT - ATTEMPT #1 RESPONSE : {response}\n")
        
        raw_content = response.choices[0].message.content.strip()
        print(f"\nGPT = ATTEMPT #1 RAW CONTENT : {raw_content}")
    
        
        # replace any new lines
        #cleaned_text = raw_content.replace('\n', ' ')

        #splitted_text = chunker(cleaned_text)
        
        #print(f"\cleaned_text CONTENT : {cleaned_text}\n")
        
        return raw_content
    
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
    with open('/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/raw_data/before_doc.txt', 'r', encoding='utf-8') as file:
        texts = file.read()
        print(texts)
        final_result = {'data': []} 

        
    generated_content = generate_documentation(texts)
            
    return generated_content

# Usage
all_documentations = process_files()

# Specify your output file path
output_file_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/completed_data/after_doc.json'

# Save the documentation_object to a JSON file
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(all_documentations)

print(f"Documentation saved to {output_file_path}")
