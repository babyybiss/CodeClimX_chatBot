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

tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


limit = 384

def chunker(context: str):
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
    print("GENERATE DOCUMENT")
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
        print(f"ATTEMPT #1 RESPONSE : {response}\n")
        raw_content = response.choices[0].message.content.strip()
        print(f"ATTEMPT #1 RAW CONTENT : {raw_content}")
        # replace any new lines
        cleaned_text = raw_content.replace('\n', ' ')

        splitted_text = chunker(cleaned_text)
        print(f"\n\nSPLITTED CONTENT : {splitted_text}\n")
        return splitted_text
        #parsed_content = json.loads(raw_content)
        #extracted_txt = parsed_content.get('text', '')
        
        
        '''
        try:
            input_ids = SUM_tokenizer.encode(cleaned_text, return_tensors="pt")
            print("Encoding Complete - PART 1")
            
            summary_text_ids = model.generate(
                input_ids=input_ids,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                length_penalty=2.0,
                max_length=120,
                min_length=56,
                num_beams=4,
            )
            print("Summary Generated - PART 2")

            summarization_txt = SUM_tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
            # Check if the string starts and ends with the specific brackets and remove them
            if summarization_txt.startswith("<s>") and summarization_txt.endswith("</s>"):
                sum_txt = summarization_txt[3:-4]  # Remove the first 3 and last 4 characters
            else:
                sum_txt = summarization_txt
            print("Summary Text:", sum_txt)
            print("Decoding Complete - PART 3")

        except Exception as e:
            print("An error occurred during summarization:", str(e))

        print("PART 3")
        '''
 
            # Call the function to process split parts and retry with GPT
            #return split_document_attempt2(text, base_filename)
        
    except Exception as e:
        print(f'Failed to process response with GPT: {str(e)}')
        return [{'error': f'Processing failed: {str(e)}'}]
    
def process_json_object(json_obj, base_filename):
    # This function will process each JSON object
    # Generate a random ID for the document
    random_number = random.randint(1000, 9999)
    json_obj['id'] = f"{base_filename}_Part{random_number}"
    return json_obj


def process_files():
    with open('/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/raw_data/00_harvard.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        final_result = {'data': []}  # Initialize 'data' as a list

        # Process each entry in the "data" list
        for item in data['data']:
            original_text = item['text']
            # Use the OpenAI API to generate new text. Replace this with your actual API call.
            # splitted_content = chunker(original_text)
            generated_content = generate_documentation(original_text)

            for doc in generated_content:
                object = {
                    'id' : item['id'],
                    'text': doc
                    }
                final_result['data'].append(object)  # Append the new object to the list

    print(f"\n Final result : {final_result}")
            
    return final_result



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
# Create a single object for JSON output
#documentation_object = {'data': all_documentations}

# Specify your output file path
output_file_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/completed_data/00_harvard.json'

# Save the documentation_object to a JSON file
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(all_documentations, f, ensure_ascii=False, indent=4)

print(f"Documentation saved to {output_file_path}")
