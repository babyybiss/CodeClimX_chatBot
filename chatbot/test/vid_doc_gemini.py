#from openai import OpenAI
#client = OpenAI()
import os
#from langchain_openai import OpenAI
import json
import random
from nltk.tokenize import sent_tokenize
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini
import google.generativeai as genai
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

import textwrap
from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
#  Load Model and Tokenize
SUM_tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")


directory_path = '/Users/babyybiss/dev/projects/CodeClimX_chatBot/chatbot/data/test.json'
openai_api_key = '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'


tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


# FUCKING SPLITTTTTT PLEASEEEEEEEEEE
def split_texts(text, base_filename):
    filename = base_filename+".txt"
    # Combine the directory path and filename
    full_path = os.path.join(directory_path, base_filename)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Adjust according to your needs
        chunk_overlap=0,  # Adjust if you want overlapping chunks
        length_function=tiktoken_len  # Your defined function for calculating text length
    )
    loader = TextLoader(full_path, encoding="UTF-8")
    data = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=0, length_function=tiktoken_len
    )

    return text_splitter.split_documents(data)



def split_document_attempt2(text, base_filename):
    parts = split_texts(text, base_filename)
    print(f"PARTS : {parts}\n")
    results = []

    for i in parts:
        print(f"SPLIT PART I : {i}")
        prompt = (
        "제공된 텍스트를 요약을 최소화하여, 사람의 말이 아닌 것처럼 문서화된 형태로 한글로 번역하고, 이를 JSON 형식으로 변환하는 작업을 진행해줘. "
        "문서화된 형태는 최소 600자이며, 직접적인 음성 언어의 특성을 제거하고, 더 공식적이고 구조화된 형식의 문서로 재구성 시키고, 요약을 최소화 시켜 'text' 키 필드에 넣어줘. "
        "추가로 최소 50자 이상의 공식적이고 구조화된 요약도 만들어내 'summarization' 키 필드에 넣어줘."
        "절대 미완성 된 json 답변을 보내지 말고 text는 절대 요약 하지마! "
        #f"\n output 예시: \n{example_str}\n"
        f"실제 데이터:\n{i.page_content}\n"
        )

        try:
            # gpt
            model = genai.GenerativeModel('gemini-pro')

            response = model.generate_content(prompt)
            
            raw_content = response.choices[0].message.content.strip()
            parsed_content = json.loads(raw_content)
            
            
            print(f"ATTEMPT #2 PARSED CONTENT : {parsed_content}")
            results.append({'id': f'{base_filename}_part{parts.index(i) + 1}', 'content': parsed_content})
            
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response for part {parts.index(i) + 1}: {str(e)}")
            results.append({'id': f'{base_filename}_part{parts.index(i) + 1}', 'error': 'Invalid JSON format in response'})

    print(f"RESULTS : {results}")
    return results


def generate_documentation(original_text):

    # Construct the prompt for the API
    prompt = (
        "제공된 텍스트를 요약을 최소화하여, 사람의 말이 아닌 것처럼 문서화된 형태로 한글로 번역해줘. "
        "문서화된 형태는 최소 600자이며, 직접적인 음성 언어의 특성을 제거하고, 더 공식적이고 구조화된 형식의 텍스트로 재구성 시켜줘."
        "그리고 꼭 한 문장의 텍스트 형태로 보내줘!!!"
        f"실제 데이터:\n{original_text}\n"
    )

    # Request the OpenAI API to process the text
    try:
        model_g = genai.GenerativeModel('gemini-pro')

        response = model_g.generate_content(prompt)
        extracted_text = response.text
        
        print(f"\nRESPONSE : {extracted_text}\n")


        cleaned_text = extracted_text.replace('\n', ' ')
        
        try:
            input_ids = SUM_tokenizer.encode(cleaned_text, return_tensors="pt")
            print("Encoding Complete - PART 1")
            
            summary_text_ids = model.generate(
                input_ids=input_ids,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                length_penalty=2.0,
                max_length=142,
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
        try:
           # parsed_content = json.loads(raw_content)
            #print(f"ATTEMPT #1 parsed_content : {parsed_content}")
            result = {
                #'id': base_filename,
                'text': cleaned_text,
                'summarization': sum_txt
            }
            print(f"RESULT : {result}")
            return result
        
        except json.JSONDecodeError:
            print("Failed to decode JSON response. Requesting documentation to GPT again.")
            
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
    
    # Load the JSON file
    with open('/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/completed_data/00_harvard_CS50.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # Process each entry in the "data" list
        for item in data['data']:
            original_text = item['text']
            # Use the OpenAI API to generate new text. Replace with your actual API call.
            generated_content = generate_documentation(original_text)
            
            if isinstance(generated_content, dict) and 'text' in generated_content:
                # Replace the old text with the new text
                item['text'] = generated_content['text']
                item['summarization'] = generated_content['summarization']
                
                print(f"Original text: {original_text}\n Replacing new text: {generated_content['text']}\n")
                print(f"\n Result : {item}")
                
            else:
                print(f'Unexpected result from generate_documentation: {generated_content}')

    



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
documentation_object = {'data': all_documentations}

# Specify your output file path
output_file_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/completed_data/gemini_doc.json'

# Save the documentation_object to a JSON file
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(documentation_object, f, ensure_ascii=False, indent=4)

print(f"Documentation saved to {output_file_path}")
