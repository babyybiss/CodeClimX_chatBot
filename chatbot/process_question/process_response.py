'''
from process_question.llm_prompting import sendPrompt

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
        return sendPrompt(prompt, p_type='p')
    except Exception as e:
        print(f"process_response generate result error : {e}")
'''