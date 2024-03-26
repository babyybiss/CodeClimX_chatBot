from __future__ import annotations
from fastapi import WebSocket

from .retreive_response import hybrid_search
from process_question.llm_prompting import sendPrompt

async def perform_hybrid_search(text: str, websocket: WebSocket):
    
    print("Hybrid Search Chat System\n")
    
    prompt_t = (
    "Translate the following Korean language question to English\n"
    "Don't answer the question. Only respond with the translated text"
    f"This is the data needed to be translated : {text}"
    )
    
    while True:
        ko_en_text = sendPrompt(prompt_t)
        search_results = hybrid_search(ko_en_text)
        print(f"==============TRANSLATED TEXT : {ko_en_text}==============")
        prompt_p = (
            f"Based on the following document: \n {search_results}"
            "\nHow would you answer the following question: "
            f"\n {text}"
            "\n - You must answer in all text format with no new lines."
            "\n - You must replace any kind of special characters with the actual meaning of that character except for punctuation marks"
            "\n - You must respond in Korean language"
        )
        response_data = "Based on your question, here are some relevant responses:\n"
        for doc_id, score, match_data in search_results:
            response_data += f"- Document ID: {doc_id}\n   Context: {match_data}\n   Relevance Score: {score}\n\n"
            
        print(f"\nresponse? : \n{response_data}\n")
        
        llm_response = sendPrompt(prompt_p)
        
        return llm_response
        #await websocket.send_text(response_data)

        

    