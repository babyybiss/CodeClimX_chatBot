from __future__ import annotations
from fastapi import WebSocket

from .retreive_response import hybrid_search
from process_question.process_response import generate_result


async def perform_hybrid_search(text: str, websocket: WebSocket):
    
    print("Hybrid Search Chat System\n")
    
    while True:
        '''
        query = input("Enter your question (or type 'quit' to Exit): ")
        if query.lower() == 'quit':
            break
        '''
        search_results = hybrid_search(text)

        response_data = "Based on your question, here are some relevant responses:\n"
        for doc_id, score, match_data in search_results:
            response_data += f"- Document ID: {doc_id}\n   Context: {match_data}\n   Relevance Score: {score}\n\n"
            
        print(f"\nresponse? : \n{response_data}\n")
        
        llm_response = generate_result(search_results, text)
        return llm_response
        #await websocket.send_text(response_data)

        

    