from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from process_question.handle_question import perform_hybrid_search

app = FastAPI()

# Add a middleware to allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.websocket("/chatbot")
async def websocket_endpoint(websocket: WebSocket):
    print("Hello world!")
    await websocket.accept()

    while True:
        try:
            # Receive text from the client
            text = await websocket.receive_text()
            print(f"Received text: {text}")

            # Perform the search and generate response
            # Assuming perform_hybrid_search returns a string response
            search_results = await perform_hybrid_search(text, websocket)

            # Make sure that search_results is a string before sending
            if isinstance(search_results, str):
                # Send the response back to the client
                await websocket.send_text(search_results)
            else:
                # If search_results is not a string, convert it or handle the error
                print("search_results is not a string:", search_results)
                # Example conversion, might need to be adjusted based on actual type
                await websocket.send_text(str(search_results))

        except Exception as e:
            # Handle exceptions such as connection closed by the client
            print(f"An error occurred: {e}")
            break
        
@app.get("/")
def helloWorld():
    print("Hello world!")