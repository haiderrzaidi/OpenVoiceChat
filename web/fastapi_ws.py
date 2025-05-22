import uvicorn
from openvoicechat.tts.tts_polly import Mouth_polly
# from openvoicechat.tts.tts_gtts import Mouth_gtts as Mouth
from openvoicechat.llm.llm_gpt import Chatbot_gpt as Chatbot
from openvoicechat.stt.stt_deepgram import Ear_deepgram as Ear
from openvoicechat.llm.prompts import llama_sales
from openvoicechat.utils import run_chat
from dotenv import load_dotenv
import threading
import queue
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openvoicechat.utils import Listener_ws, Player_ws
import torch
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="web/static"), name="static")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted from: {websocket.client}")

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    listener = Listener_ws(input_queue)
    player = Player_ws(output_queue)

    api_key = os.getenv("DEEPGRAM_API_KEY")
    ear = Ear(silence_seconds=1.0, api_key=api_key,listener=listener)
    # ear = Ear_hf(
    #     model_id="openai/whisper-tiny.en",
    #     silence_seconds=1.5,
    #     device=device,

    # )
    load_dotenv()

    chatbot = Chatbot(sys_prompt=llama_sales, api_key=os.getenv("TOGETHERAI_API_KEY"))

    mouth = Mouth_polly(
        voice_id='Matthew', 
        engine='standard',
        output_format='mp3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION_NAME"),
        player=player
    )
    # Update the call to run_chat to include enable_interruptions=True
    # Assuming the order in run_chat is: verbose, enable_interruptions
    threading.Thread(target=run_chat, args=(mouth, ear, chatbot, True, False)).start()

    try:
        while True:
            data = await websocket.receive_bytes()
            logger.info(f"Received bytes from client {websocket.client}: {len(data)}")
            if listener.listening:
                input_queue.put(data)
            
            if not output_queue.empty():
                response_data = output_queue.get_nowait()
                logger.info(f"Sending bytes to client {websocket.client}: {len(response_data)}")
            else:
                response_data = 'none'.encode()
                logger.info(f"Sending 'none' to client {websocket.client}")
            await websocket.send_bytes(response_data)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client: {websocket.client}")
        del mouth
        del ear
        #clear cuda cache
        import torch
        torch.cuda.empty_cache()
    finally:
        logger.info(f"WebSocket connection closed for client: {websocket.client}")
        await websocket.close()


@app.get("/")
def read_root():
    return FileResponse('web/static/stream_audio.html')


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000,
                ssl_keyfile="web/localhost+2-key.pem", ssl_certfile="web/localhost+2.pem")
