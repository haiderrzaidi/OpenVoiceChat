
if __name__ == "__main__":
    from base import BaseEar
else:
    from .base import BaseEar
import os
from dotenv import load_dotenv
import websockets
import json
import asyncio
import logging
import numpy as np


class Ear_deepgram(BaseEar):
    def __init__(self, silence_seconds=1, api_key=None, listener=None):
        super().__init__(silence_seconds, stream=True, listener=listener)
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        
        # Load API key from environment if not provided
        load_dotenv()
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        
        if not self.api_key:
            raise ValueError("Deepgram API key is required. Provide it as an argument or set DEEPGRAM_API_KEY environment variable.")

    def transcribe_stream(self, audio_queue, transcription_queue):
        extra_headers = {"Authorization": f"Token {self.api_key}"}

        async def f():
            try:
                async with websockets.connect(
                    "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&model=nova-2",
                    additional_headers=extra_headers,
                ) as ws:

                    async def sender(ws):
                        try:
                            while True:
                                data = audio_queue.get()
                                if data is None:
                                    await ws.send(json.dumps({"type": "CloseStream"}))
                                    break
                                # Ensure data is bytes
                                if not isinstance(data, bytes):
                                    self.logger.error(f"Invalid data type in audio_queue: {type(data)}")
                                    continue
                                await ws.send(data)
                        except Exception as e:
                            self.logger.error(f"Error while sending: {str(e)}")
                            raise

                    async def receiver(ws):
                        try:
                            async for msg in ws:
                                msg = json.loads(msg)
                                if "channel" not in msg:
                                    transcription_queue.put(None)
                                    break
                                transcript = msg["channel"]["alternatives"][0]["transcript"]
                                if transcript:
                                    transcription_queue.put(transcript)
                        except Exception as e:
                            self.logger.error(f"Error in receiver: {str(e)}")
                            transcription_queue.put(None)
                            raise

                    await asyncio.gather(sender(ws), receiver(ws))
            except Exception as e:
                self.logger.error(f"Connection error: {str(e)}")
                transcription_queue.put(None)

        try:
            asyncio.run(f())
        except Exception as e:
            self.logger.error(f"asyncio error: {str(e)}")
            transcription_queue.put(None)


if __name__ == "__main__":
    import torchaudio
    import torchaudio.functional as F
    import numpy as np

    # To test with your API key:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    ear = Ear_deepgram(api_key=api_key)
    
    # Test with microphone
    text = ear.listen()
    print(text)