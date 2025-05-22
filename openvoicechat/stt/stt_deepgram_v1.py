import os
import sys
import logging
import json
import asyncio
import websockets
import numpy as np
from dotenv import load_dotenv
from queue import Queue
from dotenv import load_dotenv  
load_dotenv(override=True)  
# Add the parent directory to the Python path to make absolute imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from the package properly
if __name__ == "__main__":
    # When running as a script, import from local files
    from base import BaseEar
else:
    # When imported as a module, use relative imports
    from .base import BaseEar
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
        
        # Initialize Deepgram client
        self.client = DeepgramClient(self.api_key)

    def transcribe_stream(self, audio_queue, transcription_queue):
        async def process_audio():
            try:
                # Set up Deepgram live transcription options
                options = LiveOptions(
                    model="nova-2",
                    encoding="linear16",
                    sample_rate=16000,
                    channels=1,
                    smart_format=True
                )

                # Create a live transcription connection
                connection = self.client.listen.live.v("1")

                def on_message(result):
                    transcript = result.channel.alternatives[0].transcript
                    if transcript:
                        transcription_queue.put(transcript)

                def on_error(error):
                    self.logger.error(f"Deepgram error: {str(error)}")
                    transcription_queue.put(None)

                def on_close(event):
                    self.logger.info("Deepgram connection closed")
                    transcription_queue.put(None)

                # Register event handlers
                connection.on(LiveTranscriptionEvents.Transcript, on_message)
                connection.on(LiveTranscriptionEvents.Error, on_error)
                connection.on(LiveTranscriptionEvents.Close, on_close)

                # Start the connection
                await connection.start(options)

                # Process audio from the queue
                while True:
                    data = audio_queue.get()
                    if data is None:
                        await connection.finish()
                        break
                    # Ensure data is bytes
                    if not isinstance(data, bytes):
                        self.logger.error(f"Invalid data type in audio_queue: {type(data)}")
                        continue
                    await connection.send(data)

            except Exception as e:
                self.logger.error(f"Error in transcription stream: {str(e)}")
                transcription_queue.put(None)

        try:
            asyncio.run(process_audio())
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


    