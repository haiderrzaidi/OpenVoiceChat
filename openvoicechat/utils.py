import threading
import queue
import librosa
import numpy as np
import os
import logging

import pandas as pd

# Configure basic logging for the module
logger = logging.getLogger(__name__)
# Ensure a handler is added if not already configured by root logger in fastapi_ws.py
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


TIMING = int(os.environ.get('TIMING', 0))


def run_chat(mouth, ear, chatbot, verbose=True, enable_interruptions=True,
             stopping_criteria=lambda x: False):
    """
    Runs a chat session between a user and a bot.

    Parameters:
        mouth (object): An object responsible for the bot's speech output.
        ear (object): An object responsible for listening to the user's input.
        chatbot (object): An object responsible for generating the bot's responses.
        verbose (bool, optional): If True, prints the user's input and the bot's responses. Defaults to True.
        enable_interruptions (bool, optional): If True, allows TTS to be interrupted by user speech. Defaults to True.
        stopping_criteria (function, optional): A function that determines when the chat should stop.
                                                It takes the bot's response as input and returns a boolean.
                                                Defaults to a function that always returns False.

    The function works by continuously listening to the user's input and generating the bot's responses in separate
    threads. If the user interrupts the bot's speech (and interruptions are enabled), the remaining part of the bot's
    response is saved and prepended to the user's next input. The chat stops when the stopping_criteria function
    returns True for a bot's response.
    """
    if TIMING:
        pd.DataFrame(columns=['Model', 'Time Taken']).to_csv('times.csv', index=False)

    pre_interruption_text = ''
    while True:
        user_input = pre_interruption_text + ' ' + ear.listen()

        if verbose:
            print("USER: ", user_input)

        llm_output_queue = queue.Queue()
        interrupt_queue = queue.Queue()
        llm_thread = threading.Thread(target=chatbot.generate_response_stream,
                                      args=(user_input, llm_output_queue, interrupt_queue))
        
        def no_interrupt_listener(duration): # duration is unused but part of the expected signature
            return False

        active_interrupt_listener = ear.interrupt_listen if enable_interruptions else no_interrupt_listener
        
        tts_thread = threading.Thread(target=mouth.say_multiple_stream,
                                      args=(llm_output_queue, active_interrupt_listener, interrupt_queue))

        llm_thread.start()
        tts_thread.start()

        tts_thread.join()
        llm_thread.join()
        if not interrupt_queue.empty():
            pre_interruption_text = interrupt_queue.get()

        res = llm_output_queue.get()
        if stopping_criteria(res):
            break
        if verbose:
            print('BOT: ', res)


class Player_ws:
    def __init__(self, q):
        self.output_queue = q
        self.playing = False
        logger.info("Player_ws initialized.")

    def play(self, audio_array, samplerate):
        logger.info(f"Player_ws: Play called. Input audio array type: {type(audio_array)}, dtype: {audio_array.dtype}, shape: {audio_array.shape}, samplerate: {samplerate}")
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / (1 << 15) # Convert to float32 and normalize
        elif audio_array.dtype != np.float32: # Ensure it's float32 for resampling
            audio_array = audio_array.astype(np.float32)

        resampled_audio = librosa.resample(y=audio_array, orig_sr=samplerate, target_sr=44100)
        logger.info(f"Player_ws: Resampled to 44100 Hz float32. Shape: {resampled_audio.shape}")
        
        audio_bytes = resampled_audio.tobytes()
        logger.info(f"Player_ws: Putting bytes to output queue. Length: {len(audio_bytes)}")
        self.output_queue.put(audio_bytes)

    def stop(self):
        logger.info("Player_ws: Stop called.")
        self.playing = False
        self.output_queue.queue.clear()
        self.output_queue.put('stop'.encode())

    def wait(self):
        time_to_wait = 0
        # while not self.output_queue.empty():
        #     time.sleep(0.1)
        #     peek at the first element
        # time_to_wait = len(self.output_queue.queue[0]) / (44100 * 4)
        # print(time_to_wait)
        # time.sleep(time_to_wait)
        self.playing = False


class Listener_ws:
    def __init__(self, q):
        self.input_queue = q
        self.listening = False
        self.CHUNK = 5945 # This seems to be unused in current read logic, consider removal if not needed elsewhere
        self.RATE = 16_000
        logger.info("Listener_ws initialized.")

    def read(self, x): # x is unused, consider removing if not planned for future use
        data = self.input_queue.get()
        logger.info(f"Listener_ws: Received raw data from queue. Type: {type(data)}, Length: {len(data)}")
        
        data_int16 = np.frombuffer(data, dtype=np.int16)
        logger.info(f"Listener_ws: Converted to int16. Shape: {data_int16.shape}, dtype: {data_int16.dtype}")

        data_float32 = data_int16.astype(np.float32) / (1 << 15)

        # Assuming input sample rate from client is 44100 Hz (common browser default)
        resampled_float32 = librosa.resample(y=data_float32, orig_sr=44100, target_sr=self.RATE)
        logger.info(f"Listener_ws: Resampled to float32. Shape: {resampled_float32.shape}")

        resampled_int16 = (resampled_float32 * (1 << 15)).astype(np.int16)
        
        output_bytes = resampled_int16.tobytes()
        logger.info(f"Listener_ws: Returning processed int16 bytes. Length: {len(output_bytes)}")
        return output_bytes

    def close(self):
        logger.info("Listener_ws: Close called.")
        pass

    def make_stream(self):
        logger.info("Listener_ws: make_stream called, setting listening to True and clearing input queue.")
        self.listening = True
        self.input_queue.queue.clear()
        return self
