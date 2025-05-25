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
        logger.info(f"Player_ws: Play method called. Input audio array type: {type(audio_array)}, dtype: {audio_array.dtype}, shape: {audio_array.shape}, samplerate: {samplerate}")
        
        # Ensure audio_array is float32 and normalized
        if audio_array.dtype == np.int16:
            # Correct normalization for int16 to float32 range [-1.0, 1.0)
            audio_array = audio_array.astype(np.float32) / 32768.0 
        elif audio_array.dtype != np.float32:
            logger.warning(f"Player_ws: Unexpected audio_array.dtype: {audio_array.dtype}. Attempting astype(np.float32). Normalization might be incorrect if original range is not standard for this type.")
            audio_array = audio_array.astype(np.float32)
        
        target_sr_frontend = 16000  # Explicitly setting target for frontend

        if samplerate == target_sr_frontend:
            processed_audio = audio_array
            logger.info(f"Player_ws: Input samplerate ({samplerate} Hz) matches target frontend samplerate ({target_sr_frontend} Hz). No resampling needed.")
        else:
            logger.info(f"Player_ws: Input samplerate ({samplerate} Hz) differs from target frontend samplerate ({target_sr_frontend} Hz). Resampling audio from {samplerate} Hz to {target_sr_frontend} Hz...")
            try:
                processed_audio = librosa.resample(y=audio_array, orig_sr=samplerate, target_sr=target_sr_frontend)
                logger.info(f"Player_ws: Resampled to {target_sr_frontend} Hz float32. New shape: {processed_audio.shape}")
            except Exception as e:
                logger.error(f"Player_ws: Error during resampling: {e}. Sending audio with original samplerate {samplerate} Hz instead.", exc_info=True)
                processed_audio = audio_array # Fallback to original audio if resampling fails
                # IMPORTANT: If resampling fails, the Target SR log below will be misleading.
                # Consider how to handle this case better if it occurs.
                # For now, we proceed, but the audio will likely be fast/slow on client.
        
        # Ensure processed_audio is C-contiguous for tobytes() if librosa output isn't guaranteed to be,
        # or if the no-resampling path resulted in a non-contiguous view (less likely for direct assignment).
        if not processed_audio.flags['C_CONTIGUOUS']:
            logger.info("Player_ws: Processed audio is not C-contiguous. Making a contiguous copy.")
            processed_audio = np.ascontiguousarray(processed_audio)

        audio_bytes = processed_audio.tobytes()
        
        # Determine the actual sample rate of the data being sent
        # This is crucial for accurate logging, especially if resampling failed.
        final_samplerate_being_sent = target_sr_frontend
        # A more robust check would be to see if processed_audio is the same as audio_array AND samplerate != target_sr_frontend
        if processed_audio is audio_array and samplerate != target_sr_frontend: # Heuristic: if it's the original and SR mismatch, resampling failed
                final_samplerate_being_sent = samplerate


        logger.info(f"Player_ws: Putting audio to output queue. Actual Sample Rate of Data: {final_samplerate_being_sent} Hz, Data type: {processed_audio.dtype}, Shape: {processed_audio.shape}, Bytes length: {len(audio_bytes)}")
        self.output_queue.put(audio_bytes)

    def stop(self):
        logger.info("Player_ws: Stop called.")
        self.playing = False
        self.output_queue.queue.clear()
        self.output_queue.put('stop'.encode())

    def wait(self):
        logger.info("Player_ws: wait() called, doing nothing for WebSocket player.")
        pass


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
