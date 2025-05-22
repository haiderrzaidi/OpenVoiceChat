# This is where all the recordings take place

import numpy as np
import pyaudio

# Define default chunk and rate for local microphone
DEFAULT_CHUNK = int(1024 * 2)
DEFAULT_RATE = 16000

FORMAT = pyaudio.paInt16
CHANNELS = 1
# CHUNK and RATE are now DEFAULT_CHUNK and DEFAULT_RATE for local mic


def make_stream():
    p = pyaudio.PyAudio()
    return p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=DEFAULT_RATE, # Use default rate for local mic
                  input=True,
                  frames_per_buffer=DEFAULT_CHUNK) # Use default chunk for local mic


def record_interruption_parallel(vad, listen_queue, streamer=None):
    #listen for interruption untill the queue is not empty
    frames = []
    if streamer is None:
        stream = make_stream()
        chunk_size = DEFAULT_CHUNK
        rate_value = DEFAULT_RATE
    else:
        stream = streamer.make_stream()
        chunk_size = streamer.CHUNK
        rate_value = streamer.RATE

    while True:
        a = listen_queue.get()
        if a is None:
            break
        data = stream.read(chunk_size) # Use local chunk_size
        frames.append(data)
        # Use local rate_value and chunk_size
        contains_speech = vad.contains_speech(frames[int(rate_value / chunk_size) * -2:])
        if contains_speech:
            stream.close()
            frames = np.frombuffer(b''.join(frames), dtype=np.int16)
            frames = frames / (1 << 15)
            return frames.astype(np.float32)
    stream.close()
    return None


def record_interruption(vad, record_seconds=100, streamer=None):
    print("* recording for interruption")
    frames = []
    if streamer is None:
        stream = make_stream()
        # global CHUNK # Removed
        # global RATE # Removed
        chunk_size = DEFAULT_CHUNK
        rate_value = DEFAULT_RATE
    else:
        stream = streamer.make_stream()
        chunk_size = streamer.CHUNK # Use streamer's CHUNK
        rate_value = streamer.RATE # Use streamer's RATE

    # Use local rate_value and chunk_size
    for _ in range(0, int(rate_value / chunk_size * record_seconds)):
        data = stream.read(chunk_size) # Use local chunk_size
        assert len(data) == chunk_size * 2, 'chunk size does not match 2 bytes per sample' # Use local chunk_size
        frames.append(data)
        # Use local rate_value and chunk_size
        contains_speech = vad.contains_speech(frames[int(rate_value / chunk_size) * -2:])
        if contains_speech:
            stream.close()
            frames = np.frombuffer(b''.join(frames), dtype=np.int16)
            frames = frames / (1 << 15)
            return frames.astype(np.float32)
    stream.close()
    return None


def record_user(silence_seconds, vad, streamer=None):
    frames = []

    started = False
    if streamer is None:
        stream = make_stream()
        # global CHUNK # Removed
        # global RATE # Removed
        chunk_size = DEFAULT_CHUNK
        rate_value = DEFAULT_RATE
    else:
        stream = streamer.make_stream()
        chunk_size = streamer.CHUNK # Use streamer's CHUNK
        rate_value = streamer.RATE # Use streamer's RATE
        
    # Use local rate_value and chunk_size
    one_second_iters = int(rate_value / chunk_size)
    print("* recording")

    while True:
        data = stream.read(chunk_size) # Use local chunk_size
        assert len(data) == chunk_size * 2, 'chunk size does not match 2 bytes per sample' # Use local chunk_size
        frames.append(data)
        contains_speech = vad.contains_speech(frames[int(-one_second_iters * silence_seconds):])
        if not started and contains_speech:
            started = True
            print("*listening to speech*")
        if started and contains_speech is False:
            break
    stream.close()

    print("* done recording")

    # creating a np array from buffer
    frames = np.frombuffer(b''.join(frames), dtype=np.int16)

    # normalization see https://discuss.pytorch.org/t/torchaudio-load-normalization-question/71470
    frames = frames / (1 << 15)

    return frames.astype(np.float32)


def record_user_stream(silence_seconds, vad, audio_queue, streamer=None):
    frames = []

    started = False
    if streamer is None:
        stream = make_stream()
        chunk_size = DEFAULT_CHUNK
        rate_value = DEFAULT_RATE
    else:
        stream = streamer.make_stream()
        chunk_size = streamer.CHUNK # Use streamer's CHUNK
        rate_value = streamer.RATE # Use streamer's RATE
        
    # Use local rate_value and chunk_size
    one_second_iters = int(rate_value / chunk_size)
    # stream = make_stream() # This line is no longer needed due to conditional stream creation
    print("* recording")

    print("*listening to speech*")
    while True:
        data = stream.read(chunk_size) # Use local chunk_size
        assert len(data) == chunk_size * 2, 'chunk size does not match 2 bytes per sample' # Use local chunk_size
        frames.append(data)
        audio_queue.put(data)
        contains_speech = vad.contains_speech(frames[int(-one_second_iters * silence_seconds):])
        if not started and contains_speech:
            started = True
        if started and contains_speech is False:
            break
    audio_queue.put(None)
    stream.close()
    print("* done recording")
