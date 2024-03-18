import sounddevice as sd
import torch
import re
from time import monotonic


def remove_words_in_brackets_and_spaces(text):
    # Pattern to match optional spaces, content inside square brackets, and optional spaces again
    # This will handle spaces before and after the brackets
    pattern = r'\s*\[.*?\]\s*'
    # Remove the matching content and replace it with a single space to avoid multiple spaces
    cleaned_text = re.sub(pattern, ' ', text)
    # Optionally, you might want to trim leading or trailing spaces
    cleaned_text = cleaned_text.strip()
    return cleaned_text


class BaseMouth:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    @torch.no_grad()
    def run_tts(self, text):
        raise NotImplementedError('This method should be implemented by the subclass')

    def say(self, text, listen_interruption_func):
        output = self.run_tts(text)
        # get the duration of audio
        duration = len(output) / self.sample_rate
        sd.play(output, samplerate=self.sample_rate)
        interruption = listen_interruption_func(duration)
        if interruption:
            sd.stop()
            return True
        else:
            sd.wait()
            return False

    def say_multiple(self, text, listen_interruption_func):
        pattern = r'[.?!]'
        sentences = re.split(pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        print(sentences)
        for sentence in sentences:
            interruption = self.say(sentence, listen_interruption_func)
            if interruption:
                break

    def say_multiple_stream(self, text_queue, listen_interruption_func):
        response = ''
        all_response = ''
        while True:
            text = text_queue.get()
            if text is None:
                response = remove_words_in_brackets_and_spaces(response).strip()
                interruption = self.say(response, listen_interruption_func)
                text_queue.put(all_response)
                # print(all_response)
                break
            response += text
            all_response += text
            pattern = r'[.?](?=\s|$)'  # TODO: This should be an attribute
            if bool(re.search(pattern, response)):  # TODO: We should wait for the next char to see if this is the end
                # of the sentence.
                # print(response)
                response = remove_words_in_brackets_and_spaces(response).strip()
                interruption = self.say(response, listen_interruption_func)
                if interruption:
                    text_queue.put(all_response)  # TODO: We have to inform the llm of an interruption
                    break
                response = ''

    def say_timing(self, text, listen_interruption_func):
        start = monotonic()
        output = self.run_tts(text)
        end = monotonic()
        # get the duration of audio
        duration = len(output) / self.sample_rate
        sd.play(output, samplerate=self.sample_rate)
        interruption = listen_interruption_func(duration)
        if interruption:
            sd.stop()
            return True, end - start
        else:
            sd.wait()
            return False, end - start

    def say_multiple_timing(self, text, listen_interruption_func):
        pattern = r'[.?!]'
        sentences = re.split(pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        print(sentences)
        time_taken = None
        for sentence in sentences:
            interruption, tt = self.say_timing(sentence, listen_interruption_func)
            if time_taken is None:
                time_taken = tt
            if interruption:
                break
        return time_taken

    def say_multiple_stream_timing(self, text_queue, listen_interruption_func):
        s = monotonic()
        first_sentence = True
        response = ''
        all_response = ''
        while True:
            text = text_queue.get()
            if text is None:
                print(all_response)
                break
            response += text
            all_response += text
            pattern = r'[.?!]'
            if bool(re.search(pattern, response)):
                response = remove_words_in_brackets_and_spaces(response).strip()
                e = monotonic()
                interruption, time_taken = self.say_timing(response, listen_interruption_func)
                if first_sentence:
                    print("Time to first sentence:", e - s)
                    print("Time taken for tts:", time_taken)
                    first_sentence = False

                if interruption:
                    break
                response = ''
