import sounddevice as sd
import boto3
import io
import numpy as np
from pydub import AudioSegment
import os
if __name__ == '__main__':
    from base import BaseMouth
else:
    from .base import BaseMouth
from dotenv import load_dotenv  
load_dotenv(override=True)  

class Mouth_polly(BaseMouth):
    def __init__(self, voice_id='Joanna', engine='neural', language_code='en-US',
                 output_format='mp3', aws_access_key_id=None, aws_secret_access_key=None,
                 region_name='us-east-1', player=sd):
        """
        Initialize Amazon Polly TTS
        
        :param voice_id: Voice to use (e.g., 'Joanna', 'Matthew', 'Amy', etc.)
        :param engine: TTS engine ('standard', 'neural', or 'long-form')
        :param language_code: Language code (e.g., 'en-US', 'en-GB', etc.)
        :param output_format: Audio format ('mp3', 'ogg_vorbis', or 'pcm')
        :param aws_access_key_id: AWS access key (optional, can use environment variables)
        :param aws_secret_access_key: AWS secret key (optional, can use environment variables)
        :param region_name: AWS region
        :param player: Audio player (default: sounddevice)
        """
        self.voice_id = voice_id
        self.engine = engine
        self.language_code = language_code
        self.output_format = output_format
        
        # Initialize boto3 client
        session_kwargs = {'region_name': region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })
        
        self.polly_client = boto3.client('polly', **session_kwargs)
        
        # Set sample rate based on format
        if output_format == 'pcm':
            sample_rate = 16000  # PCM is 16kHz
        else:
            sample_rate = 22050  # MP3/OGG default
            
        super().__init__(sample_rate=sample_rate, player=player)

    def run_tts(self, text):
        """
        Convert text to speech using Amazon Polly
        
        :param text: Text to synthesize
        :return: Audio as numpy array
        """
        try:
            # Prepare the request parameters
            params = {
                'Text': text,
                'VoiceId': self.voice_id,
                'OutputFormat': self.output_format,
                'Engine': self.engine,
                'LanguageCode': self.language_code
            }
            
            # Add sample rate for PCM format
            if self.output_format == 'pcm':
                params['SampleRate'] = str(self.sample_rate)
            
            # Make the request to Polly
            response = self.polly_client.synthesize_speech(**params)
            
            # Get audio data
            audio_data = response['AudioStream'].read()
            
            if self.output_format == 'pcm':
                # PCM format - convert directly to numpy array
                samples = np.frombuffer(audio_data, dtype=np.int16)
                # Convert to float32 for sounddevice
                samples = samples.astype(np.float32) / 32768.0
            else:
                # MP3/OGG format - use pydub to decode
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(audio_data), 
                    format=self.output_format.replace('_', '-')
                )
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                
                # Convert to float32 for sounddevice
                if audio_segment.sample_width == 2:  # 16-bit
                    samples = samples.astype(np.float32) / 32768.0
                elif audio_segment.sample_width == 4:  # 32-bit
                    samples = samples.astype(np.float32) / 2147483648.0
                else:  # 8-bit
                    samples = samples.astype(np.float32) / 128.0
                
                # Update sample rate if different
                self.sample_rate = audio_segment.frame_rate
            
            return samples
            
        except Exception as e:
            print(f"Error in TTS synthesis: {e}")
            # Return silence if there's an error
            return np.zeros(int(self.sample_rate * 0.1))  # 0.1 seconds of silence


if __name__ == '__main__':
    # Example usage with different voices and engines
    

    # Standard voice (lower cost)
    mouth_standard = Mouth_polly(
        voice_id='Matthew', 
        engine='standard',
        output_format='mp3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION_NAME")
    )
    
    text = ("If there's one thing that makes me nervous about the future of self-driving cars, it's that they'll "
            "replace human drivers.\nI think there's a huge opportunity to make human-driven cars safer and more "
            "efficient. There's no reason why we can't combine the benefits of self-driving cars with the ease of use "
            "of human-driven cars.")
    
    print("\nTesting with standard voice (Matthew)...")
    mouth_standard.say_multiple(text, lambda x: False)
    sd.wait()

