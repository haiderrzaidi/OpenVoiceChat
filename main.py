import os

from openvoicechat.tts.tts_polly import Mouth_polly
# from openvoicechat.tts.tts_gtts import Mouth_gtts as Mouth
from openvoicechat.llm.llm_gpt import Chatbot_gpt as Chatbot
from openvoicechat.stt.stt_deepgram import Ear_deepgram as Ear
# from openvoicechat.stt.stt_hf import Ear_hf
from openvoicechat.utils import run_chat
from openvoicechat.llm.prompts import llama_sales
from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    device = 'cpu'

    print('loading models... ', device)
    api_key = os.getenv("DEEPGRAM_API_KEY")
    ear = Ear(silence_seconds=1.5, api_key=api_key)
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
        region_name=os.getenv("REGION_NAME")
    )

    run_chat(mouth, ear, chatbot, verbose=True, enable_interruptions=False,stopping_criteria=lambda x: '[END]' in x)
