if __name__ == '__main__':
    from base import BaseChatbot
else:
    from .base import BaseChatbot
import os
from dotenv import load_dotenv
load_dotenv(override=True)
# Qwen/Qwen2.5-7B-Instruct-Turbo
class Chatbot_gpt(BaseChatbot):
    def __init__(self, sys_prompt='',
                 Model='gpt-3.5-turbo',
                 api_key=os.getenv('TOGETHERAI_API_KEY')):
        from openai import OpenAI
        from dotenv import load_dotenv
        if api_key == '':
            load_dotenv()
            api_key = os.getenv('TOGETHERAI_API_KEY')
        self.MODEL = Model
        # self.client = OpenAI(api_key=api_key,base_url="https://api.together.xyz/v1",)
        self.client = OpenAI(api_key=api_key,base_url="https://api.together.xyz/v1",)
        self.messages = []
        self.messages.append({"role": "system", "content": sys_prompt})
        # Warmup call to OpenAI API to reduce initial latency
        try:
            print("armup call to OpenAI API")
            self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct-Turbo",
                messages=[{"role": "system", "content": sys_prompt}],
                max_tokens=1,
                stream=False,
            )
        except Exception as e:
            print(f"OpenAI API warmup failed: {e}")

    def run(self, input_text):
        self.messages.append({"role": "user", "content": input_text})
        stream = self.client.chat.completions.create(
            # model=self.MODEL,
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=self.messages,
            # max
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def post_process(self, response):
        self.messages.append({"role": "assistant", "content": response})
        return response


if __name__ == "__main__":
    preprompt = 'You are a helpful assistant.'
    john = Chatbot_gpt(sys_prompt=preprompt)
    print("type: exit, quit or stop to end the chat")
    print("Chat started:")
    while True:
        user_input = input(" ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            break

        response = john.generate_response(user_input)

        print(response)
