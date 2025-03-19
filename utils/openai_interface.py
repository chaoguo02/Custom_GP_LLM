import time

from openai import OpenAI
import os

qwen_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# qwen_api = os.environ.get("DASHSCOPE_API_KEY", None)
qwen_api = "sk-a4c8d17b5eba495e8e6cca04804f4320"
qwen_model = "qwen-plus"

deepseek_url = "https://api.deepseek.com/v1"
deepseek_api = os.environ.get("DEEPSEEK_API_KEY", None)
deepseek_model = "deepseek-chat"

class OpenAIInterface:
    def __init__(self):
        self.client = OpenAI(api_key=qwen_api,base_url=qwen_url)
        # self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))
        # self.client = OpenAI(api_key='ollama',
        #                      base_url="http://localhost:11434/v1/")
    def predict_text_logged(self, prompt, temp=1):
        """
        Queries OpenAI's GPT-3 model given the prompt and returns the prediction.
        """
        n_prompt_tokens = 0
        n_completion_tokens = 0
        start_query = time.perf_counter()
        content = "-1"

        message = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="qwen-plus", messages=message, temperature=temp
        )
        print(response.model)
        n_prompt_tokens = response.usage.prompt_tokens
        n_completion_tokens = response.usage.completion_tokens
        # end_query = time.perf_counter()
        print(f"response.choices[0]:{response.choices[0]}")
        content = response.choices[0].message.content
        print(f"content:{content}")
        end_query = time.perf_counter()

        response_time = end_query - start_query
        return {
            "prompt": prompt,
            "content": content,
            "n_prompt_tokens": n_prompt_tokens,
            "n_completion_tokens": n_completion_tokens,
            "response_time": response_time,
        }

    def generate_context(self, prompt, temp=1.5):
        """
        Queries OpenAI's GPT-3 model given the prompt and returns the prediction.
        """
        n_prompt_tokens = 0
        n_completion_tokens = 0
        start_query = time.perf_counter()
        content = "-1"

        message = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="qwen-plus", messages=message, temperature=temp
        )
        # print(response.model)
        n_prompt_tokens = response.usage.prompt_tokens
        n_completion_tokens = response.usage.completion_tokens
        # end_query = time.perf_counter()
        print(f"response.choices[0]:{response.choices[0]}")
        content = response.choices[0].message.content
        print(f"content:{content}")
        end_query = time.perf_counter()

        response_time = end_query - start_query
        content = response.choices[0].message.content.strip()
        return content