import time
import json
import requests
import sseclient
import openai
from langchain import OpenAI
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
class FuseBot:
    def __init__(self):
        self.hard_skills=['pandas','docker','github']
        self.bot_name = "FuseBot"
        self.welcome_message_log =  [
            {
                "role": "system", 
                "content": """ 
                        You are an Interview Agent for an AI services company Fusemachines. Welcome candidate introducing yourself as the Fuse Interviewing Agent for first only.
                        Candidate name is Sunil appearing in the interview. Ask candidate if he is ready for the interview. Then stop introducing yourself and also don't ask introduction of candidate or candidate's name.
                """.strip()
            }
            ]
        self.hard_skills_log = [
            {
                "role": "system",
                "content": f"""
                        Here is the selected skill pandas and ignore the rest skills, also asked personalized technical questions targeting project experience in the selected skill, one after another in a conversation style, to the candidate in the whole interview.  
                        Make Sure you should ask technical questions one by one or one after another.
                        Do not reply with any extended answer explanations to any of the questions you asked, even if the candidate cannot answer them. Simply move to the next question.
                        Do not write all the questions at once.
                        I want you to only reply as the interviewer. 
                        Ask me the questions and wait for my answers. 
                        Do not write explanations.  
                """
            }
        ]
        
        self.soft_skills_log = [
            {
                "role": "system",
                "content": f"""
                        Ask only one question about experience where candidate demonstrated his/her leadership skill. Do not ask technical questions after that and try to end the interview.
                """
            }
        ]
        
        self.past_conversation_list = []
        self.welcome_response = 0
        self.first_hard_skill_question = 0
    
    def print_output(self, text, word_speed=0.3):
        for word in text.split(" "):
            word_len = len(word)
            if word_len == 0:
                continue
            timer_speed = round(word_speed / word_len, 2)
            for letter in word:
                time.sleep(timer_speed)
                print(letter, end="", flush=True)
            print(" ", end="", flush=True)
        print()
        
    def streaming_response(self, message_log):
        responses = ''
        req_url = 'https://api.openai.com/v1/chat/completions'
        req_headers = {
            'Accept': 'text/event-stream',
            'Authorization': 'Bearer ' + openai.api_key,
        }
        req_data = {
            'model': 'gpt-3.5-turbo',
            "messages": message_log,
            "max_tokens": 1024,
            "temperature": 0.1,
            "stop": None,
            "frequency_penalty":0,
            "presence_penalty":0,
            'stream': True,
        }
        request = requests.post(req_url, stream = True, headers=req_headers, json=req_data)
        client = sseclient.SSEClient(request)
    
        for event in client.events():
            if event.data != '[DONE]':
                if 'content' in json.loads(event.data)['choices'][0]['delta']:
                    print(json.loads(event.data)['choices'][0]['delta']['content'], end = "", flush=True)
                    responses += json.loads(event.data)['choices'][0]['delta']['content']
        print()
        return responses
        
    def query_response(self):
        user_input = input("You: ")
        self.welcome_message_log.append({"role": "user", "content": user_input})
        print(f"{self.bot_name}: ", end="", flush=True)
        response = self.streaming_response(self.welcome_message_log)
        self.welcome_message_log.append({"role": "system", "content": response})
        print('*'*100)
        print(self.welcome_message_log)
        print('*'*100)
        
    
    def hard_skill_response(self):
        user_input = input("You: ")
        self.hard_skills_log.append({"role": "user", "content": user_input})
        print(f"{self.bot_name}: ", end="", flush=True)
        response = self.streaming_response(self.hard_skills_log)
        self.hard_skills_log.append({"role": "system", "content": response})
    
    def soft_skill_response(self):
        user_input = input("You: ")
        self.soft_skills_log.append({"role": "user", "content": user_input})
        print(f"{self.bot_name}: ", end="", flush=True)
        response = self.streaming_response(self.soft_skills_log)
        self.soft_skills_log.append({"role": "system", "content": response})
        
    def construct_index(self, directory_path):
        max_input_size = 4096
        num_outputs = 256
        max_chunk_overlap = 20
        chunk_size_limit = 600

        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4", max_tokens=num_outputs))
        documents = SimpleDirectoryReader(directory_path).load_data()
        index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index.save_to_disk('index.json')
        return index

    def hr_message_reponse(self, input_index = 'index.json'):
        index = GPTSimpleVectorIndex.load_from_disk(input_index)        
        query = input('You: ')
        response = index.query(query, response_mode="compact")
        if response.response is not None:
            print(f"{self.bot_name}: ", end="", flush=True)
            self.print_output(response.response)
            
        else:
            print("\nSorry, I couldn't understand your question. Please try again.\n")
    
    def good_bye_response(self):
        while True:
            user_input = input("You: ")
            message_log = [{"role": "user", "content": user_input}]
            response = self.streaming_response(message_log)
            print(f"{self.bot_name}: ", end="", flush=True)
            self.print_output(response)
            if "goodbye" in user_input.lower() or "bye" in user_input.lower():
                break
        
if __name__ == '__main__':
    bot = FuseBot()
    while True:
        bot.send_message()