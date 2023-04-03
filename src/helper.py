import os
import random
import openai
import numpy as np
from gtts import gTTS
from io import BytesIO
from pathlib import Path
from collections import deque
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time


class FuseBot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        # self.asr = sr.Recognizer()
        self.SAVE_PATH = Path("./src/data") / "fuse-vectorstore.pkl"
        self.docsearch = FAISS.load_local(self.SAVE_PATH, self.embeddings)
        self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',  device='cpu')
        self.past_conversation_list = []
        self.session_state = dict()
        self.session_state["generated"] = []
        self.session_state["queue"] = deque([], maxlen=4)
        self.session_state["question_asked"] = 0
        self.session_state["past"] = []

    def speech_to_text(self,audio):
        try:
            return True, self.asr.recognize_google(audio, language="en")
        except Exception as e:
            return False, e.__class__
    
    def find_embedding(self, text):
        return self.model.encode(text)


    def send_message(self, message_log):
        response = openai.ChatCompletion.create(
                    model = "gpt-4", 
                    messages = message_log,
                    temperature=0.7,
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    timeout=20,
                    frequency_penalty=0,
                    presence_penalty=0,
            )
        
        for choice in response.choices:
            if "text" in choice:
                return choice.text

        return response["choices"][0]["message"]["content"]
 

    def query_response(self,payload, hard_skills):
        docs = self.docsearch.similarity_search(payload["text"])
        context = " ".join([d.page_content for d in docs[:2]])
        
        for i, j in zip(
            payload["past_user_inputs"][-3:], payload["generated_responses"][-3:]
        ):
            self.past_conversation_list.append({"role": "user", "content": i})
            self.past_conversation_list.append({"role": "assistant", "content": j})

        first_message_log = [
            {
                "role": "system",
                "content": f"""
                    You are an Interview Agent for an AI services company Fusemachines. Welcome candidate introducing yourself as the Fuse Interviewing Agent for first only.
                    Candidate name is Sunil appearing in the interview.
                    Ask candidate if he is ready for the interview.
                    Then stop introducing yourself and also don't ask introduction of candidate or candidate's name.
                    Here is the selected skill {hard_skills[0]} and ignore the rest skills, also asked personalized technical questions targeting project experience in the selected skill, one after another in a conversation style, to the candidate in the whole interview.  
                    Make Sure you should ask technical questions one by one or one after another.
                    Do not reply with any extended answer explanations to any of the questions you asked, even if the candidate cannot answer them. Simply move to the next question.
                    Do not write all the questions at once.
                    I want you to only reply as the interviewer. 
                    Ask me the questions and wait for my answers. 
                    Do not write explanations. 
                    Do not ask me technical questions after that and try to end the interview.
                    At the end of the interview, ask me I have any company-related queries. 
                    If I ask such questions, use the following context: {context}. 
                    Do not give answers to company-specific factual questions if you don't find them in the context. 
                    Refrain from answering questions outside of the company's or the interview's context.
                """
            },
            *self.past_conversation_list,
            {"role": "user", 
             "content": payload["text"]}
        ]
          
        hard_skills_message = [
                        {
                            "role": "system",
                            "content": f"""
                            Here is the selected skill {hard_skills[1]} and ignore the rest skills, also asked personalized technical questions targeting project experience in the selected skill, one after another in a conversation style, to the candidate in the whole interview.  
                            Make sure you should ask technical questions one by one or one after another. 
                            Make sure you do not ask all the questions at once.
                            I want you to only reply as the interviewer. 
                            Ask me the questions and wait for my answers. 
                            Do not write explanations. 
                            """
                        },
                        *self.past_conversation_list,
                        {"role": "user", "content": payload["text"]}
                    ]
        
        hr_message = [
                        {
                            "role": "system",
                            "content": f"""
                            Give candidate chance to ask question related to company.
                            If they ask question related to company, use the following context: {context}. 
                            Do not give answers to company-specific factual questions if you don't find them in the context. 
                            Refrain from answering questions outside of the company's or the interview's context.
                            """
                        },
                        *self.past_conversation_list,
                        {"role": "user", "content": payload["text"]}
                    ]
        
        # set a flag to check if the question is asked or not
        first_request = True
        
        while first_request:
            if first_request:
                response = self.send_message(first_message_log)

            for output_message in ''.join(response.lower().split('\n')).splitlines():
                if any(ele in output_message for ele in hard_skills):
                    self.session_state["question_asked"] += 1 
                self.session_state['queue'].append(self.session_state['question_asked'])
                check_queue = (len(self.session_state['queue']) == 4) & (len(set(self.session_state['queue'])) == 1)
                
                if self.session_state["question_asked"] == 4:
                    response = self.send_message(hard_skills_message)
                    
                elif self.session_state["question_asked"] == 8 | check_queue:
                    response = self.send_message(hr_message)
                    
                    # set the flag to false so that the loop breaks
                    first_request = False
            else:
                break      
                    
        # print(past_conversation_list)
        return response
        
    def get_answer_from_bot(self, text, hard_skills):
        try:
            output_text = self.query_response(
                {
                    "past_user_inputs": self.session_state['past'],
                    "generated_responses": self.session_state['generated'],
                    "text": text,
                }, hard_skills
            )
        except:
            output_text = self.query_response(
                {
                    "past_user_inputs": self.session_state['past'],
                    "generated_responses": self.session_state['queue'],
                    "text": text,
                }, hard_skills
            )
        
        self.session_state["past"]+=[text]
        self.session_state["generated"]+=[output_text]

        return output_text
    
    def text_to_speech(self,output_text,i):
        wav = self.tts.tts_to_file(text=output_text,file_path="./src/audio/a_b_"+str(i)+".wav")
        return True
    
    def get_answer(self, text, hard_skills,i):
        text_response = self.get_answer_from_bot(text, hard_skills)
        print("Text Response:", text_response)
        wav = self.text_to_speech(text_response,i)
        return wav
    
    def get_answer_in_text(self,text,hard_skills):
        
        return self.get_answer_from_bot(text, hard_skills)
    
