from openai import AzureOpenAI

from dotenv import load_dotenv
from dotenv import dotenv_values

from azureopenaimanager.prompts import *
from llm_helper.interface_llm_helper import *
import uuid
import logging
import os

header = """
If the answer is not found within the context, 
please mention that the answer is not found.
Please return the response in plain text. 
Convert any markdown to plain text.
"""

SYSTEM_PROMPT = header

class AzureOpenAIManager(ILLMHelper):
    
    def __init__(self,endpoint =os.getenv("AZURE_OPENAI_ENDPOINT"),
                 api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                 deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4O_ID"),
                 api_version = os.getenv("AZURE_OPENAI_API_VERSION_GPT_4O"),
                 cosmosdb_helper = None,
                 token = None):
        
        self.client = AzureOpenAI(
            azure_endpoint = endpoint,
            api_key=api_key,  
            api_version=api_version
        )

        self.deployment_id = deployment_id
        self.cosmosdb_helper = cosmosdb_helper
        self.token = token


    def generate_answer(self,conversation):
        response = self.client.chat.completions.create(
        model=self.deployment_id,
        messages=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop = [' END']
        )
        
        return (response.choices[0].message.content).strip(), \
            response.usage.total_tokens, \
            response.usage.prompt_tokens,response.usage.completion_tokens
    

    def generate_answer_document(self,query):
        messages=[{"role": "assistant", "content": question},
                {"role": "user", "content": query}
                ]
        return self.generate_answer(messages)
    
    def create_prompt(self,context,query):
        header = "If the answer is not found within the context, please mention \
        that the answer is not found \
        Do not answer anything which is not in the context.Please return the response in plain text. Convert any markdown to plain text"
       
        return header + context + "\n\n" + query + "\n"
     
    
    def generate_reply_from_context(self,user_input, content, conversation,
                                    conversation_id = None):
        # prompt = self.create_prompt(content,user_input)

        conversation.append( {"role": "system", "content": SYSTEM_PROMPT})

        query = f'SELECT * FROM c WHERE c.token = "{conversation_id}" \
            ORDER BY c._ts ASC'

        # If conversation_id is not provided, create a new conversation id
        if conversation_id is not None:
            if conversation_id == "":
                conversation_id = str(uuid.uuid4())
                
        logging.info(f"conversation_id: {conversation_id}")

        if self.cosmosdb_helper:
            items = self.cosmosdb_helper.read_items(query)

            if not items:
                # If the conversation_id is not found, 
                # create a new conversation
                # Get the user input and insert into 
                # the conversation HEADER
                item_to_create = {"conversation_id": conversation_id,
                                  "name": user_input,
                                  "short_name": user_input[:10],
                                    "id": str(uuid.uuid4()),
                                  }                
                pass
            
            # If the conversation_id is found,
            # get the conversation and append the user input
            if items:
                for item in items:
                    logging.info(item)
                    item2 = {}
                    item2["role"] = item["role"]
                    item2["content"] = item["content"]
                    conversation.append(item2)
                
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "system", "content": content})       


        reply = self.generate_answer(conversation)

        
        # SAVE THE CONVERSATION INTO COSMOSDB
        # If cosmosdb_helper is provided,
        # insert the user input and the reply into the CosmosDB
        # with the conversation_id

        if self.cosmosdb_helper:

            total_response = user_input + " " + reply[0]

            item_to_create = {"token": conversation_id,
                              "role": "user",
                                "content":total_response,
                                "id": str(uuid.uuid4()),
                              }
            self.cosmosdb_helper.create_item(item_to_create)

            # item_to_create = {"token": conversation_id,
            #                   "role": "assistant",
            #                     "content": reply[0],
            #                     "id": str(uuid.uuid4()),
            #                   }
            # self.cosmosdb_helper.create_item(item_to_create)


        return reply, conversation_id
    
    def get_image_analysis(self,prompt,data):
        response = self.client.chat.completions.create(
        model=self.deployment_id,
       messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{data}"}
            }
        ]}
    ],
    temperature=0.0,
)

        return response.choices[0].message.content
    
    def generate(self,context,query):
        response = (self.generate_reply_from_context(query, context, 
                                         conversation = [],
                                    conversation_id = None))
        logging.info(f"Response: {response[0]}")
        if response:
            if response[0]:
                if response[0][0]: 
                    return response[0][0]
       
        return None