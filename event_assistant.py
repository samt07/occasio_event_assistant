#!/usr/bin/env python
# coding: utf-8

# # Occasio - Event Management Assistant

# In[1]:


# imports

import os
import json
import time
import pprint
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai
import gradio as gr


# In[2]:


# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


# In[3]:


# Connect to OpenAI, Anthropic and Google

openai = OpenAI()
OPENAI_MODEL = "gpt-4o-mini"

claude = anthropic.Anthropic()
ANTHROPIC_MODEL = "claude-3-haiku-20240307"

genai.configure()
GOOGLE_MODEL = "gemini-2.0-flash"


# In[4]:


system_message = (
    'You are called "EventAI," a virtual assistant for an Elementary school called Eagle Elementary School. '
    "You will help users by providing details of upcoming school events such as event name, description, and location. "
    "You MUST always call the appropriate tool function to fetch event details and NEVER generate information yourself. "
    "If you do not have the required details, simply state that you do not have that information. "
    "Responses should be short and courteous, no more than 2 sentences. "
    "When asked to list user questions, summarize based on conversation history."

    "\n\nExample:\n"
    "User: What is read aloud?\n"
    "Assistant: [calls tool function to fetch read aloud event]\n"
)


# In[5]:


# system_message = "You are called \"EventAI\", a virtual assistant for an Elementary school called Eagle Elementary School. You will help users by giving \
# them details of upcoming school events like event name, description, location etc. by calling a tools function"
# #system_message += "Introduce yourself with a warm welcome message on your first response ONLY."
# system_message += "Give short, courteous answers, no more than 2 sentences. "
# system_message += "Do not make up your own event details information"
# system_message += "You might be asked to list the questions asked by the user so far. In that situation, based on the conversation history provided to you, \
# list the questions and respond"


# In[6]:


# Some imports for handling images

import base64
from io import BytesIO
from PIL import Image


# In[7]:


def artist(event_text):
    image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing an {event_text}, showing typical activities that happen for that {event_text}, in a vibrant pop-art style that elementary school kids will like",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


# In[8]:


import base64
from io import BytesIO
from PIL import Image
from IPython.display import Audio, display

def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message)

    audio_stream = BytesIO(response.content)
    output_filename = "output_audio.mp3"
    
    with open(output_filename, "wb") as f:
        f.write(audio_stream.read())
    return output_filename


# In[9]:


school_events = [
    {
        "event_id": "science_fair",
        "name": "Annual Science Fair",
        "description": "Students showcase their science projects and experiments.",
        "date_time": "May 25th, 2025 10 AM",
        "location": "School Auditorium"
    },
    {
        "event_id": "sports_day",
        "name": "Sports Day",
        "description": "A day full of fun and competitive sports activities for students.",
        "date_time": "Jun 10th, 2025 9 AM",
        "location": "School Playground"
    },
    {
        "event_id": "art_exhibition",
        "name": "Art Exhibition",
        "description": "Students display their creative artwork for parents and teachers to admire.",
        "date_time": "Jun 20th, 2025 3 PM",
        "location": "School Art Hall"
    },
    {
        "event_id": "music_festival",
        "name": "Annual Music Festival",
        "description": "A musical event where students perform solo and group performances.",
        "date_time": "Jul 5th, 2025 6 PM",
        "location": "School Auditorium"
    },
    {
        "event_id": "career_day",
        "name": "Career Day",
        "description": "Professionals from various fields share their experiences with students.",
        "date_time": "Aug 12th, 2025 10 AM",
        "location": "Conference Hall"
    },
    {
        "event_id": "math_olympiad",
        "name": "Math Olympiad",
        "description": "A competitive math event to challenge students' problem-solving skills.",
        "date_time": "Sep 1st, 2025 2 PM",
        "location": "Math Lab"
    },
    {
        "event_id": "book_fair",
        "name": "Book Fair",
        "description": "A school-wide book fair to encourage reading and literacy.",
        "date_time": "Sep 20th, 2025 11 AM",
        "location": "Library"
    },
    {
        "event_id": "halloween_parade",
        "name": "Halloween Parade",
        "description": "Students dress up in costumes and parade around the school.",
        "date_time": "Oct 31st, 2025 1 PM",
        "location": "School Courtyard"
    },
    {
        "event_id": "winter_concert",
        "name": "Winter Concert",
        "description": "A special musical performance to celebrate the winter season.",
        "date_time": "Dec 15th, 2025 5 PM",
        "location": "School Auditorium"
    },
    {
        "event_id": "pta",
        "name": "Parent Teachers Meeting (PTA/PTM)",
        "description": "Parent teachers meeting (PTA/PTM) to discuss students' progress.",
        "date_time": "Apr 1st, 2025 11 AM",
        "location" : "Glove Annexure Hall"
    },
    {
        "event_id": "read aloud",
        "name": "Read Aloud to your class/Reading to your class",
        "description": "Kids can bring their favorite book and read it to their class.",
        "date_time": "Apr 15th, 2025 1 PM",
        "location": "Classroom"
    },
     {
        "event_id": "100 days of school",
        "name": "Celebrating 100 days of school. Dress up time for kids",
        "description": "Kids can dress up as old people and celebrate the milestone with their teachers.",
        "date_time": "May 15th, 2025 11 AM",
        "location": "Classroom"
    },
    {
        "event_id": "Book fair",
        "name": "Scholastic book fair",
        "description": "Kids can purchase their favorite scholastic books.",
        "date_time": "Jun 22nd, 2025 10:30 AM",
        "location": "Library"
    },
    {
        "event_id": "Halloween",
        "name": "Halloween",
        "description": "Kids can dress up as their favorite characters",
        "date_time": "Oct 31st, 2025",
        "location": "Classroom"
    },
    {
        "event_id": "Movie Night",
        "name": "Movie Night",
        "description": "A popular and kids centric movie will be played. Kids and families are welcome.",
        "date_time": "May 3rd, 2025",
        "location": "Main auditorium"
    },
    {
        "event_id": "Intruder Drill",
        "name": "Intruder Drill",
        "description": "State mandated monthly intruder drill to prepare staff and students with necessary safety skills in times of a crisis",
        "date_time": "May 3rd, 2025",
        "location": "Main auditorium"
    }
]


# In[10]:


def get_event_details(query):
    search_words = query.lower().split()    
    for event in school_events:
        event_text = event['name'].lower() + ' ' + event['description'].lower()
        if all(word in event_text for word in search_words):
            return event
    return None


# ## Tools
# 
# Tools are an incredibly powerful feature provided by the frontier LLMs.
# 
# With tools, you can write a function, and have the LLM call that function as part of its response.
# 
# Sounds almost spooky.. we're giving it the power to run code on our machine?
# 
# Well, kinda.

# In[11]:


# for claude
tools_claude = [
    {
        "name": "get_event_details",
        "description": "Get the details of a particular upcoming event in Eagle Elementary School. Call this whenever you need to know the event details, for example when a user asks \
'When is the pta meeting scheduled?",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_text": {
                    "type": "string",
                    "description": "The event keyword that the user wants to getails on"
                }
            },
        "required": ["event_text"]
    }
}
]


# In[12]:


# For GPT
events_function_gpt = {
    "name": "get_event_details",
    "description": "Get the details of a particular upcoming event in Eagle Elementary School. Call this whenever you need to know the event details, for example when a user asks \
    'When is the pta meeting scheduled?",
    "parameters": {
        "type": "object",
        "properties": {
            "event_text": {
                "type": "string",
                "description": "The event keyword that the user wants to getails on",
            },
        },
        "required": ["event_text"],
        "additionalProperties": False
    }
}


# In[13]:


# And this is included in a list of tools:
tools_gpt = [{"type": "function", "function": events_function_gpt}]


# In[14]:


#Gemini function declaration structure
gemini_event_details = [{
            "name": "get_event_details",
            "description":"Get the details of a particular upcoming event in Eagle Elementary School. Call this whenever you need to know the event details, for example when a user asks 'When is the pta meeting scheduled?",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_text": {
                        "type": "string",
                        "description": "The event keyword that the user wants to details on",
                    },
                },
                "required": ["event_text"],
            },
        },
        {
            "name": "get_event_test",
            "description":"This is a test function to validate if the function call picks up the right function if there are multiple functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_text": {
                        "type": "string",
                        "description": "The event keyword that the user wants to details on",
                    },
                },
                "required": ["event_text"],
            },
        }
]


# In[15]:


def chat_claude(history):
    #print(f"\nhistory is {history}\n")
    #Claude doesnt take any other key value pair other than role and content. Hence filtering only those key value pairs
    history_claude = list({"role": msg["role"], "content": msg["content"]} for msg in history if "role" in msg and "content" in msg)
    #history is [{'role': 'user', 'metadata': None, 'content': 'when is pta', 'options': None}]
    #messages =  history
    message = claude.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        system=system_message,
        messages=history_claude,
        tools=tools_claude
    )
    image = None
    #print(f"Claude's message is \n {pprint.pprint(message)}\n")
    try:        
        if message.stop_reason == "tool_use":
            tool_use = next(block for block in message.content if block.type == "tool_use")
            event_text = tool_use.input.get('event_text')
            image = artist(event_text)
            tool_result = handle_tool_call(event_text)
            #tool_result = handle_tool_call(tool_use, "Claude")
            
            #print(f"Tool Result: {tool_result}")
            
            response = claude.messages.stream(
                model=ANTHROPIC_MODEL,
                max_tokens=4096,
                system=system_message,
                messages=[
                    {
                        "role": "user", 
                         "content": [
                            {
                                "type": "text",
                                "text": history[-1].get('content')
                            }
                        ]
                    },
                    {
                        "role": "assistant", 
                        "content": message.content
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": tool_result,
                            }
                        ],
                    },
                ],
                tools=tools_claude
            )
            result = ""
            with response as stream:
                for text in stream.text_stream:
                    result += text or ""
                    yield result, None
            #talker(result)
            #image= artist(tool_input.get('event_text'))
            yield result, image
        else:
            response = next((block.text for block in message.content if hasattr(block, "text")), None,)
            chunk_size=30
            for i in range(0, len(response), chunk_size):
                yield response[:i + chunk_size], None
                time.sleep(0.05) #Simulate streaming delay
            #talker(response)
            #image= artist(tool_input.get('event_text'))
            yield response, None
    except Exception as e:
        error_message = "Apologies, my server is acting weird. Please try again later."
        print(e)
        yield error_message, None
    


# In[16]:


def chat_gpt(history):
    #print(f"\nhistory is {history}\n")
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages, tools=tools_gpt)
    image = None
    try:
        if response.choices[0].finish_reason=="tool_calls":
            message = response.choices[0].message
            tool = message.tool_calls[0]
            arguments = json.loads(tool.function.arguments)
            event_text = arguments.get('event_text')
            image = artist(event_text)
            event_json = handle_tool_call(event_text)
            tool_output = {
                "role": "tool",
                "content": event_json,
                "tool_call_id": tool.id
                }
            messages.append(message)
            messages.append(tool_output)
            stream = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True
            )
            result = ""
            for chunk in stream:
                result += chunk.choices[0].delta.content or ""
                yield result, None
            #talker(result)
            yield result, image
        else:        
            reply = response.choices[0].message.content
            chunk_size=30
            for i in range(0, len(reply), chunk_size):
                yield reply[:i + chunk_size], None
                time.sleep(0.05)
            #talker(reply)
            #image= artist("No such event")
            yield reply, None
    except Exception as e:
        error_message = "Apologies, my server is acting weird. Please try again later."
        print(e)
        yield error_message, None


# In[17]:


def chat_gemini(history):
    #print(f"\nhistory is {history}\n")
    history_gemini = [{'role': m['role'], 'parts': [{'text': m['content']}]} if 'content' in m       #if content exists, change it to parts format
                      else {'role': m['role'], 'parts': m['parts']} if 'parts' in m                      #else if parts exists, just copy it as it is
                      else {'role': m['role']} for m in history]        #else neither content nor parts exists, copy only the role ignoring all other keys like metadata, options etc
    
    #print(f"\nhistroy_gemini is {history_gemini}\n")
    model = genai.GenerativeModel(
        model_name=GOOGLE_MODEL,
        system_instruction=system_message
    )
    response = model.generate_content(
        contents = history_gemini,
        #contents = contents,
        tools = [{
            'function_declarations': gemini_event_details,
        }],
    )
    #print(f"response is {response}")

    image = None
    try:
            # Check if the model wants to use a tool
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            event_text = function_call.args.get("event_text")
            image = artist(event_text)
            tool_result = handle_tool_call(event_text)
           
            #print(f"\ntool_result is {tool_result}\n")
            stream = model.generate_content(
                "Based on this information `" + tool_result + "`, extract the details of the event and provide the event details to the user",
                 stream=True               
                )
            #print(f"\nSecond response is {stream}\n")
            result = ""
            for chunk in stream:
                result += chunk.candidates[0].content.parts[0].text or ""
                #print(f"REsult is \n{result}\n")
                yield result, None
            #talker(result)            
            yield result, image
            #print(f"REsult is \n{result}\n")
        else: 
            reply = response.text
            chunk_size=30
            for i in range(0, len(reply), chunk_size):
                yield reply[:i + chunk_size], None
                time.sleep(0.05)
            #talker(reply)
            #image= artist("No such event")
            yield reply, None
        
    except Exception as e:
        error_message = "Apologies, my server is acting weird. Please try again later."
        print(e)
        yield error_message, None      
    


# In[18]:


def call_and_process_model_responses(fn_name, chatbot):#, response, image):
    response = ""
    image = None
    for response, image in fn_name(chatbot):
        if chatbot and chatbot[-1]["role"] == "assistant": 
            chatbot[-1]["content"] = response  # Update the last message
        else:
            chatbot.append({"role": "assistant", "content": response})  # First assistant message
        #print(chatbot)
        yield chatbot, image  # Stream updated history to UI
        


# In[20]:


def handle_tool_call(event_text):
    #print(f"event text is {event_text}")
    event_found = get_event_details(event_text)
    #print(f"event_found is {event_found}")
    
    if event_found:
        response = json.dumps({"name": event_found['name'],"description": event_found['description'], "when": event_found['date_time'], "where": event_found['location']})
    else: 
        response = json.dumps({"event": f"Sorry, there is no schedule currently for {event_text}"})
    return response       
    


# In[21]:


def process_chosen_model(chatbot, model):
    if model == 'GPT':
        for chatbot, image in call_and_process_model_responses(chat_gpt, chatbot):
            yield chatbot, image
    elif model == 'Claude':        
        for chatbot, image in call_and_process_model_responses(chat_claude, chatbot):
            yield chatbot, image
    else:
        #for Gemini, the content is to be replaced with parts.
        for chatbot, image in call_and_process_model_responses(chat_gemini, chatbot):
            yield chatbot, image
        


# In[22]:


# More involved Gradio code as we're not using the preset Chat interface!
# Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.

with gr.Blocks(css="""
    select.gr-box { 
        appearance: auto !important; 
        -webkit-appearance: auto !important; 
    }
""") as ui:
    with gr.Row():
        gr.HTML("<h1 style='text-align: center; color: #4CAF50;'>Occasio! An Event Management Assistant</h1>")  # Added title
    with gr.Row():
        with gr.Column(scale=0):
            model = gr.Dropdown(
                choices=["GPT", "Claude", "Gemini"], 
                label="Select model", 
                value="GPT",
                interactive=True,
                container=True  # Applying the CSS class
            )
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Ask me \"when is pta meeting\", \"how about book fair\" and more... ")
    with gr.Row():
        clear = gr.Button("Clear", min_width=150)
        #message=None

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history
    
    
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        process_chosen_model, inputs=[chatbot, model], outputs=[chatbot, image_output]
    ).then(#added this specifically for Hugging face spaces deployment
        lambda chat: talker(chat[-1]["content"]), inputs=[chatbot], outputs=gr.Audio(autoplay=True, visible=False)
    )
    
    clear.click(lambda: (None, None), inputs=None, outputs=[image_output, chatbot], queue=False)


# In[23]:


ui.launch(inbrowser=True)

input("Press Enter to exit\n")

# In[ ]:




