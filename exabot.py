import chainlit as cl
import os
import time
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import gradio as gr
from threading import Thread
from dotenv import load_dotenv

load_dotenv()

MODEL_LIST = ["LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"]
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL = os.environ.get("MODEL_ID")

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    ignore_mismatched_sizes=True)

system_prompt='You are EXAONE model from LG AI Research, a helpful assistant.'
conversation = []

@cl.on_chat_start
def on_chat_start():
    conversation.append({"role": "system", "content": system_prompt})

@cl.on_chat_end
def on_chat_end():
    conversation = []
    print("The user disconnected!")

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    msg = cl.Message(content="")
    conversation.append({"role": "user", "content": message.content})

    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=inputs, 
        max_new_tokens = 4096,
        do_sample = True,
        temperature = 0.1,
        streamer = streamer,
        top_p = 1,
        top_k = 50,
        pad_token_id = 0,
        eos_token_id = 361, # 361
    )
    
    with torch.no_grad():
        thread = Thread( target=model.generate, kwargs=generate_kwargs)
        thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        await msg.stream_token( new_text)
    conversation.append({"role": "assitant", "content": buffer})
    await msg.send()
   
