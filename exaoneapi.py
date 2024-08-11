from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from dotenv import load_dotenv
import torch
import os
import json

app = FastAPI()

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

conversation = []

class ExaoneParam(BaseModel):
    system_prompt: str  = "You are EXAONE model from LG AI Research, a helpful assistant."
    user_prompt: str 
    max_new_token: int = 4096
    temperature: float = 0.1
    top_p : int = 1
    top_k : int = 50	

@app.post("/exaone/chat/completion")
async def post_generate( request: ExaoneParam):
    conversation = []
    conversation.append({"role": "system", "content": request.system_prompt})            
    conversation.append({"role": "user", "content": request.user_prompt}) 

    
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=inputs, 
        max_new_tokens = request.max_new_token,
        do_sample = True,
        temperature = request.temperature,
        streamer = streamer,
        top_p = request.top_p,
        top_k = request.top_k,
        pad_token_id = 0,
        eos_token_id = 361, # 361
    )
    
    with torch.no_grad():
        thread = Thread( target=model.generate, kwargs=generate_kwargs)
        thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
    
    conversation.append({"role": "assitant", "content": buffer})
    return json.dumps( conversation, ensure_ascii=False)


if __name__ == "__main__":
    import uvicorn
    unicorn.run( app, host="0.0.0.0", port=5001)
