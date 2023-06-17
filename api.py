
# !pip install -q transformers einops accelerate langchain bitsandbytes

# !nvidia-smi

import os

import torch
import torch.nn as nn
import bitsandbytes as bnb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
max_memory = f"{free_in_GB-2}GB"

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
print("====================",max_memory)

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct
print("model downloading")
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
print("model setup done")
llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

from langchain import PromptTemplate,  LLMChain

template = """
You are an intelligent chatbot. Help the following question with brilliant answers.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Explain what is Artificial Intellience as Nursery Rhymes "

print(llm_chain.run(question))


from flask import Flask, request, jsonify

# ... (copy your code here)

app = Flask(__name__)

print("ip:5000/generate  question")
@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    question = data['question']
    response = llm_chain.run(question)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)