# Databricks notebook source
# MAGIC %pip install ollama
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Install Ollama

# COMMAND ----------

!curl -fsSL https://ollama.com/install.sh | sh

# COMMAND ----------

# !ollama pull deepseek-r1:70b
!ollama pull deepseek-r1:32b


# COMMAND ----------

!ollama create DeepSeek-R1-Distill-Llama-8B-Q8_0-GGUF -f Modelfile

# COMMAND ----------

from ollama import chat

stream = chat(
    model='DeepSeek-R1-Distill-Llama-8B-Q8_0-GGUF',
    messages=[{'role': 'user', 'content': 'What is the difference between eudaimonia and pneumonia?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

# COMMAND ----------

from ollama import chat

stream = chat(
    model='deepseek-r1:32b',
    messages=[{'role': 'user', 'content': 'What is the difference between eudaimonia and pneumonia?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

# COMMAND ----------

!ollama pull llama3.2-vision

# COMMAND ----------

import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'What is the product on this image. Include the original price and the current price if present. Also extract the dimensions, color and other specifications listed?',
        'images': ['/Volumes/main/sgfs/sg-vol/furniture1.png']
    }]
)

vlm_response = response.message.content
print(vlm_response)


# COMMAND ----------

from ollama import chat

stream = chat(
    model='deepseek-r1:32b',
    messages=[{'role': 'user', 'content': f'I am considering the purchase of this sofa. I asked an Image AI to extract the specs of the sofa. It got me this {response.message.content}. Can you help me understand the specs and if this is normally considered a good deal. Whats the discount in percentage that I would manage to score if I got this? Also I have a 150 inches wide, 60 inches deep and 100 inches tall room. Will this sofa fit in that room? I also dont want the space to be clogged up by the sofa. Will that be a problem?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)