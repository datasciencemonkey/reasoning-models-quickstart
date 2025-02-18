# Databricks notebook source
# MAGIC %sh
# MAGIC export LD_LIBRARY_PATH=/Workspace/Users/sathish.gangichetty@databricks.com/reasoning-models/build/bin:$LD_LIBRARY_PATH
# MAGIC echo $LD_LIBRARY_PATH
# MAGIC

# COMMAND ----------

# MAGIC %pip install llama-cpp-python
# MAGIC %pip install huggingface_hub
# MAGIC %pip install rich
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
    filename="DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
)

# COMMAND ----------

two_bit_model_path =  hf_hub_download(
    repo_id="unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
    filename="DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
)

# COMMAND ----------

from rich import print
print(model_path)

# COMMAND ----------

from llama_cpp import Llama

# Initialize the model
llm = Llama(
    model_path=model_path, # Maximum context length
     n_ctx=32768,
    n_gpu_layers=61
)

# Example prompt format
prompt = "<｜User｜>What is are carpenter bees? <｜Assistant｜>"

# Generate response
response = llm(
    prompt,
    temperature=0.6,  # Recommended temperature
    top_p=0.95,
    max_tokens= 32000
)


# COMMAND ----------

print(response["choices"][0]['text'])

# COMMAND ----------

