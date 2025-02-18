# Databricks notebook source
!pip install transformers==4.44.2 mlflow hf_transfer
%restart_python

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
transformers_model = {"model": model, "tokenizer": tokenizer}
task = "llm/v1/chat"

with mlflow.start_run():
   model_info = mlflow.transformers.log_model(
       transformers_model=transformers_model,
       artifact_path="model",
       task=task,
       registered_model_name="main.sgfs.deepseek_r1_distill_8b_llama",
       metadata={
           "pretrained_model_name": "meta-llama/Llama-3.1-8B-Instruct",
           "databricks_model_family": "LlamaForCausalLM",
           "databricks_model_size_parameters": "8b",
        },
    )

# COMMAND ----------

from mlflow.deployments import get_deploy_client


client = get_deploy_client("databricks")
uc_model_name = "deepseek_r1_distill_8b_llama"

endpoint = client.create_endpoint(
    name=uc_model_name,
    config={
        "served_entities": [{
            "entity_name": "main.sgfs.deepseek_r1_distill_8b_llama",
            "entity_version": 1,
             "min_provisioned_throughput": 0,
             "max_provisioned_throughput": 9500,
            "scale_to_zero_enabled": True
        }],
        "traffic_config": {
            "routes": [{
                "served_model_name": f"{uc_model_name}-{1}",
                "traffic_percentage": 100
            }]
        }
    }
)