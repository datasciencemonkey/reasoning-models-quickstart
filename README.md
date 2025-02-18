# DeepSeek Explorations
**AI Generated ReadME**

This repository contains various experiments and implementations using DeepSeek and Llama models on Databricks, with different deployment approaches including direct model usage, Ollama integration, and model fine-tuning.

## Notebooks Overview

### 1. Basic DeepSeek Model Usage (101-Start.py)
- Direct implementation using llama-cpp-python
- Uses DeepSeek-R1-Distill-Llama-8B model
- Includes basic inference setup and example prompts
- Configurable parameters for context length and GPU layers

### 2. Ollama Integration (102-ollama-r1.py)
- Integration with Ollama for model deployment
- Supports multiple models including:
  - DeepSeek-R1 32B
  - DeepSeek-R1-Distill-Llama-8B
  - Llama3.2-vision
- Includes vision capabilities for image analysis
- Example of multi-modal interactions (text + image)

### 3. Model Fine-tuning (300-Llama3_1_8B_GRPO.py)
- Implementation of GRPO (Generative Reinforcement Policy Optimization)
- Uses Unsloth for optimization
- Features:
  - Data preparation utilities
  - Training configuration
  - Reward functions
  - Model saving and conversion options
- Supports export to various formats (GGUF, float16, etc.)

### 4. Production Deployment (103-pt-deepseek-distill-llama8b.py)
- MLflow integration for model tracking
- Databricks model registry deployment
- Endpoint creation and configuration
- Production-ready serving setup

## Requirements
- Databricks Runtime
- Python packages:
  - llama-cpp-python
  - huggingface_hub
  - ollama
  - unsloth
  - transformers
  - mlflow
  - rich

## Model Variants
The repository works with several model variants:
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1 32B
- Llama 3.1 8B
- Llama 3.2 Vision

## Usage
Each notebook is self-contained and can be run independently on Databricks. Make sure to:
1. Install required dependencies
2. Configure appropriate compute resources
3. Follow the notebook-specific instructions for model loading and usage

## Deployment Options
The repository demonstrates multiple deployment approaches:
1. Direct model usage with llama-cpp
2. Ollama-based deployment
3. MLflow-managed deployment on Databricks
4. GGUF format conversion for llama.cpp compatibility

## Notes
- Some notebooks require specific GPU configurations
- Memory requirements vary based on the model size
- Scale-to-zero is enabled for production deployments
- Various quantization options are available for optimized inference 