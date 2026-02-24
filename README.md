
# ✨ LLM based QA chatbot builder
## Paper Link
https://www.sciencedirect.com/science/article/pii/S235271102400400X

## 🏁 An end-to-end solution to develop a fully open-source application based on open-source models and libraries.

### 🎯 What Is LLM based QA chatbot builder?
There are various stages involved in developing an LLM-based QA chatbot: a) collecting and preprocessing data; b) fine-tuning, testing, and inference of the LLM; and c) developing the chat interface. In this work, we offer the LLM QA builder, a web application that assembles all the processes and simplifies the building of the LLM QA chatbot for both technical and non-technical users, in an effort to speed this development process. Zepyhr, Mistral, Llama-3, Phi, Flan-T5, and a user-provided model for retrieving information relevant to an organization can all be fine-tuned using the system; these LLMs can then be further improved through the application of retrieval-augmented generation (RAG) approaches. We have included an automatic RAG data scraper that is based on web crawling. Furthermore, our system has a human evaluation component to determine the quality of the model. 


## 🎯 Features


| 🦾 Model Support             | Implemented | Description                                   |
|------------------------------|-------------|-----------------------------------------------|
| **Mistral**                  | ✅           | Fine-tuning model powered by Mistral         |
| **Zephyr**                   | ✅           | Fine-tuning model powered by HuggingFace      |
| **Llama-3**                  | ✅           | Fine-tuning model powered by Facebook    |
| **Microsoft Phi-3**          | ✅           | Fine-tuning model powered by Microsoft  |
| **Flan-T5**                  | ✅           | Fine-tuning model powered by Google    |
| **ColBERT**                  | ✅           | Embedding model     |
| **bge-large-en-v1.5**        | ✅           | Embedding model |

## 🎯 Getting started
### 🚀 Installation
```
git clone https://github.com/shahidul034/LLM-based-QA-chatbot-builder
```
```bash
conda create -n llm python=3.10
conda activate llm
pip install torch torchvision torchaudio jupyter langchainhub sentence-transformers faiss-gpu docx2txt langchain bitsandbytes transformers peft accelerate pynvml trl datasets packaging ninja wandb colbert-ai[torch,faiss-gpu] RAGatouille
pip install -U flash-attn --no-build-isolation

```
or 
```
pip install -r requirements.txt
```
## Run
```
cd src
python full_UI.py
```
## 🎯 Contributing

Contributions are always welcome!



## 🎯 License

[MIT](https://github.com/shahidul034/LLM-based-QA-chatbot-builder/blob/main/LICENSE)

