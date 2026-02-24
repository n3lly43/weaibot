import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_huggingface  import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

def load_model(repo_id, 
                hf_token,
                use_hf_endpoint=False,
                billing_account=None):

    if use_hf_endpoint:
        assert billing_account is not None, 'Please specify account for inference endpoints'

        return HuggingFaceEndpoint(
        task='conversational',
        repo_id = repo_id,
        temperature = 0.5,
        huggingfacehub_api_token=hf_token,
        max_new_tokens = 1500,
        server_kwargs={"bill_to":billing_account}
        )

    bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        repo_id,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    query_pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=1500)

    llm = HuggingFacePipeline(pipeline=query_pipeline)

    return llm