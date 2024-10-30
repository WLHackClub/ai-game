import transformers
import torch

hf_token = ''
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=bnb_config, token=hf_token)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)
generator = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)

chat_history = []

while True:
    next_user_input = input('  >')
    chat_history.append({'role': 'user', 'content': next_user_input})
    next_chat = generator(chat_history)[-1]['generated_text'][-1]
    print(next_chat)
    chat_history.append(next_chat)
