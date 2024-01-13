from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=0, eos_token_id=1)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token_id=0, eos_token_id=1)

def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

prompt = "Tell me about"
generated_text = generate_text(prompt)

print("Prompt:", prompt,'\n')
print("Generated Text:", generated_text)