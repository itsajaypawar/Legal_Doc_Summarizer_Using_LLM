from transformers import T5Tokenizer, T5ForConditionalGeneration


# Load model and tokenizer once
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(document: str, max_len=1000, min_len=50) -> str:

    input_text = "summarize: " + document.strip() 
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        temperature = 0.7,
        
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

