from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer only once
MODEL_NAME = "t5-large"  # or path to your fine-tuned model directory (e.g., "./t5-large-legal")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Function to summarize text
def summarize_text(text, max_input_length=512, max_output_length=150):
    input_text = "summarize: " + text.strip().replace("\n", " ")
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    )
    output_ids = model.generate(
        input_ids,
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Function to answer questions
def answer_question(question, context, max_input_length=512, max_output_length=64):
    input_text = f"question: {question.strip()} context: {context.strip()}"
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    )
    output_ids = model.generate(
        input_ids,
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
