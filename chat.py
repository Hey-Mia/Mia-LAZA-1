import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Loading the learned model and tokenizer
def load_model_and_tokenizer(model_file, tokenizer_file):
    loaded_model = TFT5ForConditionalGeneration.from_pretrained(model_file)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_file)
    return loaded_model, tokenizer

# Pre-processing the question
def preprocess_question(question, tokenizer):
    input_seq = tokenizer.encode(question, padding='longest', truncation=True, return_tensors='tf')
    return input_seq

# Generating a response
def generate_answer(input_seq, model, tokenizer):
    output = model.generate(input_seq, max_length=100, num_beams=5, early_stopping=True)
    predicted_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return predicted_sentence

def main_chat():
    model_file = 'trained_model'
    tokenizer_file = 'tokenizer'

    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_file, tokenizer_file)
    while True:
        question = input("\033[91mAsk a question (or type '!end' to exit): ")
        if question.lower() == '!end':
            break
        input_seq = preprocess_question(question, loaded_tokenizer)
        answer = generate_answer(input_seq, loaded_model, loaded_tokenizer)
        print("\033[92mAI Answer:", answer)

if __name__ == "__main__":
    main_chat()