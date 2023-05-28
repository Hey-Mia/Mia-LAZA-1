import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Loading and preprocessing data for model training
def preprocess_data(train_file):
    with open(train_file, 'r') as file:
        text = file.read()

    # Tokenizace textu
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    total_words = tokenizer.vocab_size

    # Data preprocessing
    input_sequences = []
    output_sequences = []
    for line in text.split('\n'):
        if line.startswith('[Q]'):
            input_seq = line[4:].strip()
        elif line.startswith('[A]'):
            output_seq = line[4:].strip()
            input_sequences.append(input_seq)
            output_sequences.append(output_seq)

    # Conversion to numeric sequences
    input_sequences = tokenizer.batch_encode_plus(input_sequences, padding='longest', truncation=True, return_tensors='tf')['input_ids']
    output_sequences = tokenizer.batch_encode_plus(output_sequences, padding='longest', truncation=True, return_tensors='tf')['input_ids']

    # Split into inputs and outputs
    x_train = input_sequences
    y_train = output_sequences

    return x_train, y_train, tokenizer, total_words

# Creating and training a model
def train_model(x_train, y_train, total_words):
    model = TFT5ForConditionalGeneration.from_pretrained('t5-base')

    # Adapting the model to a specific task
    model.resize_token_embeddings(total_words)

    # Fine-tuning model
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer)
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    return model

# Saving the trained model
def save_model(model, tokenizer, model_file, tokenizer_file):
    model.save_pretrained(model_file)
    tokenizer.save_pretrained(tokenizer_file)

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

def main():
    train_file = 'train.txt'
    model_file = 'trained_model'
    tokenizer_file = 'tokenizer'

   # Learning the model
    x_train, y_train, tokenizer, total_words = preprocess_data(train_file)
    model = train_model(x_train, y_train, total_words)

    # Storing the model and tokenizer
    save_model(model, tokenizer, model_file, tokenizer_file)

    # Model testing
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_file, tokenizer_file)
    while True:
        question = input("\033[91mAsk a question (or type '!end' to exit): ")
        if question.lower() == '!end':
            break
        input_seq = preprocess_question(question, loaded_tokenizer)
        answer = generate_answer(input_seq, loaded_model, loaded_tokenizer)
        print("\033[92mAI Answer:", answer)

if __name__ == "__main__":
    main()
