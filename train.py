import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Načtení a předzpracování dat pro trénink modelu
def preprocess_data(train_file):
    with open(train_file, 'r') as file:
        text = file.read()

    # Tokenizace textu
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    total_words = tokenizer.vocab_size

    # Předzpracování dat
    input_sequences = []
    output_sequences = []
    for line in text.split('\n'):
        if line.startswith('[Q]'):
            input_seq = line[4:].strip()
        elif line.startswith('[A]'):
            output_seq = line[4:].strip()
            input_sequences.append(input_seq)
            output_sequences.append(output_seq)

    # Převod na číselné sekvence
    input_sequences = tokenizer.batch_encode_plus(input_sequences, padding='longest', truncation=True, return_tensors='tf')['input_ids']
    output_sequences = tokenizer.batch_encode_plus(output_sequences, padding='longest', truncation=True, return_tensors='tf')['input_ids']

    # Rozdělení na vstupy a výstupy
    x_train = input_sequences
    y_train = output_sequences

    return x_train, y_train, tokenizer, total_words

# Vytvoření a trénování modelu
def train_model(x_train, y_train, total_words):
    model = TFT5ForConditionalGeneration.from_pretrained('t5-base')

    # Přizpůsobení modelu na specifický úkol
    model.resize_token_embeddings(total_words)

    # Fine-tuning modelu
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer)
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    return model

# Uložení natrénovaného modelu
def save_model(model, tokenizer, model_file, tokenizer_file):
    model.save_pretrained(model_file)
    tokenizer.save_pretrained(tokenizer_file)

def main_train():
    train_file = 'train.txt'
    model_file = 'trained_model'
    tokenizer_file = 'tokenizer'

    x_train, y_train, tokenizer, total_words = preprocess_data(train_file)
    model = train_model(x_train, y_train, total_words)
    save_model(model, tokenizer, model_file, tokenizer_file)

if __name__ == "__main__":
    main_train()
