import torch
import pickle
import sys
from preprocessing import TextPreprocessor
from model import SentimentBiLSTM

def load_components():
    preprocessor = TextPreprocessor()

    try:
        with open('models/nb_weights.pkl', 'rb') as f:
            nb_weights = pickle.load(f)
    except FileNotFoundError:
        print("Naive Bayes weights file not found. Please ensure 'models/nb_weights.pkl' exists.")
        sys.exit(1)

    try:
        with open('models/word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)
        with open('models/config.pkl', 'rb') as f:
            config = pickle.load(f)
        max_len = config['max_len']
    except FileNotFoundError as e:
        print(f"Error loading required files: {e}")
        sys.exit(1)

    model = SentimentBiLSTM(
        embedding_dim=config.get('embedding_dim', 100),
        hidden_dim=config.get('hidden_dim', 128),
        vocab_size=len(word_to_idx),
        output_dim=config.get('output_dim', 3)
    )

    try:
        model.load_state_dict(torch.load('models/model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("Trained model file not found. Please ensure 'models/model.pth' exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    model.eval()

    return preprocessor, nb_weights, word_to_idx, max_len, model

def predict_sentiment(text, preprocessor, nb_weights, word_to_idx, max_len, model):
    # preprocess
    processed_text = preprocessor.preprocess(text)
    
    # get Naive Bayes weights
    weights = nb_weights.get_weights([processed_text])[0]
    
    # encode sentence
    encoded_text = [word_to_idx.get(word, word_to_idx.get('<UNK>', 0)) for word in processed_text]
    

    padded_text = encoded_text + [word_to_idx.get('<PAD>', 0)] * (max_len - len(encoded_text))
    padded_text = padded_text[:max_len]
    
    # convert to tensor
    tensor_text = torch.tensor([padded_text], dtype=torch.long)
    tensor_weights = torch.tensor([weights], dtype=torch.float)
    
    # get prediction
    with torch.no_grad():
        outputs = model(tensor_text, tensor_weights)
        _, preds = torch.max(outputs, 1)
    
    # map prediction to label
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = label_map.get(preds.item(), "Unknown")
    
    return sentiment

def main():
    preprocessor, nb_weights, word_to_idx, max_len, model = load_components()
    
    while True:
        user_input = input("Enter a text to analyze (type 'ex::' to exit): ")
        if user_input.strip().lower() == "ex::":
            print("Exiting the program")
            break
        if not user_input.strip():
            print("Enter a valid text!\n")
            continue
        sentiment = predict_sentiment(user_input, preprocessor, nb_weights, word_to_idx, max_len, model)
        print(f"The sentiment of the text is: {sentiment}\n")

if __name__ == "__main__":
    main()