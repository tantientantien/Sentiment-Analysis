import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import TextPreprocessor
from naive_bayes_weights import NaiveBayesWeights
from model import SentimentBiLSTM
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    texts = df['text'].astype(str).tolist()
    selected_texts = df['selected_text'].astype(str).tolist()
    return texts, selected_texts, df['sentiment'].tolist()

def prepare_dataloaders(texts, selected_texts, labels, preprocessor, nb_weights, vocab_size, batch_size=64):
    processed_texts = [preprocessor.preprocess(text) for text in texts]
    processed_selected_texts = [preprocessor.preprocess(text) for text in selected_texts]
    encoded_labels = LabelEncoder().fit_transform(labels)
    
    nb_weights.train_weights(processed_texts, encoded_labels)
    weights = nb_weights.get_weights(processed_texts)
    
    all_tokens = [token for sublist in processed_texts for token in sublist]
    token_freq = Counter(all_tokens)
    

    most_common = token_freq.most_common(vocab_size - 2)
    word_to_idx = {word: idx+2 for idx, (word, _) in enumerate(most_common)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    
    def encode_sentence(tokens):
        return [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokens]
    
    encoded_texts = [encode_sentence(text) for text in processed_texts]
    max_len = max(len(text) for text in encoded_texts)
    padded_texts = [text + [word_to_idx['<PAD>']] * (max_len - len(text)) for text in encoded_texts]
    
    tensor_texts = torch.tensor(padded_texts, dtype=torch.long)
    tensor_weights = torch.tensor(weights, dtype=torch.float)
    tensor_labels = torch.tensor(encoded_labels, dtype=torch.long)
    
    dataset = TensorDataset(tensor_texts, tensor_weights, tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, word_to_idx, max_len, processed_selected_texts

def train_model(dataloader, selected_texts, model, epochs=10, learning_rate=0.001, weight_decay=1e-5, save_path='models/model.pth'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # use Adam optimizer with weight_decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True) # reduce learning rate when loss doesn't decrease
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    

    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for i, (texts, weights, labels) in enumerate(dataloader):
            selected_text = selected_texts[i * dataloader.batch_size:(i + 1) * dataloader.batch_size]
            optimizer.zero_grad()
            outputs = model(texts, weights, selected_text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        average_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        train_losses.append(average_loss)
        train_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
        scheduler.step(average_loss) # reduce learning rate when loss doesn't decrease

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {os.path.abspath(save_path)}')
    
    # draw graph to show loss and accuracy
    plot_training_metrics(train_losses, train_accuracies, epochs)

def plot_training_metrics(losses, accuracies, epochs):
    epochs_range = range(1, epochs + 1)
    
    # draw loss graph
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, losses, marker='o', color='b', label='Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True)
    
    # draw accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracies, marker='o', color='g', label='Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_metrics.png')
    plt.show()

if __name__ == "__main__":
    filepath = 'dataset/train.csv'
    texts, selected_texts, labels = load_data(filepath)
    
    preprocessor = TextPreprocessor()
    nb_weights = NaiveBayesWeights()
    
    batch_size = 128
    vocab_size = 10000
    
    dataloader, word_to_idx, max_len, processed_selected_texts = prepare_dataloaders(texts, selected_texts, labels, preprocessor, nb_weights, vocab_size, batch_size)
    
    # initialize model
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 3  # positive, negative, neutral
    model = SentimentBiLSTM(embedding_dim, hidden_dim, vocab_size, output_dim)
    
    train_model(dataloader, processed_selected_texts, model, epochs=7, learning_rate=0.001, save_path='models/model.pth')
    
    config = {
        'max_len': max_len,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'vocab_size': vocab_size,
        'output_dim': output_dim
    }
    
    # after fit vectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(texts)
    
    # save vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # save config
    with open('models/config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    # save word_to_idx
    with open('models/word_to_idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f)
    
    # save nb_weights
    with open('models/nb_weights.pkl', 'wb') as f:
        pickle.dump(nb_weights, f)
