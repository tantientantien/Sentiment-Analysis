import torch
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import TextPreprocessor
from model import SentimentBiLSTM
import pandas as pd
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    df['text'] = df['text'].fillna('')
    return df['text'].tolist(), df['sentiment'].tolist()

def prepare_evaluation(texts, labels, preprocessor, nb_weights, word_to_idx, max_len, batch_size=64):
    desired_labels = ['negative', 'neutral', 'positive']
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels):
        if label in desired_labels:
            filtered_texts.append(text)
            filtered_labels.append(label)
    
    # preprocess
    processed_texts = [preprocessor.preprocess(text) for text in filtered_texts]
    
    # encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(filtered_labels)
    
    # check number of classes
    unique_classes = sorted(list(set(encoded_labels)))
    if len(unique_classes) != len(desired_labels):
        raise ValueError(f"The number of classes ({len(unique_classes)}) does not match the desired number of labels ({len(desired_labels)}).")
    
    weights = nb_weights.get_weights(processed_texts)
    
    def encode_sentence(tokens):
        return [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokens]
    
    encoded_texts = [encode_sentence(text) for text in processed_texts]
    padded_texts = [text + [word_to_idx['<PAD>']] * (max_len - len(text)) for text in encoded_texts]
    
    tensor_texts = torch.tensor(padded_texts, dtype=torch.long)
    tensor_weights = torch.tensor(weights, dtype=torch.float)
    tensor_labels = torch.tensor(encoded_labels, dtype=torch.long)
    
    dataset = TensorDataset(tensor_texts, tensor_weights, tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, encoded_labels

def plot_roc(y_true, y_score, n_classes, class_names):
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # draw ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-class Classification')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_model(dataloader, model, true_labels):
    model.eval()
    predictions = []
    all_scores = []
    with torch.no_grad():
        for texts, weights, _ in dataloader:
            outputs = model(texts, weights)
            probs = torch.softmax(outputs, dim=1)
            all_scores.extend(probs.tolist())
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.tolist())
    
    report = classification_report(true_labels, predictions, target_names=['negative', 'neutral', 'positive'], output_dict=True)
    print(classification_report(true_labels, predictions, target_names=['negative', 'neutral', 'positive']))
    
    # draw confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # draw classification report
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.iloc[:-3, :].reset_index().rename(columns={'index': 'class'})
    
    x = np.arange(len(report_df['class']))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(x - width, report_df['precision'], width, label='Precision')
    rects2 = ax.bar(x, report_df['recall'], width, label='Recall')
    rects3 = ax.bar(x + width, report_df['f1-score'], width, label='F1-Score')
    ax.set_ylabel('Scores')
    ax.set_title('Classification Report Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(report_df['class'])
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.show()

    # draw ROC curves
    n_classes = len(np.unique(true_labels))
    y_true = true_labels
    y_score = np.array(all_scores)
    plot_roc(y_true, y_score, n_classes, ['negative', 'neutral', 'positive'])

if __name__ == "__main__":
    filepath = 'dataset/test.csv'
    texts, labels = load_data(filepath)
    
    preprocessor = TextPreprocessor()
    
    try:
        with open('models/nb_weights.pkl', 'rb') as f:
            nb_weights = pickle.load(f)
    except FileNotFoundError:
        print("Naive Bayes weights file not found. Please train the model first by running 'src/train_naive_bayes.py'.")
        exit(1)
    
    model = SentimentBiLSTM(embedding_dim=100, hidden_dim=256, vocab_size=10000, output_dim=3)
    
    try:
        model.load_state_dict(torch.load('models/model.pth', weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load('models/model.pth'))
    
    model.eval()
    
    try:
        with open('models/word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)
        with open('models/config.pkl', 'rb') as f:
            config = pickle.load(f)
        max_len = config['max_len']
    except FileNotFoundError as e:
        print(f"Error loading required files: {e}")
        exit(1)
    
    dataloader, true_labels = prepare_evaluation(
        texts, labels, preprocessor, nb_weights, word_to_idx, max_len, batch_size=64
    )
    
    evaluate_model(dataloader, model, true_labels)