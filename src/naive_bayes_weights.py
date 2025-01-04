from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayesWeights:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = None
    
    def train_weights(self, texts, labels):
        X = self.vectorizer.fit_transform([' '.join(text) for text in texts])
        self.model = MultinomialNB()
        self.model.fit(X, labels)
        
    def get_weights(self, texts):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        X = self.vectorizer.transform([' '.join(text) for text in texts])
        return self.model.predict_proba(X) 