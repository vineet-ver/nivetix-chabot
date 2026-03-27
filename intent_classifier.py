import yaml
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
import os

class ScikitIntentClassifier:
    def __init__(self, nlu_path="data/nlu.yml", model_path="intent_model.pkl"):
        self.nlu_path = nlu_path
        self.model_path = model_path
        self.model = None

    def load_data(self):
        with open(self.nlu_path, 'r', encoding='utf-8') as f:
            nlu_data = yaml.safe_load(f)
            
        texts = []
        labels = []
        
        for item in nlu_data.get('nlu', []):
            intent = item['intent']
            examples = item['examples'].strip().split('\n')
            for ex in examples:
                ex = ex.replace('- ', '').strip()
                if ex:
                    texts.append(ex)
                    labels.append(intent)
                    
        return texts, labels

    def train(self):
        print(f"Reading NLU semantic data from {self.nlu_path}...")
        X, y = self.load_data()
        
        print(f"Executing mathematical training of LinearSVC Model on {len(X)} Hinglish parameters...")
        # Employ a Linear Support Vector Classifier, calibrated for probability mappings
        base_svm = SVC(kernel='linear', class_weight='balanced')
        classifier = CalibratedClassifierCV(base_svm, cv=5)
        
        # Build NLP Pipeline mapping TF-IDF ngram structures into the Classifier
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), classifier)
        self.model.fit(X, y)
        
        joblib.dump(self.model, self.model_path)
        print(f"SUCCESS: Offline Intent Model compiled and written to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def predict(self, text: str):
        if not self.model:
            success = self.load_model()
            if not success:
                return "nlu_fallback", 0.0
            
        probs = self.model.predict_proba([text])[0]
        max_prob_index = probs.argmax()
        confidence = probs[max_prob_index]
        intent = self.model.classes_[max_prob_index]
        
        return intent, float(confidence)

if __name__ == "__main__":
    clf = ScikitIntentClassifier()
    clf.train()
