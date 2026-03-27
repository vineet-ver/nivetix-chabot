import json
import re
from rapidfuzz import process, fuzz
from symspellpy import SymSpell
import pkg_resources
import os

class TextPreprocessor:
    def __init__(self, synonyms_path: str = "data/synonyms.json"):
        self.synonym_dict = {}
        if os.path.exists(synonyms_path):
            try:
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    self.synonym_dict = json.load(f)
            except Exception as e:
                print(f"Failed to load synonyms.json: {e}")
            
        # Initialize SymSpell for ultra-fast typo mitigation
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def clean(self, text: str) -> str:
        # 1. Base clean
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)

        if not text:
            return text

        # 2. SymSpell Correction (Handles severe typos structurally)
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2, transfer_casing=True)
        if suggestions:
            text = suggestions[0].term

        # 3. RapidFuzz Synonyms mapping (Hinglish/Slang standardization)
        words = text.split()
        corrected_words = []
        for word in words:
            best_match = word
            best_score = 0
            
            for root, variations in self.synonym_dict.items():
                if not variations: continue
                # Match against variants using fuzzy logic
                res = process.extractOne(word, variations, scorer=fuzz.ratio)
                if res:
                    match_str, score, _ = res
                    if score > 85 and score > best_score:
                        best_match = root
                        best_score = score
                        
            corrected_words.append(best_match)

        return " ".join(corrected_words)

if __name__ == "__main__":
    # Test block
    tp = TextPreprocessor()
    print("Testing Preprocessor against messy data:")
    print("Original: websit kitne ka banega sir")
    print("Cleaned:", tp.clean("websit kitne ka banega sir"))
