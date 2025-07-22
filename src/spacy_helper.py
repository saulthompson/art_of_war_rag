import spacy
from spacy.matcher import PhraseMatcher
import json

class SpacyHelper:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self.find_patterns_in_document()


    def find_patterns_in_document(self):
        entities = []

        with open('assets/entities.json', 'r') as f:
            data = json.load(f)

        seen = set()
        for chunk in data:
            for entity in chunk["entities"]:
                key = (entity["text"].lower(), entity["label"])
                if key not in seen:
                    seen.add(key)
                    entities.append(entity)

        patterns_by_label = {}
        for entity in entities:
            label = entity["label"]
            patterns_by_label.setdefault(label, []).append(self.nlp.make_doc(entity["text"]))

        for label, patterns in patterns_by_label.items():
            self.matcher.add(label, patterns) 

    
    def parse_user_query_for_entities(self, query):
        results = []
        doc = self.nlp(query)
        matches = self.matcher(doc)

        print('matches:', matches)
        filtered_matches = []

        # filter out shorter, less specific matches in case of overlapping matches
        for idx, (match_id, start, end) in enumerate(matches):
            if idx < len(matches) - 1:
                next_match = matches[idx + 1]
                if next_match[1] == start and next_match[2] > end:
                    continue
            
            filtered_matches.append((match_id, start, end))

        for match_id, start, end in filtered_matches:
            label = self.nlp.vocab.strings[match_id] 
            span = doc[start:end]
            print("Matched:", span.text, "Label:", label)
            results.append({"text": span.text, "label": label})
        
        return results

    def extract_entities(self, text):
        """
        Returns a list of detected entities with type, e.g.,
        [{'text': 'Tao river', 'label': 'GPE'}, ...]
        """
        doc = self.nlp(text)

        return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

    def is_date_question(self, question):
      doc = self.nlp(question)
      for token in doc:
          print('dep:', token.lemma_, token.dep_)
          if token.lemma_ in {'when', 'date', 'year', 'century', 'time', 'period', 'dynasty'} and token.dep_ in {"advmod", "npadvmod"}:
              return True
      return False

def is_date_question(query):
    lowered = query.lower()
    date_keywords = ['when', 'year', 'which century', 'date', 'time period']
    return any(kw in lowered for kw in date_keywords)

helper = SpacyHelper()
entities = []

print(helper.parse_user_query_for_entities("what did Zhu Ran do?"))