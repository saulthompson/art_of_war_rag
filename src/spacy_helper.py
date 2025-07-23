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

    
    def filter_subspan_entities(self, entities):
        entities = sorted(entities, key=lambda e: (e['end'] - e['start']), reverse=True)
        filtered = []
        for e in entities:
            if not any(e['start'] >= f['start'] and e['end'] <= f['end'] for f in filtered):
                filtered.append(e)
        return filtered
        
    def parse_user_query_for_entities(self, query):
        results = []
        doc = self.nlp(query)
        matches = self.matcher(doc)
        match_spans = [{"match_id": match_id, "start": start, "end": end} for match_id, start, end in matches]

        print('matches:', matches)
        filtered_spans = self.filter_subspan_entities(match_spans)
        
        for match in filtered_spans:
            span = doc[match["start"]:match["end"]]
            label = self.nlp.vocab.strings[match["match_id"]]
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
