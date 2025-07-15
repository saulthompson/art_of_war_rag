import spacy
import json

class SpacyHelper:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

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

print(helper.is_date_question("what year did I marry?"))