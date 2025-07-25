import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from typing import List, Dict, Tuple, Optional, Union
import json
import os


_spacy_helper_instance: Optional["SpacyHelper"] = None

def get_spacy_helper() -> "SpacyHelper":
    """
    Returns a singleton instance of SpacyHelper.
    """
    global _spacy_helper_instance
    if _spacy_helper_instance is None:
        _spacy_helper_instance = SpacyHelper()
    return _spacy_helper_instance


class SpacyHelper:
    """
    A helper class for using spaCy to extract entities from text,
    match against known entities from a JSON file, and identify generic terms.
    """

    GENERICS = ['event', 'people', 'person', 'who', 'when', 'period', 'place',
                'location', 'battle', 'dynasty', 'historical figure']

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """
        Initializes the spaCy pipeline and loads matchers from entity definitions.
        """
        self.nlp = spacy.load(model)
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self._load_phrase_patterns()

    def _load_phrase_patterns(self, path: str = 'assets/entities.json') -> None:
        """
        Loads entity patterns from a JSON file and adds them to the matcher.

        Args:
            path (str): Path to the JSON file containing entity definitions.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Entity file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        seen: set[Tuple[str, str]] = set()
        entities: List[Dict[str, str]] = []

        for chunk in data:
            for entity in chunk.get("entities", []):
                key = (entity["text"].lower(), entity["label"])
                if key not in seen:
                    seen.add(key)
                    entities.append(entity)

        patterns_by_label: Dict[str, List[Doc]] = {}
        for entity in entities:
            label = entity["label"]
            patterns_by_label.setdefault(label, []).append(self.nlp.make_doc(entity["text"]))

        for label, patterns in patterns_by_label.items():
            self.matcher.add(label, patterns)

    def filter_subspan_entities(self, entities: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """
        Filters out spans that are subspans of others (based on start and end positions).

        Args:
            entities (List[Dict]): List of entity spans with start and end.

        Returns:
            List[Dict]: Filtered list of entity spans.
        """
        entities = sorted(entities, key=lambda e: (e['end'] - e['start']), reverse=True)
        filtered = []
        for e in entities:
            if not any(e['start'] >= f['start'] and e['end'] <= f['end'] for f in filtered):
                filtered.append(e)
        return filtered

    def parse_user_query_for_entities(self, query: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Matches query against known entities and generics.

        Args:
            query (str): User input text.

        Returns:
            Tuple[List[Dict[str, str]], List[str]]: Matched known entities and generic keywords.
        """
        doc = self.nlp(query)
        matches = self.matcher(doc)
        match_spans = [{"match_id": match_id, "start": start, "end": end} for match_id, start, end in matches]
        filtered_spans = self.filter_subspan_entities(match_spans)

        entity_results = []
        for match in filtered_spans:
            span = doc[match["start"]:match["end"]]
            label = self.nlp.vocab.strings[match["match_id"]]
            entity_results.append({"text": span.text, "label": label})

        generic_results = self.parse_user_query_for_generics(query)

        return entity_results, generic_results

    def parse_user_query_for_generics(self, query: str) -> List[str]:
        """
        Identifies generic keywords in a query.

        Args:
            query (str): User input text.

        Returns:
            List[str]: List of matched generic terms.
        """
        lowered = query.lower()
        return [word for word in self.GENERICS if word in lowered]

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts standard spaCy-detected entities from the input text.

        Args:
            text (str): Input text.

        Returns:
            List[Dict[str, str]]: List of entities with labels.
        """
        doc = self.nlp(text)
        return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

    def is_date_question(self, question: str) -> bool:
        """
        Checks whether a question likely relates to a date or time.

        Args:
            question (str): Input question string.

        Returns:
            bool: True if it's a date-related question, False otherwise.
        """
        doc = self.nlp(question)
        for token in doc:
            print('dep:', token.lemma_, token.dep_)
            if token.lemma_ in {'when', 'date', 'year', 'century', 'time', 'period', 'dynasty'} and token.dep_ in {"advmod", "npadvmod"}:
                return True
        return False
