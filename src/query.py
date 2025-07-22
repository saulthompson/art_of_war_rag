import os
from src.retriever import Retriever
from src.embeddings_generator import Generator
from src.spacy_helper import SpacyHelper
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class QueryMachine:
    def __init__(self, model='gpt-4.1'):
      self.db_search = Retriever()
      self.spacy_helper = SpacyHelper()
      self.embeddings_generator = Generator()
      self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
      self.MODEL = model
      self.prompt_template = """
      You are an expert on Sun-Tzu's The Art of War.

      You will helpfully answer users' questions about the Art of War,
      with close reference to relevant context from a book-length modern commentary on the Art of War by Hua Shan.
      Make sure to explicitly reference at least 2 passages from the context provided to illustrate
      your points. In each reference, you should quote from the 'chunk' field, and include the chapter title, and clearly
      distinguish between Hua Shan's own words and Sun Tzu's original text whenever you cite a quote. You should loosely follow this format:
      "<point>, as pointed out by Hua Shan in the chapter entitled <chapter title> - <quotation from the chunk text>"

      Make sure to always include at least one direct quote from Sun-Tzu.
      Bear in mind the users are not very familiar with Chinese history, culture, and geogrpahy. Add brief explanations
      of people, places, and events. e.g. 'the Fei river - a river that no longer exists, but which is believed to 
      have flowed through modern Anhui province, at the southern limit of the Central China Plain.'

      Do not limit the length of your output - answer as fully as possible

      extracts from the book: {context}

      question: {question}
      """
    
    def get_answer(self, question, context):
      try:
        final_prompt = self.prompt_template.format(context=context, question=question)
        response = self.openai_client.responses.create(
            model=self.MODEL,
            temperature=0.7,
            input=final_prompt,
          )

        return response.output_text
      
      except Exception as e:
        print('Error while getting answer:', e)

    def enter_query(self, website_input=None, history=None):
      try:
        while True:
          if website_input:
            query = website_input
          else:
            query = input('Please enter a question about the Art of War:\n')

          query_entities = self.spacy_helper.extract_entities(query)
          query_embedding = self.embeddings_generator.generate_single_embedding(query)
          retrieved_context = self.db_search.find_similar(query_embedding, limit=6)

          print('retrieved context:', retrieved_context)

          return self.get_answer(query, retrieved_context)

      except Exception as e:
        print(f"error while prompting {self.MODEL}: {e}")

# query_machine = QueryMachine()
# print(query_machine.enter_query())

# 1. parse query with NER
# 2. construct CYPHER query
# 3. query neo4j db
# 4. rerank context?

# todo - clean up database connections in retriever.py / centralize all db logic
# low frequency words
# NER : /city of.../, /River/, /Pass/, / Dynasty/ / Battle .. until one or more capitalized words/