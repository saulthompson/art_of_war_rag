import os
from src.vector_retriever import Retriever
from src.embeddings_generator import Generator
from src.spacy_helper import get_spacy_helper
from src.neo4j.scripts.retriever import GraphModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class QueryMachine:
    def __init__(self, model='gpt-4.1'):
      self.db_search = Retriever()
      self.spacy_helper = get_spacy_helper()
      self.embeddings_generator = Generator()
      self.graph_db_retriever = GraphModel()
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
    
    def get_answer_stream(self, question, context):
      try:
        final_prompt = self.prompt_template.format(context=context, question=question)
        response_stream = self.openai_client.chat.completions.create(
            model=self.MODEL,
            temperature=0.7,
            messages=[{"role": "user", "content": final_prompt}],
            stream=True,
        )

        for event in response_stream:
            if event.choices[0].delta.content:
                yield event.choices[0].delta.content
      
      except Exception as e:
        yield f"\n[Error while generating answer: {e}]"

    def enter_query(self, website_input=None, history=None):
      try:
        if website_input:
          query = website_input
        else:
          while True:
            query = input('Please enter a question about the Art of War:\n')
            if query:
              break

        graph_db_chunks = self.graph_db_retriever.run(query)

        query_embedding = self.embeddings_generator.generate_single_embedding(query)
        retrieved_context = self.db_search.find_similar(query_embedding, limit=6)

        if graph_db_chunks:
          retrieved_context = graph_db_chunks + retrieved_context

        answer_so_far = ""
        if history is None:
            history = []

        updated_history = history + [{"role": "user", "content": query}]

        for token in self.get_answer_stream(query, retrieved_context):
            answer_so_far += token
            yield updated_history + [{"role": "assistant", "content": answer_so_far}]

      except Exception as e:
        print(f"error while prompting {self.MODEL}: {e}")

# machine = QueryMachine()
# response = machine.enter_query("What historical figures mentioned in Hua Shan's book best illustrate the clever use of terrain?")

# print(response["answer"], "context_count:", len(response["context"]))

# query_machine = QueryMachine()
# print(query_machine.enter_query())

# Todos
# 1. roadmap of next steps:
#    - handle multiple entities that don't occur in the same chunk (return 5 chunks for each, send to llm)
#    - handle generic words in user queries - event/s, people/person, place, dynasty (return 15 nodes in order of most relations)
#    - programatically create new relationships in graph

# 2. Write new examples dataset, upload to langfuse, update langfuse evaluator
# 3. implement next steps, creating langfuse evaluation for each one
# 
# . rerank context?

# todo - clean up database connections in retriever.py / centralize all db logic
# low frequency words
# NER : /city of.../, /River/, /Pass/, / Dynasty/ / Battle .. until one or more capitalized words/