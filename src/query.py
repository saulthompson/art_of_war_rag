import os
from typing import Generator, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk

from src.vector_retriever import Retriever
from src.embeddings_generator import Generator as EmbeddingsGenerator
from src.spacy_helper import get_spacy_helper
from src.neo4j.scripts.graph_retriever import GraphModel

load_dotenv()


class QueryMachine:
    """
    QueryMachine handles user questions about The Art of War by retrieving relevant
    context from a vector database and a knowledge graph, then generating a response
    using OpenAI's chat completion API.
    """

    def __init__(self, model: str = 'gpt-4.1') -> None:
        """
        Initializes all required components and clients.
        
        Args:
            model (str): OpenAI model to use for completions.
        """
        self.db_search = Retriever()
        self.spacy_helper = get_spacy_helper()
        self.embeddings_generator = EmbeddingsGenerator()
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
        Bear in mind the users are not very familiar with Chinese history, culture, and geography. Add brief explanations
        of people, places, and events. e.g. 'the Fei river - a river that no longer exists, but which is believed to 
        have flowed through modern Anhui province, at the southern limit of the Central China Plain.'

        Do not limit the length of your output - answer as fully as possible

        extracts from the book: {context}

        question: {question}
        """

    def get_answer_stream(self, question: str, context: Union[str, List[dict]]) -> Generator[str, None, None]:
        """
        Streams the response from the OpenAI chat completion API.

        Args:
            question (str): The user question.
            context (Union[str, List[dict]]): Retrieved context relevant to the question.

        Yields:
            str: Partial tokens from the streamed response.
        """
        try:
            final_prompt = self.prompt_template.format(context=context, question=question)

            response_stream: Generator[ChatCompletionChunk, None, None] = self.openai_client.chat.completions.create(
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

    def enter_query(
        self,
        website_input: Optional[str] = None,
        history: Optional[List[dict]] = None
    ) -> Generator[List[dict], None, None]:
        """
        Orchestrates the full query process: embedding generation, context retrieval, and response streaming.

        Args:
            website_input (Optional[str]): User input from a website, or None for CLI input.
            history (Optional[List[dict]]): Previous messages for multi-turn dialogue.

        Yields:
            List[dict]: Chat history updated with streaming assistant content.
        """
        try:
            # Step 1: Get query from user or input
            query = website_input or ""
            if not query:
                while not query:
                    query = input('Please enter a question about the Art of War:\n')

            # Step 2: Retrieve related graph knowledge and vector-based context
            graph_db_chunks = self.graph_db_retriever.run(query)
            query_embedding = self.embeddings_generator.generate_single_embedding(query)
            vector_context = self.db_search.find_similar(query_embedding, limit=6)

            # Combine graph and vector context
            full_context = graph_db_chunks + vector_context if graph_db_chunks else vector_context

            # Step 3: Manage history and stream OpenAI response
            answer_so_far = ""
            history = history or []
            updated_history = history + [{"role": "user", "content": query}]

            for token in self.get_answer_stream(query, full_context):
                answer_so_far += token
                yield updated_history + [{"role": "assistant", "content": answer_so_far}]

        except Exception as e:
            print(f"[Error while prompting {self.MODEL}]: {e}")
