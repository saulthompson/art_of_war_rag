from langfuse import get_client, observe
from src.query import QueryMachine
from datetime import datetime
 
dataset = get_client().get_dataset("art_of_war")

llm = QueryMachine()
observed_query = observe(llm.enter_query)

for item in dataset.items:
    with item.run(
        run_name=f"run #2",
        run_description="added graph db support for non-related entities in user query",
        run_metadata={f"version: 1.2", f"model: {llm.MODEL}"},
    ) as root_span:
        response = observed_query(item.input)
        root_span.update(
          input=item.input,
          output={"answer": response["answer"], "context": response["context"]},
          metadata={"note": "eval run with LLM-as-judge"}
      )

get_client().flush()