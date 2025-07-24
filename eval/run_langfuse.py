from langfuse import get_client, observe
from src.query import QueryMachine
from datetime import datetime
 
dataset = get_client().get_dataset("art_of_war")

llm = QueryMachine()
observed_query = observe(llm.enter_query)

for item in dataset.items:
    with item.run(
        run_name=f"run #3",
        run_description="added support for generic user questions involving people, places, events, and time periods e.g. 'what events illustrate the importance of terrain'",
        run_metadata={f"version: 1.4", f"model: {llm.MODEL}", "git_hash: d973055ac0c7fe0a0264805316bba43df8aa879a"},
    ) as root_span:
        response = observed_query(item.input)
        if response:
          root_span.update(
            input=item.input,
            output={"answer": response.get("answer", None), "context": response.get("context", None)},
            metadata={"note": "eval run with LLM-as-judge"}
      )

get_client().flush()