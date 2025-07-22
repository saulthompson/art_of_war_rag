import gradio as gr
from src.query import QueryMachine

query_machine = QueryMachine()

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.ChatInterface(
    fn=query_machine.enter_query,
    type="messages"
)

demo.launch(share=True)
