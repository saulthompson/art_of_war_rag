import gradio as gr
from src.query import QueryMachine

query_machine = QueryMachine()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Chat about the Art of War", type="messages")
    msg = gr.Textbox(label="Enter your question")
    clear = gr.Button("Clear")

    def reset():
        return [], ""

    msg.submit(query_machine.enter_query, [msg, chatbot], [chatbot], queue=True)
    clear.click(reset, outputs=[chatbot, msg])

demo.launch(share=True)
