from fastapi import FastAPI
import gradio as gr
from src.query import QueryMachine

app = FastAPI()

query_machine = QueryMachine()

def create_gradio_interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="Chat about the Art of War", type="messages", height=720)
        msg = gr.Textbox(label="Enter your question")
        clear = gr.Button("Clear")

        def reset():
            return [], ""

        msg.submit(query_machine.enter_query, [msg, chatbot], [chatbot], queue=True)
        clear.click(reset, outputs=[chatbot, msg])
    return demo

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="")
