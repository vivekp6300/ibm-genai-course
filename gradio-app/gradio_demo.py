import gradio as gr

from huggingface_hub import HfFolder

def add_numbers(Num1, Num2):
    return Num1 + Num2

def combine(a, b):
    return a + " " + b

demo2 = gr.Interface(
    fn=combine,
    inputs = [
        gr.Textbox(label="Input 1"),
        gr.Textbox(label="Input 2")
    ],
    outputs = gr.Textbox(label="Output")
)

# Define the interface
demo1 = gr.Interface(
    fn=add_numbers, 
    inputs=[gr.Number(), gr.Number()], # Create two numerical input fields where users can enter numbers
    outputs=gr.Number() # Create numerical output fields
)
# Launch the interface
demo2.launch(server_name="127.0.0.1", server_port= 7860)