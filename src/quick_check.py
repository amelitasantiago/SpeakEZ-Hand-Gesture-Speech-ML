import gradio as gr
def echo_path(p): return f"type={type(p).__name__}\npath={p}"
with gr.Blocks() as demo:
    f = gr.File(type="filepath")
    o = gr.Textbox()
    f.change(echo_path, f, o)
demo.launch(share=True)
