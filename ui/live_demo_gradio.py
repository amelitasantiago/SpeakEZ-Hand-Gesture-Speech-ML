# speakez/src/live_demo.py
import cv2, gradio as gr
from utils import load_config
from inference_ import predict_letter, LetterBuffer
from word_detector import classify_word_video

cfg = load_config()
letters_buffer = LetterBuffer(N=cfg["ui"]["letters_smooth_N"])

def letters_stream(frame):
    # Gradio gives RGB uint8; convert to BGR for OpenCV
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tok, conf = predict_letter(bgr)
    emitted = letters_buffer.push(tok)
    out_text = letters_buffer.buf
    token_str = f"{tok} ({conf:.2f})"
    return out_text, token_str

def clear_text():
    global letters_buffer
    letters_buffer = LetterBuffer(N=cfg["ui"]["letters_smooth_N"])
    return "", "cleared"

def backspace():
    letters_buffer.buf = letters_buffer.buf[:-1]
    return letters_buffer.buf

#def recognize_word(video_path):
    #label, conf, probs = classify_word_video(
        #video_path,
        #thr=cfg["ui"]["prob_threshold_default"],   # json overrides handled inside
        #margin=cfg["ui"]["top2_margin_default"],
        #agreement=cfg["ui"]["agreement_default"]
    #)
    #return f"{label} ({conf:.2f})", probs

# NEW
probs = gr.Dataframe(
    headers=["class", "prob"],
    datatype=["str", "number"],
    col_count=(2, "fixed"),
    row_count=(0, "dynamic"),
    label="Class probabilities"
)

def recognize_word(video_path):
    label, conf, probs_dict = classify_word_video(
        video_path,
        thr=cfg["ui"]["prob_threshold_default"],
        margin=cfg["ui"]["top2_margin_default"],
        agreement=cfg["ui"]["agreement_default"]
    )
    rows = [[k, float(f"{v:.3f}")]
            for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)]
    return f"{label} ({conf:.2f})", rows


with gr.Blocks(title="SpeakEZ — Local Demo") as app:
    gr.Markdown("## SpeakEZ — Live Letters & Word Clips")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Letters (webcam)")
            #cam = gr.Image(source="webcam", streaming=True, mirror_webcam=bool(cfg["ui"]["webcam_mirror"]), type='numpy', label="Webcam")
            cam = gr.Image(sources=["webcam"], streaming=True, mirror_webcam=bool(cfg["ui"]["webcam_mirror"]), type="numpy", label="Webcam")
            out_text = gr.Textbox(label="Text buffer", lines=3)
            last_tok = gr.Textbox(label="Last token", lines=1)
            btn_clear = gr.Button("Clear")
            btn_back  = gr.Button("Backspace")

            cam.stream(letters_stream, outputs=[out_text, last_tok])
            btn_clear.click(clear_text, outputs=[out_text, last_tok])
            btn_back.click(backspace, outputs=[out_text])

        with gr.Column():
            gr.Markdown("### Word Clip")
            vid = gr.Video(label="Upload/record a short clip (~1s)")
            pred = gr.Textbox(label="Prediction")
            #probs = gr.Label(label="Class probabilities")
            btn_run = gr.Button("Recognize")
            btn_run.click(recognize_word, inputs=[vid], outputs=[pred, probs])

if __name__ == "__main__":
    #app.launch(server_name="127.0.0.1", server_port=7860, share=True)
    #app.queue(concurrency_count=2, max_size=16)
    #app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    app.launch(share=True)
