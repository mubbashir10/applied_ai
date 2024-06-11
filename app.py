# to dpeloy on huggingface space
from fastai.learner import load_learner
from fastai.vision.all import PILImage
import gradio as gr

#load the trained model (exported from the previous chapter)
learn = load_learner('model.pkl')

def predict(img):
    img = PILImage.create(img)
    pred, _, probs = learn.predict(img)
    return f'{str(pred)}: {float(probs.max())}'

interface = gr.Interface(fn=predict, inputs=gr.Image(), outputs=gr.Textbox())

#launch the gradio interface 
interface.launch()