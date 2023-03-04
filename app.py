import matplotlib.pyplot as plt
models = ['Logistic Regression','Decision Tree','SVM','Random Forest']
accuracies = [81.97,82.42,89.01,82.42]
plt.figure(figsize=(14,8))
plt.bar(models,accuracies,width=0.9)
plt.show()
fig, ax = plt.subplots(figsize = (13,13))
ax.bar(models, accuracies)
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Machine Learning Models Accuracy')
fig.show()
import gradio as gr
from PIL import Image
%matplotlib inline
def heart():
  img = Image.open("graph.jpg")
  return img

interface = gr.Interface(
      fn=heart,
      inputs= None,
      outputs=[gr.outputs.Image(type="pil", label = "Model Accuracies")],
      server_name="0.0.0.0",
  )
interface.launch()