import gradio as gr

from PIL import Image as PImage

from utils.SigLip2 import SigLip2
from utils.image_utils import heatmap_image, heatmap_image_rbf, mask_image

msl = SigLip2()

def display_heatmaps(img, text):
  text = [text]
  similarity_map_np = msl.get_gradient_activation_map(img, text)

  masked_img = mask_image(img, similarity_map_np)

  heatmap_img = heatmap_image(similarity_map_np, size=img.size, sampling=PImage.Resampling.BILINEAR)
  overlay_img = PImage.blend(img, heatmap_img.resize(img.size), 0.65)

  heatmap_img_rbf = heatmap_image_rbf(similarity_map_np, size=img.size)
  overlay_img_rbf = PImage.blend(img, heatmap_img_rbf.resize(img.size), 0.65)

  return [masked_img, overlay_img, overlay_img_rbf]


with gr.Blocks() as demo:
  gr.Markdown("# Image / Text Alignment Heatmap")
  gr.Interface(
    fn=display_heatmaps,
    api_name="heatmaps",
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Activation Term", show_label=True)],
    outputs=[gr.Image(format="jpeg"), gr.Image(format="jpeg"), gr.Image(format="jpeg")],
    flagging_mode="never",
  )

if __name__ == "__main__":
  demo.launch()
