import numpy as np
from torch import cuda, no_grad, relu, Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor

class SigLip2:
  # MODEL_NAME = "google/siglip2-giant-opt-patch16-256"  # from utils
  # MODEL_NAME = "google/siglip2-so400m-patch16-512"     # bigger image
  # MODEL_NAME = "google/siglip2-so400m-patch14-384"     # more patches
  MODEL_NAME = "siglip2-so400m-patch16-256"
  DEVICE = "cuda" if cuda.is_available() else "cpu"

  @staticmethod
  def scaleMinMax(data):
    dmin = data.min()
    dmax = data.max()
    return (data - dmin) / (dmax - dmin + 1e-10)

  def __init__(self, model=None):
    model_name = SigLip2.MODEL_NAME if model is None else model
    self.processor = AutoProcessor.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(self.DEVICE)

    self.model_grid_size = (
      self.model.config.vision_config.image_size //
      self.model.config.vision_config.patch_size
    )
    self.model_scale = self.model.logit_scale.exp().detach().cpu().numpy()
    self.model_bias = self.model.logit_bias.detach().cpu().numpy()

  def get_gradient_activation_map(self, img, texts, *, img_idx=0, text_idx=None):
    if isinstance(texts, str):
      texts = [texts]

    if isinstance(texts, list) and isinstance(texts[0], str):
      inputs = self.processor(
        text=texts,
        padding="max_length", max_length=64, truncation=True,
        return_tensors="pt"
      ).to(self.model.device)

      with no_grad():
        text_embs = self.model.text_model(**inputs).pooler_output
        texts = text_embs.detach().cpu().numpy()

    if isinstance(texts, np.ndarray) and len(texts.shape) == 1:
      texts = texts[None, :]

    if (isinstance(texts, np.ndarray) or isinstance(texts, Tensor)) and len(texts.shape) == 2:
      texts = Tensor(texts).to(self.model.device)
      return self.get_gradient_activation_map_from_embeddings(img, texts, img_idx=img_idx, text_idx=text_idx)
    else:
      raise TypeError(f"Expected a 2D np.ndarray or Tensor, got {type(texts)}")

  def get_gradient_activation_map_from_embeddings(self, img, texts, *, img_idx=0, text_idx=None):
    text_idxs = range(len(texts)) if text_idx is None else [text_idx]
    text_activations = []

    inputs = self.processor(
      images=img,
      padding="max_length", max_length=64, truncation=True,
      return_tensors="pt"
    ).to(self.model.device)

    outputs = self.model.vision_model(**inputs)
    img_embedding = F.normalize(outputs.pooler_output[img_idx], p=2, dim=-1)

    patch_features = outputs.last_hidden_state
    patch_features.retain_grad()

    for tidx in text_idxs:
      text_embedding = F.normalize(texts[tidx], p=2, dim=-1)
      similarity_score = (text_embedding * img_embedding).sum()

      self.model.zero_grad()
      similarity_score.backward(retain_graph=True) # need L4 GPU with 24GB (15.5GB)
      patch_grads = patch_features.grad
      patch_weights = patch_grads[img_idx].mean(dim=0)

      cam = (patch_weights * patch_features[img_idx]).sum(dim=-1)
      cam = relu(cam).detach().cpu().numpy()
      text_activations.append(SigLip2.scaleMinMax(cam))

    cam01 = SigLip2.scaleMinMax(np.array(text_activations).sum(axis=0))
    return cam01.reshape(self.model_grid_size, self.model_grid_size)
