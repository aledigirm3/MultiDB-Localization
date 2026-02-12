import torch
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device_name: str = "cpu"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.device = self.assign_device(device_name)
        self.model = self.model.to(self.device)
        self.embedding_size = self.model.get_sentence_embedding_dimension()

    def get_sentence_embedding(self, sentence: str, as_tensor: bool = True):
        return self.model.encode(sentence, convert_to_tensor=as_tensor)

    def get_sentences_embeddings(self, sentences: list[str], as_tensor: bool = True):
        return self.model.encode(sentences, convert_to_tensor=as_tensor)

    def assign_device(self, device_name: str):
        if device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU (CUDA).")
        elif device_name == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS).")
        else:
            device = torch.device("cpu")
            print("Using CPU.")
        return device