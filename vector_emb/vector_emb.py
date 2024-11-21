from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


def get_best_match(object_descriptions, required_description):
    object_embeddings = model.encode(object_descriptions, convert_to_tensor=True)
    required_embedding = model.encode(required_description, convert_to_tensor=True)

    cosine_scores = util.cos_sim(required_embedding, object_embeddings)[0]
    best_match_idx = torch.argmax(cosine_scores).item()
    return object_descriptions[best_match_idx]
