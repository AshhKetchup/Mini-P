from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


def get_best_match(object_descriptions, required_description):
    object_embeddings = model.encode(object_descriptions, convert_to_tensor=True)
    required_embedding = model.encode(required_description, convert_to_tensor=True)

    cosine_scores = util.cos_sim(required_embedding, object_embeddings)[0]
    threshold = 0.8

    best_match_index = None 
    best_score = 0.0

    for i, score in enumerate(cosine_scores):
        if score > best_score and score >= threshold:
            best_match_index = i
            best_score = score

    return best_match_index
