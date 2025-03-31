from sentence_transformers import SentenceTransformer, util


intents = {
    "pick": ["Pick up the apple", "Grab the apple", "I want the apple", "Take the apple"],
    "reject": ["I don't want the apple", "Leave the apple", "Don't pick the apple"],
    "move": ["Move to the right", "Go left", "Turn around", "Move forward"]
}

candidate_objects = ["apple", "cup", "towel", "banana", "table", "chair"]


def classify_intent(nlp_model, command):
    """
    Classifies the intent (pick, reject, move) of the command.
    """
    command_embedding = nlp_model.encode(command, convert_to_tensor=True)
    best_intent = None
    best_score = -1
    for intent, examples in intents.items():
        for example in examples:
            example_embedding = nlp_model.encode(example, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(command_embedding, example_embedding).item()
            if similarity > best_score:
                best_score = similarity
                best_intent = intent
    return best_intent

def extract_object(nlp_model, command):
    """
    Extracts the target object from the command by comparing the transcript embedding
    against a list of candidate objects. (No NLTK used.)
    """
    transcript_embedding = nlp_model.encode(command, convert_to_tensor=True)
    best_object = None
    best_score = -1
    for obj in candidate_objects:
        obj_embedding = nlp_model.encode(obj, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(transcript_embedding, obj_embedding).item()
        if similarity > best_score:
            best_score = similarity
            best_object = obj
    return best_object if best_score > 0.5 else None