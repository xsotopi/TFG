from transformers import AutoModelForCausalLM, AutoTokenizer
import time

import ast

import re

# Define checkpoint
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

def extract_intent_and_object(command):
    """
    Uses SmolLM model extract actions and objects from a command
    Returns:
        dict: Dictionary with actions and objects
    """

    prompt = (
            f"You are an AI assistant that receives a natural language transcription of a voice command. Your task is to analyze the command and extract all the actions and their corresponding objects in the order they appear."

            f"There are only three possible actions: 'pick', 'place'."

            f"Return the result as a JSON-like dictionary with the following structure:"
            f"{{"
            f"'pick': [list of objects in the order to pick them],"
            f"'place': [list of locations to where place the objects],"
            f"}}"

            f"The movement pick, should only be referencing objects, not places, so things like apple, book, glass... tangible things"
            f"The movement place, should only be referencing objects, not places, so things like apple, book, glass... tangible things"

            f"Provide only the the dictionary with the extracted actions and objects. Do not include any additional text or explanations."
            f"If there are no actions or objects in the command, return an empty dictionary."
            f"Here is the command:"
            f"Command: \"{command}\"\n"
            f"Answer:"
        )
    messages = [{"role": "user", "content": prompt}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    response = tokenizer.decode(outputs[0])

    dict_matches = re.findall(r"\{[^{}]+\}", response)
    if dict_matches:
        try:
            return ast.literal_eval(dict_matches[-1])
        except Exception as e:
            print("Failed to parse dictionary:", e)
            return {}
    return {}


# Model loading and main execution
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

command = "Take the apple from the fridge, place it in the table and then take the glass from the table and put it in the sink."
start = time.time()
text = extract_intent_and_object(command)
print(text)
end = time.time()
print("SMOLLM:", end - start)
