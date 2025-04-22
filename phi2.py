import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import ast

def extract_intent_and_object(command):
    prompt = (
        f"You are an AI assistant that receives a natural language transcription of a voice command. Your task is to analyze the command and extract all the actions and their corresponding objects in the order they appear."

        f"There are only three possible actions: 'pick', 'place', and 'move'."

        f"Return the result as a JSON-like dictionary with the following structure:"
        f"{{"
        f"'pick': [list of objects in the order to pick them],"
        f"'place': [list of objects in the order to place them],"
        f"}}"

        f"The movement pick, should only be referencing objects, not places, so things like apple, book, glass... tangible things"
        f"The movement place, should only be referencing objects, not places, so things like apple, book, glass... tangible things"

        f"Provide only the the dictionary with the extracted actions and objects. Do not include any additional text or explanations."
        f"If there are no actions or objects in the command, return an empty dictionary."
        f"Here is the command:"
        f"Command: \"{command}\"\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in response:
        only_response = response.split("Answer:")[1].strip()
    else:
        only_response = response.strip()

    try:
        # Convert to actual Python dictionary
        response_dict = ast.literal_eval(only_response)
    except Exception as e:
        print("⚠️ Failed to parse model response into dictionary:", only_response)
        raise e

    return response_dict


model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="cpu", torch_dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

command = "Take the apple from the fridge, place it in the table."
start = time.time()
text = extract_intent_and_object(command)
end = time.time()
print("PHI-2 timing:", end - start)


print(text)


