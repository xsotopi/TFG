import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import ast
import re

def extract_intent_and_object(command, model, tokenizer, device):
    """
    Extracts intents and objects from a voice command using a provided small language model.
    Returns:
        dict: A dictionary with the action 'pick' and a list of object to pick.
        Example: {'pick': ['apple']}
        If no action returns an empty dictionary.
    """
    # Prompt to guide the model to extract 'pick' action and its objects
    prompt = (
            f"You are an AI assistant that receives a natural language transcription of a voice command. Your task is to analyze the command and extract all the actions and their corresponding objects in the order they appear."

            f"There is only one possible action: 'pick'."

            f"Return the result as a JSON-like dictionary with the following structure:"
            f"{{"
            f"'pick': [list of objects in the order to pick them],"
            f"}}"

            f"The movement pick, should only be referencing objects, not places, so things like apple, book, glass... tangible things."
            f"Example: if we have 2 apples, in pick we should have [apple, apple]. If we have 3 books, in pick we should have [book, book, book]."

            f"Provide only the the dictionary with the extracted actions and objects. Do not include any additional text or explanations."
            f"If there are no actions or objects in the command, return an empty dictionary."
            f"Here is the command:"
            f"Command: \"{command}\"\n"
            f"Answer:"
        )
    # Prepare input for the model
    messages = [{"role": "user", "content": prompt}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate response from the model
    outputs = model.generate(inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0])

    # Parse dictionary form the model's response
    dict_matches = re.findall(r"\{[^{}]+\}", response)
    if dict_matches:
        try:
            return ast.literal_eval(dict_matches[-1])
        except Exception as e:
            print("Failed to parse dictionary:", e)
            return {}
    return {}