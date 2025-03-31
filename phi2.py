import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_intent_and_object(command):
    prompt = (
        f"Instruction: Identify the intent and the object in the following user command.\n"
        f"Command: \"{command}\"\n"
        f"Return it in the format: intent: <intent>, object: <object>\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="cpu", torch_dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

command = "Pick up the apple"
text = extract_intent_and_object(command)

print(text)



