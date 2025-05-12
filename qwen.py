from transformers import AutoModelForCausalLM, AutoTokenizer
import time
model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

command = "pick up the apple and place it on the table"
# prepare the model input
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
messages = [
    {"role": "user", "content": prompt}
]

t0 = time.time()
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=100
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

t1 = time.time()
print("time:", t1-t0)