from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""

USER_PROMPT_TEMPLATE = """{question}"""

model_name = "netease-youdao/Confucius3-Math"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT_TEMPLATE},
    {'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(question=question)},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def parse_result_nostep(result):
     think_pattern = r"<think>(.*?)</think>(.*)"

    think_list = re.findall(think_pattern, result, re.DOTALL)

    assert len(think_list) == 1, \
        f"The parsing results do not meet the expectations.\n{result}"

    think = think_list[0][0].strip()
    summary = think_list[0][1].strip()
    return think, summary

thinking, summary = parse_result_nostep(response)