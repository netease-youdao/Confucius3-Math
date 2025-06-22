APP_SERVICE_PORT = 8827
APP_TEST_PORT = 8828
MODEL_CHOICES = ["Confucius3-Math"]
DEFAULT_PROMPT = """You are helpful AI."""
ALLOW_TAGS = ['think', 'answer']

VLLM_TEXT_MODELS = ['Confucius3-Math']
# 模型定义
VLLM_SERVE_NAME={
    "Confucius3-Math": {
        "model": "confucius3-math",
        "api_url": "http://127.0.0.1:8199/v1/chat/completions", 
    },
}

MAX_ATTEMPT_COUNT = 3
INTERVAL_TIME_FAILED = 2

SYSTEM_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""
USER_PROMPT_TEMPLATE = """{question}"""

max_tokens_slider = {
    'minimum': 0,
    'maximum': 16384,
    'value': 16384,
    'step': 1,
    'interactive': True,
    'label': 'Max Tokens'    
}
temperature_slider = {
    'minimum': 0,
    'maximum': 1,
    'value': 0.7,
    'step': 0.05,
    'interactive': True,
    'label': 'Temperature'    
}
top_p_slider = {
    'minimum': 0,
    'maximum': 1,
    'value': 0.7,
    'step': 0.05,
    'interactive': True,
    'label': 'Top P'    
}
presence_penalty_slider = {
    'minimum': -2,
    'maximum': 2,
    'value': 0,
    'step': 0.05,
    'interactive': True,
    'label': 'Presence Penalty'
}
frequency_penalty_slider = {
    'minimum': -2,
    'maximum': 2,
    'value': 0,
    'step': 0.05,
    'interactive': True,
    'label': 'Frequency Penalty'
}
top_k_slider = {
    'minimum': -1,
    'maximum': 20,
    'value': -1,
    'step': 2,
    'interactive': True,
    'label': 'Top K'    
}