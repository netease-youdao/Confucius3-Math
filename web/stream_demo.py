import gradio as gr
from core.chat import chat
import os
from PIL import Image
import io
import base64
from config.config import (
    APP_SERVICE_PORT,
    APP_TEST_PORT,
    MODEL_CHOICES,
    DEFAULT_PROMPT,
    ALLOW_TAGS,
    max_tokens_slider,
    temperature_slider,
    top_p_slider,
    presence_penalty_slider,
    frequency_penalty_slider,
    top_k_slider,
)

app_port = APP_SERVICE_PORT

def create_slider(params):
    return gr.Slider(
        minimum=params['minimum'],
        maximum=params['maximum'],
        value=params['value'],
        step=params['step'],
        interactive=params['interactive'],
        label=params['label']
    )

# 将logo转换为base64编码（便于在CSS中引用）
def logo_to_base64(logo):
    buffer = io.BytesIO()
    logo.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

logo_path = os.path.join(os.path.dirname(__file__), "../assets/confucius_logo.png")
icons = ["assets/icon/1.png", "assets/icon/2.png"]
logo = Image.open(logo_path)
logo_base64 = logo_to_base64(logo)

# 定义CSS样式，插入logo并设置位置（示例中放在左上角）
css = """
.container {
    text-align: center;
    padding: 20px;
}

.icon {
    width: 60px;
    height: 60px;
    background: linear-gradient(45deg, #6666ff, #9933ff);
    clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
    display: inline-block;
    vertical-align: middle;
    margin-right: 10px;
}

h1 {
    display: inline-block;
    vertical-align: middle;
    margin: 0;
    font-size: 28px;
}

p {
    margin: 10px 0 0 0;
    font-size: 16px;
}
"""


with gr.Blocks() as demo:
    gr.HTML(f'''
<div class="container" style="display: flex; align-items: center; justify-content: center;">
    <img src="data:image/png;base64,{logo_base64}" alt="logo" style="width:150px; height:80px; margin-right: 10px;">
    <h1 style="margin: 0; font-size: 28px;">Hello, I'm Confucius3</h1>
</div>
''')

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            model_selection = gr.Dropdown(choices=MODEL_CHOICES, value='Confucius3-Math', label="Model Selection")
            with gr.Accordion("Parameters", open=True):
                max_tokens = create_slider(max_tokens_slider)
                temperature = create_slider(temperature_slider)
                top_p = create_slider(top_p_slider)
                top_k = create_slider(top_k_slider)
                presence_penalty = create_slider(presence_penalty_slider)
                frequency_penalty = create_slider(frequency_penalty_slider)
        
        with gr.Column(scale=3):
            # 只使用 ChatInterface，移除手动创建的 Chatbot 和 MultimodalTextbox
            chat_interface = gr.ChatInterface(
                fn=chat,
                type='messages',
                multimodal=True,
                fill_height=True,
                additional_inputs=[model_selection, max_tokens, temperature, top_p, top_k, presence_penalty, frequency_penalty],
                examples=[
                    ["函数 $y = \\frac{\\sqrt{2-x}}{\\lg (x+1)}$ 的定义域是 __________"],
                    ['''存在实数 \( x \)，使得 \( x^2 + 2ax + 2 - a > 0 \) 成立，试求实数 \( a \) 的取值范围.'''],
                ],
                example_icons=[
                    icons[0],
                    icons[1],
                ],
            )
    
    # 模型选择变更时重置对话
    model_selection.input(
        fn=lambda: ([], []), 
        inputs=[],
        outputs=[chat_interface.chatbot, chat_interface.chatbot_state]
    )

demo.launch(share=False, debug=True, server_port=app_port, server_name="0.0.0.0", max_threads=20)
