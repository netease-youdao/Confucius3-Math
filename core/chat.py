from config.config import (
    VLLM_SERVE_NAME, 
    VLLM_TEXT_MODELS, 
    MAX_ATTEMPT_COUNT, 
    INTERVAL_TIME_FAILED, 
    USER_PROMPT_TEMPLATE, 
    SYSTEM_PROMPT_TEMPLATE
)
from core import OCRService, MockOCRService
from api.doubao_client import DoubaoClient
import aiohttp
import json
import base64
import asyncio

_doubao_client = DoubaoClient()
_ocr_service = OCRService(_doubao_client)

def reset(z):
    return [], []

async def connect_to_vllm_text_model(
    model, formatted_messages, max_tokens, temperature, top_p, top_k, stop, 
    presence_penalty, frequency_penalty, stream=True
):
    """
    连接到vllm模型API并获取流式响应
    """
    query = {
        'model': VLLM_SERVE_NAME[model]["model"],
        'messages': formatted_messages,
        'max_tokens': max_tokens - len(str(formatted_messages)),
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'stop': stop,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'stream': stream  # 启用流式响应
    }
    
    # 错误提示作为默认返回
    error_message = "总token超出max_token限制，请通过clear重启对话。"
    
    for attempt in range(MAX_ATTEMPT_COUNT):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    VLLM_SERVE_NAME[model]["api_url"], 
                    json=query,
                    timeout=aiohttp.ClientTimeout(total=10000)
                ) as response:
                    # 检查HTTP状态码
                    if response.status == 200:
                        # 处理流式响应
                        if stream:
                            full_response = ""
                            async for line in response.content:
                                line = line.decode('utf-8').strip()
                                if line.startswith('data: ') and not line.endswith('[DONE]'):
                                    data = json.loads(line[6:])
                                    if 'choices' in data and len(data['choices']) > 0:
                                        if 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                                            content = data['choices'][0]['delta']['content']
                                            if content:  # 忽略空内容
                                                yield content  # 流式返回当前内容
                        else:
                            # 非流式响应处理
                            response_data = await response.json()
                            yield response_data['choices'][0]['message']['content']
                        return  # 成功返回后退出重试循环
                    else:
                        # 处理非OK状态码
                        error_data = await response.json()
                        error_message = error_data.get('desc', f'HTTP Error {response.status}')
                        raise ValueError(f"HTTP Error {response.status}: {error_message}")
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            if attempt < MAX_ATTEMPT_COUNT - 1:
                await asyncio.sleep(INTERVAL_TIME_FAILED)  # 等待后重试
            else:
                yield error_message  # 最后一次尝试失败后返回错误信息
        except Exception as e:
            if attempt < MAX_ATTEMPT_COUNT - 1:
                await asyncio.sleep(INTERVAL_TIME_FAILED)
            else:
                yield error_message 

def format_messages(model, messages):
    if model == "Confucius3-Math":
        if len(messages) == 2:
            formatted_messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT_TEMPLATE},
                {'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(question=messages[-1]['content'][0]['text'])},
            ]
        else:
            system = [{'role': 'system', 'content': SYSTEM_PROMPT_TEMPLATE}]
            history = []
            for message in messages[1:-1]:
                if "url" in str(message):
                    try:
                        image_path = str(message).split("'url': '")[1].split("‘")[0]
                        ocr_result = _ocr_service.get_ocr(image_path)
                        history.append({'role': 'user', 'content': ocr_result})
                    except Exception as e:
                        history.append(message)
                else:
                    message['content'] = message['content'][0]['text']
                    message['content'] = message['content'].replace('==思考开始==', '<think>').replace('==思考结束==', '</think>')
                    history.append(message) 
                    
            question = [{'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(question=messages[-1]['content'][0]['text'])}]
            formatted_messages = system + history + question
        return formatted_messages
    else:
        return messages

async def call_streaming_api(model, messages, max_tokens=4096, temperature=1, top_p=0.1, top_k=-1, presence_penalty=0, frequency_penalty=0):
    if model in VLLM_TEXT_MODELS:
        try:
            formatted_messages = format_messages(model, messages)
            
            buffer = ""  # 新增缓冲区用于完整响应
            async for chunk in connect_to_vllm_text_model(
                model=model,
                formatted_messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stream=True
            ):
                try:
                    if chunk:
                        buffer += chunk
                        yield (chunk, buffer)  # 同时返回当前chunk和完整响应

                except Exception as e:
                    yield (f"数据流解析异常: {str(e)}", "")
                    return
            yield (None, buffer) # Indicate completion after the loop
        except aiohttp.ClientError as e:
            yield (f"网络连接异常: {str(e)}", "")
        except Exception as e:
            yield (f"未知错误: {str(e)}", "")

    
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

async def chat(message, history, model_selection, max_tokens, temperature, top_p, top_k, presence_penalty, frequency_penalty):
    # 确保消息始终是字典格式
    if not isinstance(message, dict):
        message = {"text": message, "files": []}
    
    user_text = message.get("text", "")
    user_files = message.get("files", [])
    
    # 构建消息（保持原有逻辑不变）
    model = model_selection
    system_prompt = """You are a helpful AI."""
    full_message = []
    
    system_message = {"role": "system", "content": system_prompt}
    full_message.append(system_message)
    
    # 处理历史消息（保持原有逻辑不变）
    i = 0
    while i < len(history):
        hist = history[i]
        if isinstance(hist['content'], tuple):
            image_path = hist['content'][0]
            hist['content'] = _ocr_service.get_ocr(image_path)
        hist['content'] = [{'type': 'text', 'text': hist['content']}]
        full_message.append(hist)
        i += 1
    
    # 处理当前用户输入
    user_content = []
    if user_files:
        if user_text:
            user_content.append({"type":"text", "text": user_text})
        for user_file in user_files:
            ocr_result = _ocr_service.get_ocr(user_file)
            user_content = [{"type":"text", "text":ocr_result}]
    else:
        user_content.append({"type":"text", "text": user_text})
    
    user_message = {"role": "user", "content": user_content}
    full_message.append(user_message)
    
    # 流式响应（返回OpenAI格式的字典列表）
    accumulated_response = ""
    async for chunk, _ in call_streaming_api(model, full_message, max_tokens, temperature, top_p, top_k, presence_penalty, frequency_penalty):
        if chunk is None:
            break
        if chunk.startswith("错误"):
            yield [{"role": "assistant", "content": chunk}]  # 错误消息
            return
        chunk = chunk.replace("<think>", "==思考开始==").replace("</think>", "==思考结束==")
        accumulated_response += chunk
        # 每次返回助手的部分回复（OpenAI格式）
        yield [{"role": "assistant", "content": accumulated_response}]
