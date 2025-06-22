import logging
from api.doubao_client import DoubaoClient
from utils.image_utils import encode_image  # 更新导入语句以反映新位置

class OCRService:
    def __init__(self, api_client: DoubaoClient):
        """
        初始化OCRService
        
        Args:
            api_client: ApiClient的实例
        """
        self.api_client = api_client
        logging.info("OCRService initialized.")

    def get_ocr(self, problem: str) -> str:
        """
        获得问题的ocr。

        Args:
            problem: 需要解答的问题图片。

        Returns:
            问题的ocr结果。
        """
        # 将图片转换为base64编码
        base64_image = encode_image(problem)

        # 构建请求消息，包含图片
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64_image}'
                        }
                    },
                    {
                        'type': 'text',
                        'text': '请将图片中的内容转换为文本。\n注意数学公式需要用latex格式输出。\n注意只输出文本内容，不要输出任何解释。'
                    }
                ]
            }
        ]

        logging.info(f"Calling API for ocr for problem: {problem[:50]}...") # Log first 50 chars
        solution = self.api_client.chat_completion(messages, temperature=0.6, max_tokens=2048)

        if solution is None:
            logging.error("Failed to get ocr from API.")
            return "ocr出现错误，请检查输入或稍后重试。"
        
        logging.info("Successfully got ocr.")
        return solution