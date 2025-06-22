import logging

class MockOCRService:
    def __init__(self, api_client):
        """
        初始化MockOCRService
        
        Args:
            api_client: ApiClient的实例
        """
        self.api_client = api_client
        logging.info("MockOCRService initialized.")

    def get_ocr(self, problem: str) -> str:
        """
        获取问题的模拟ocr结果。
        Args:
            problem: 需要解答的问题字符串。
        Returns:
            模拟ocr字符串。
        """
        return """(8) 函数 $y = \\frac{\\sqrt{2-x}}{\\lg (x+1)}$ 的定义域是 __________"""