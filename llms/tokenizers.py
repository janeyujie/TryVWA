from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore
import logging
logger = logging.getLogger("logger")

class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            try:
                # 尝试按模型名称自动加载
                #logger.info(f"[Tokenizer] 尝试自动加载分词器，模型: {model_name}")
                self.tokenizer = tiktoken.encoding_for_model(model_name)
                #logger.info(f"[Tokenizer] 成功自动加载分词器: {model_name}")

            except KeyError:
                # 自动加载失败，回退到 cl100k_base (用于 qwen 等模型)
                # logger.warning(
                #     #f"[Tokenizer] 自动映射 '{model_name}' 失败。"
                #     #f" 正在回退到 'cl100k_base'..."
                # )
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                    #logger.info("[Tokenizer] 成功加载 'cl100k_base' 作为后备。")
                except Exception as e:
                    # 如果连 cl100k_base 都加载失败了，这是个大问题
                    logger.error(f"[Tokenizer] 错误: 加载 'cl100k_base' 失败: {e}")
                    raise e  # 抛出异常，停止程序
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
