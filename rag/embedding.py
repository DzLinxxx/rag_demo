import os
import re
import time
import requests
from typing import List
from utils import print_colorful, Fore
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


class SiliconflowEmbedding(Embeddings):
    def __init__(self, *, model_name: str, api_key: str, **kwargs) -> None:
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.cn/v1/embeddings"
        self.model_name = model_name
        # 不能有md链接 - [xx](xxx)、https://xx
        self.pattern = re.compile(r"-?\s*\[.*?\]\(.*?\)")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """

        # 删除非法的字符串
        text = self.pattern.sub("", text)
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        max_retry_times = 10
        _sleep_time = 1
        while max_retry_times > 0:
            try:
                response = requests.request(
                    "POST", self.base_url, json=payload, headers=headers
                )
                embedding = response.json()["data"][0]["embedding"]
                break
            except Exception as e:
                _sleep_time *= 1.5
                max_retry_times -= 1
                print(
                    f"速度限制, 稍后重试[{10-max_retry_times}/10]（{_sleep_time:.3f}s）：{response.json()}"
                )
                if max_retry_times == 9:
                    print(f"错误文本：|{text}|")
                time.sleep(_sleep_time)

        return embedding


def get_embeddings(embedding_cfg: dict) -> Embeddings:
    provider = embedding_cfg["provider"]
    model_name = embedding_cfg["model_name"]

    if provider == "openai":
        print_colorful(f"正在加载{model_name}模型 (openai)")
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=embedding_cfg["api_key"],
            base_url=embedding_cfg["base_url"],
        )
    elif provider == "siliconflow":
        embeddings = SiliconflowEmbedding(
            model_name=model_name, api_key=embedding_cfg["api_key"]
        )
    elif provider == "bge":
        print_colorful(f"正在加载{model_name}模型 (bge)")
        if not os.path.exists(model_name):
            print_colorful("embedding模型不存在，请检查文件路径", text_color=Fore.RED)
            return None
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=embedding_cfg["model_kwargs"],
            encode_kwargs=embedding_cfg["encode_kwargs"],
            query_instruction=embedding_cfg["query_instruction"],
        )
    elif provider == "bce":
        print_colorful(f"正在加载{model_name}模型 (bce)")
        # from BCEmbedding import EmbeddingModel
        # # init embedding model
        # embeddings = EmbeddingModel(
        #     model_name_or_path=model_name, use_fp16=embedding_cfg["use_fp16"]
        # )
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=embedding_cfg["model_kwargs"],
            encode_kwargs=embedding_cfg["encode_kwargs"],
        )
    elif provider == "jina" in model_name and not os.path.exists(model_name):
        # 使用在线的jina embedding 模型
        print_colorful(f"正在加载{model_name}模型(jina)")
        from langchain_community.embeddings import JinaEmbeddings

        embeddings = JinaEmbeddings(
            model_name=model_name, jina_api_key=embedding_cfg["api_key"]
        )
    elif "cohere" in model_name:
        # cohere在线模型
        print_colorful(f"正在加载{model_name}模型(cohere)")
        from langchain_community.embeddings import CohereEmbeddings

        embeddings = CohereEmbeddings(
            model=model_name, cohere_api_key=embedding_cfg["api_key"]
        )
    else:
        # 本地大模型 通用的huggingface embedding 模型
        print_colorful(f"正在加载{model_name}模型 (SentenceTransformerEmbeddings)")
        if not os.path.exists(model_name):
            print_colorful("embedding模型不存在，请检查文件路径", text_color=Fore.RED)
            return None

        embeddings = SentenceTransformerEmbeddings(
            model_name=model_name,
            model_kwargs=embedding_cfg["model_kwargs"],
            encode_kwargs=embedding_cfg["encode_kwargs"],
        )

    return embeddings
