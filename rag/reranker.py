import os
import requests
from utils import print_colorful, Fore


def get_reranker(reranker_cfg: dict):
    """
    返回的实例，必须实现compute_score方法
    compute_score([['query','doc1'], ['query','doc2']])
    """
    if not reranker_cfg:
        return None

    provider = reranker_cfg["provider"]
    model_name = reranker_cfg["model_name"]
    if provider == "jina":
        print_colorful(f"正在加载{model_name}模型 (jina)")

        class JinaReranker:
            def __init__(self, model_name, api_key) -> None:
                self.model = model_name
                self.api_key = api_key
                self.base_url = "https://api.jina.ai/v1/rerank"

            def _get_score(self, query, docs, top_n=None):
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {
                    "model": self.model,
                    "query": query,
                    "documents": docs,
                    "top_n": top_n,
                }
                response = requests.post(self.base_url, headers=headers, json=data)
                response = sorted(response.json()["results"], key=lambda x: x["index"])
                response = [i["relevance_score"] for i in response]
                return response

            def compute_score(self, docs: list, top_n=None):
                if not docs:
                    return []
                query = docs[0][0]
                docs = [i[1] for i in docs]
                top_n = top_n or len(docs)
                return self._get_score(query=query, docs=docs, top_n=top_n)

        reranker = JinaReranker(model_name, reranker_cfg["api_key"])

    elif provider == "bce":
        print_colorful(f"正在加载{model_name}模型 (bce)")
        from BCEmbedding import RerankerModel

        reranker = RerankerModel(
            model_name_or_path=model_name, use_fp16=reranker_cfg["use_fp16"]
        )
    elif provider == "siliconflow":

        class SiliconflowReranker:
            def __init__(self, model_name, api_key) -> None:
                self.model = model_name
                self.api_key = api_key
                self.base_url = "https://api.siliconflow.cn/v1/rerank"

            def _get_score(self, query, docs, top_n=None) -> list[float]:
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": docs,
                    "top_n": top_n,
                    "return_documents": True,
                    "max_chunks_per_doc": 123,
                    "overlap_tokens": 79,
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                response = requests.request(
                    "POST", self.base_url, json=payload, headers=headers
                )
                response = sorted(response.json()["results"], key=lambda x: x["index"])
                response = [i["relevance_score"] for i in response]
                return response

            def compute_score(self, docs: list, top_n=None) -> list[float]:
                if not docs:
                    return []
                query = docs[0][0]
                docs = [i[1] for i in docs]
                top_n = top_n or len(docs)
                return self._get_score(query=query, docs=docs, top_n=top_n)

        reranker = SiliconflowReranker(model_name, reranker_cfg["api_key"])

    elif os.path.exists(model_name):
        from FlagEmbedding import FlagReranker

        print_colorful(f"正在加载{model_name}模型 (FlagEmbedding)")
        reranker = FlagReranker(model_name, use_fp16=reranker_cfg["use_fp16"])
    else:
        print_colorful("reranker模型不存在，请检查文件路径", text_color=Fore.RED)
        reranker = None

    return reranker
