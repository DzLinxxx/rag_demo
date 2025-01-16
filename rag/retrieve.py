import os
import re
import streamlit as st

# from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import ConfigurableField
from utils import print_colorful, Fore
from configs import rag_configs
from rag.create_db import get_embeddings
from rag.reranker import get_reranker


@st.cache_resource
def get_embeddings_wrapper(configs):
    return get_embeddings(configs)


@st.cache_resource
def get_reranker_wrapper(config):
    return get_reranker(config)


def get_retriever(embeddings, database_cfg: dict):
    if embeddings is None:
        return None
    if database_cfg["db_type"] == "faiss":
        print_colorful("正在加载FAISS向量数据库")
        if not os.path.exists(database_cfg["db_vector_path"]):
            print_colorful("数据库不存在，请检查文件路径", text_color=Fore.RED)
            return None
        try:
            faiss_vectorstore = FAISS.load_local(
                folder_path=database_cfg["db_vector_path"],
                embeddings=embeddings,
                index_name=database_cfg["index_name"],
            )
        except ValueError:
            faiss_vectorstore = FAISS.load_local(
                folder_path=database_cfg["db_vector_path"],
                embeddings=embeddings,
                index_name=database_cfg["index_name"],
                allow_dangerous_deserialization=True,
            )
        retriever = faiss_vectorstore.as_retriever(**database_cfg["faiss_params"])
        retriever.configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs_faiss",
                name="Search Kwargs",
                description="The search kwargs to use",
            )
        )
    else:
        retriever = None
        raise NotImplementedError("没有定义此向量数据的实现")

    # bm25_retriever = BM25Retriever.from_documents(docs_chunks, k=3)
    # bm25_retriever.configurable_fields(
    #     k=ConfigurableField(
    #         id="search_kwargs_bm25",
    #         name="Search Kwargs",
    #         description="The search kwargs to use",
    #     )
    # )
    # retriever = EnsembleRetriever(
    #     retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
    # )

    return retriever


def init_models(configs: dict = {}):
    configs = configs or rag_configs

    # embedding model
    embeddings = get_embeddings_wrapper(configs["embedding"])

    # reranker model
    reranker = get_reranker_wrapper(configs["reranker"])

    # retriever
    Retriever = get_retriever(embeddings, configs["database"])

    return Retriever, reranker


REWRITE_SYSTEM_PROMPT = """
你是一个逻辑推断能力非常强的专家。你的任务是根据用户给的A和B之间的对话历史，将A最后说的内容转换成一句语义明确、没有歧义、指代消解后的内容。

例如：

A：帮我计算一下1+1=？
B：好的。1+1=2
A：谢谢你
A实际想说：谢谢你

A：北京今天多少度？
B：北京今天的温度是21度。
A：那上海的呢？
A实际想说：上海今天多少度？

核心点：
- 如果A最后说的内容与上下文信息有关，则需要结合上下文将其改写成一句语义明确、没有歧义、指代消解后的内容；
- 如果没有关系，则返回原内容；
- 只需要改写，不需要具体回答！！！
""".strip()

CONDITION_SYSTEM_PROMPT = """
你是一名数据分析专家。现在你的任务是分析用户提供的一段客服和用户A的对话内容，判断是否需要检索知识库才能回答用户A“最后”说的内容。
* 如果需要进行检索，则将用户最后说的内容改写成“语义明确、没有歧义、指代消解”的新内容，必须是陈述句。
* 如果不需要则返回：#不需要检索@

回答格式：
结论：<#需要检索@/#不需要检索@>
理由：<检索/不检索的理由>
改写：<改写后的内容，必须是陈述句>

例子1：
结论：#需要检索@
理由：结合客服和用户A的对话记录，可以总结到用户A想知道RAG是什么
改写：rag的含义和技术细节

例子2：
结论：#不需要检索@
理由：结合客服和用户A的对话记录，用户A想问计算1+1=多少，很简单的问题，不需要进行检索

开始！
""".strip()


def msgs_to_str(msgs: list[dict]):
    text = ""
    for msg in msgs:
        if msg["role"] == "system":
            continue

        if msg["role"] in ["user", "human"]:
            text += f'用户A：{msg["content"]}\n'
        else:
            text += f'客服：{msg["content"]}\n'

    return text.strip()


def rewrite_query(llm_generate, messages):
    messages = [
        msg for msg in messages if msg["role"] in ["user", "assistant", "ai", "human"]
    ]

    print_colorful(f"query:{messages[-1]['content']}")
    # resp = llm_generate(
    #     messages=[
    #         {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
    #         {"role": "user", "content": msgs_to_str(messages)},
    #     ],
    #     temperature=0.0,
    # )
    # requery = resp.choices[0].message.content.strip("A实际想说：").strip()
    resp = llm_generate(
        messages=[
            {"role": "system", "content": CONDITION_SYSTEM_PROMPT},
            {"role": "user", "content": msgs_to_str(messages)},
        ],
        temperature=0.0,
    )
    resp = resp.choices[0].message.content.strip()
    print_colorful(resp)
    if "#需要检索@" in resp:
        requery = [r for r in resp.split("\n") if r.startswith("改写：")][0][3:].strip()
    else:
        requery = None
    print_colorful(f"requery:{requery}")

    return requery


def fetch_relevant_docs(query, retriever=None, reranker=None, configs: dict = {}):
    """从知识库检索相关性的信息并进行重排序
    Params:
        query: 检索信息
        retriever: 检索器
        reranker: 重排序模型
        configs: 配置文件
    Return:
        return:list,[('doc', 'metadata'),...]

    """
    if retriever is None:
        return []

    # 混合检索
    # retriever_config = {"configurable": {"search_kwargs_faiss": {"k": 1}, "search_kwargs_gm25": 1}}
    retriever_config = configs.get("retriever_config", {})
    docs = retriever.invoke(query, config=retriever_config)
    print_colorful(f"共检索到 {len(docs)} 个docs", text_color=Fore.RED)

    # 重排序
    if reranker is not None:
        docs_content = [d.page_content for d in docs]

        # score排序
        sample_top = configs.get("sample_top", 1)
        scores = reranker.compute_score([[query, kn] for kn in docs_content])
        scores = scores if isinstance(scores, list) else [scores]
        for d, score in zip(docs, scores):
            d.metadata["score"] = score
        docs = sorted(docs, key=lambda x: x.metadata["score"], reverse=True)
        for idx, d in enumerate(docs, 1):
            print(
                repr(f'{idx}->{d.page_content[:10]}..., score: {d.metadata["score"]}')
            )
        print_colorful(f"提取前topk {sample_top}")
        docs = docs[:sample_top]

        # 相似度过滤
        if sample_threshold := configs.get("sample_threshold"):
            docs = [d for d in docs if d.metadata["score"] >= sample_threshold]
        print_colorful(f"相似度阈值 {sample_threshold}")
        print_colorful(f"过滤后得到 {len(docs)} 个docs")

    return docs


Knownledge_QA_prompt = """
下边是从数据库中检索到的相关内容。如果这些内容与问题相关，则参考其进行回答；否则忽略这些内容。
--------------------

{context}

--------------------
我的问题是：{question}
""".strip()


def create_prompt(QA_prompt, question, docs):
    """得到最后的prompt"""
    info = ""
    for idx, d in enumerate(docs, 1):
        page_content = re.sub(r"\n+", "\n", d.page_content)
        info += f"内容 {idx}：```{page_content}```"
        info += f"(数据源:'{d.metadata['source']}')"
        info += "\n\n"
    info = info.strip() or "<没有检索到相关内容>"

    return QA_prompt.format(context=info, question=question)


def retrieve(
    llm_generate,
    messages,
    retriever,
    reranker,
    rewrite=True,
    show_info=False,
    configs={},
):
    configs = {
        "sample_top": 5,  # 参考检索的数量 现在是三个，就改这个值
        "sample_threshold": -1000,
        # "retriever_config": {  # 初始化已经配置
        #     "configurable": {
        #         "search_kwargs_faiss": {"search_type": "similarity", "k": 10},
        #         # "search_kwargs_gm25": 2,
        #     }
        # },
        **configs,
    }
    docs = []

    if rewrite:
        requery = rewrite_query(llm_generate, messages)
    else:
        requery = messages[-1]["content"]
    if requery is not None:
        docs = fetch_relevant_docs(
            requery,
            retriever=retriever,
            reranker=reranker,
            configs=configs,
        )
        query = create_prompt(Knownledge_QA_prompt, messages[-1]["content"], docs)

        if show_info:
            print("=" * 20)
            print(query)
            print("=" * 20)

        messages[-1]["content"] = query
        # sources = [i.metadata for i in docs]

    return messages, docs
