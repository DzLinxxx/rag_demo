import os
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from rag.create_db import get_embeddings
from configs import rag_configs

# 读取所有的db
db_path = rag_configs["database"]["db_vector_path"]
index_names = set([os.path.splitext(i)[0] for i in os.listdir(db_path)])
embedding = get_embeddings(rag_configs["embedding"])

index_map = {
    name: FAISS.load_local(
        db_path,
        index_name=name,
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )
    for name in index_names
}


app = FastAPI()


# 检索相关知识
@app.get("/db/v1/retriever")
async def get_relantic_documents(query: str, index_name: str, k: int = 1):
    if index_name not in index_names:
        return {"Error": f"not find {index_name}."}
    if k < 1:
        return {"Error": f"k={k}, must be bigger than 1"}
    vectore = index_map[index_name]
    docs = vectore.similarity_search_with_relevance_scores(query, k=k)
    docs = [
        {"doc": d[0].page_content, "metadata": d[0].metadata, "score": d[1]}
        for d in docs
    ]

    return docs


# 获取所有的知识库
@app.get("/db/v1/index_name/")
async def get_index_names():
    return {"index-names": index_names}
