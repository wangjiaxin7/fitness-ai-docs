"""
Embedding RAG: 基于向量检索的知识库增强生成
使用 SiliconFlow BAAI/bge-m3 做 embedding，本地余弦相似度检索
"""
import os, json, math, hashlib
from pathlib import Path

# 加载 .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

EMBEDDING_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"
KNOWLEDGE_DIR = os.environ.get("KNOWLEDGE_DIR", "/app/knowledge")
INDEX_PATH = os.environ.get("KNEDX_PATH", "/app/knowledge/index.json")
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
CHUNK_SIZE = 500       # 每个 chunk 最大字符数
CHUNK_OVERLAP = 50     # chunk 重叠字符数
TOP_K = 3              # 返回最相关的 chunk 数

# 跳过的文件（非知识文档）
SKIP_FILES = {
    "ARCHITECTURE.md", "问题排查记录.md", "长期记忆系统方案.md",
    "测试报告.md", "问题记录.md",
}


def _chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """将长文本切成重叠的 chunks"""
    # 按段落先分割
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current)
            # 保留最后一部分作为 overlap
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text[:chunk_size]]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """调用 SiliconFlow embedding API"""
    import urllib.request
    if not API_KEY:
        raise ValueError("SILICONFLOW_API_KEY 未配置")

    # API 每次最多处理一定数量，分批
    all_embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        data = json.dumps({
            "model": EMBEDDING_MODEL,
            "input": batch,
        }).encode()
        req = urllib.request.Request(EMBEDDING_API_URL, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        })
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        for item in result["data"]:
            all_embeddings.append(item["embedding"])
    return all_embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度"""
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_documents() -> list[dict]:
    """加载知识目录下所有 markdown 文档，返回 [{title, text, source}]"""
    docs = []
    knowledge_dir = Path(KNOWLEDGE_DIR)
    if not knowledge_dir.exists():
        return docs

    for md_file in sorted(knowledge_dir.rglob("*.md")):
        if md_file.name in SKIP_FILES:
            continue
        text = md_file.read_text(encoding="utf-8")
        if len(text.strip()) < 50:
            continue
        # 相对路径作为 source
        source = str(md_file.relative_to(knowledge_dir))
        title = md_file.stem
        # 尝试从第一行提取标题
        first_line = text.split("\n")[0].strip().lstrip("#").strip()
        if first_line:
            title = first_line
        docs.append({"title": title, "text": text, "source": source})
    return docs


def build_index():
    """构建索引：加载文档 → 切 chunk → 编码 → 保存"""
    docs = _load_documents()
    if not docs:
        print(f"未找到知识文档: {KNOWLEDGE_DIR}")
        return

    # 切 chunks
    all_chunks = []
    for doc in docs:
        chunks = _chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "title": doc["title"],
                "source": doc["source"],
                "chunk_idx": i,
            })

    print(f"加载 {len(docs)} 篇文档，切分为 {len(all_chunks)} 个 chunks")

    # 编码（加标题前缀提高语义精度）
    texts_to_embed = [f"【{c['title']}】{c['text']}" for c in all_chunks]
    embeddings = _embed_texts(texts_to_embed)
    print(f"编码完成，维度: {len(embeddings[0])}")

    # 保存索引
    index = {
        "chunks": all_chunks,
        "embeddings": embeddings,
        "model": EMBEDDING_MODEL,
    }
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)
    print(f"索引已保存: {INDEX_PATH} ({len(all_chunks)} chunks)")


def search(query: str, top_k: int = TOP_K) -> list[str]:
    """向量检索：返回最相关的 chunk 文本列表"""
    if not API_KEY:
        return []

    # 加载索引
    if not Path(INDEX_PATH).exists():
        try:
            build_index()
        except Exception as e:
            print(f"构建索引失败: {e}")
            return []

    try:
        with open(INDEX_PATH) as f:
            index = json.load(f)
    except Exception:
        return []

    chunks = index.get("chunks", [])
    embeddings = index.get("embeddings", [])
    if not chunks or not embeddings:
        return []

    # 编码 query
    try:
        query_embeddings = _embed_texts([query])
        query_vec = query_embeddings[0]
    except Exception as e:
        print(f"Query 编码失败: {e}")
        return []

    # 计算相似度
    scores = []
    for i, emb in enumerate(embeddings):
        sim = _cosine_similarity(query_vec, emb)
        scores.append((sim, i))

    # 取 top_k
    scores.sort(reverse=True)
    results = []
    seen_titles = set()
    for sim, idx in scores[:top_k * 2]:  # 多取一些去重
        chunk = chunks[idx]
        # 同一篇文章最多取 1 个 chunk
        if chunk["title"] in seen_titles:
            continue
        seen_titles.add(chunk["title"])
        results.append(chunk["text"])
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        query = sys.argv[2] if len(sys.argv) > 2 else "深蹲膝盖疼怎么办"
        results = search(query)
        print(f"\n查询: {query}")
        print(f"返回 {len(results)} 条结果:")
        for i, r in enumerate(results):
            print(f"\n--- [{i+1}] ---")
            print(r[:200] + "...")
    else:
        print("用法: python rag.py build | python rag.py test [query]")
