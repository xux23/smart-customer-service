"""知识库：加载 JSON、jieba 分词、TF-IDF、余弦相似度检索。

向量用稀疏字典表示，避免词表膨胀，代码也更直观。
数据量小（十几条），TF-IDF 零外部依赖且结果可解释；
将来升级 embedding 检索时只替换本文件内部实现。
"""

import json
import math

import jieba

# 停用词：代词、语气词和纯疑问引导词，对检索打分没有区分度
STOPWORDS = {
    "的", "了", "我", "你", "他", "她", "它", "我们", "你们", "他们", "咱们",
    "是", "在", "和", "就", "都", "也", "还", "很", "被", "把", "让", "用",
    "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "嘛", "呗",
    "能", "可以", "请问", "一下", "怎么", "如何", "这", "那", "有", "不",
}

# 模块级索引：服务启动时 build_index 一次，之后只读
_index: list[dict] = []


def _is_meaningful(word: str) -> bool:
    # 过滤标点等无字面意义的词，只保留含中文或字母数字的词
    for char in word:
        if "\u4e00" <= char <= "\u9fff":
            return True
        if char.isascii() and char.isalnum():
            return True
    return False


def tokenize(text: str) -> list[str]:
    words = jieba.lcut(text)
    return [
        word for word in words
        if word.strip() and word not in STOPWORDS and _is_meaningful(word)
    ]


def load_knowledge(path: str) -> int:
    with open(path, encoding="utf-8") as file:
        entries = json.load(file)
    build_index(entries)
    return len(entries)


def build_index(entries: list[dict]) -> None:
    documents_tokens = []
    document_frequency: dict[str, int] = {}
    for entry in entries:
        # keywords 与 question 拼接后一起分词，弥补单一问题表述覆盖面不足
        text = entry["question"] + " " + " ".join(entry.get("keywords", []))
        tokens = tokenize(text)
        documents_tokens.append(tokens)
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    total = len(entries)
    index = []
    for entry, tokens in zip(entries, documents_tokens):
        term_frequency: dict[str, int] = {}
        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1
        vector = {
            token: count * math.log(total / (1 + document_frequency[token]))
            for token, count in term_frequency.items()
        }
        index.append({"entry": entry, "vector": vector})
    _index.clear()
    _index.extend(index)


def search(question: str, top_n: int) -> list[tuple[dict, float]]:
    """返回 [(知识条目, 相似度分数)]，按分数降序，最多 top_n 条"""
    query_tokens = tokenize(question)
    if not query_tokens or not _index:
        return []

    query_tf: dict[str, int] = {}
    for token in query_tokens:
        query_tf[token] = query_tf.get(token, 0) + 1
    # 查询向量只用词频本身：查询没有的维度为 0，无需乘 idf 也能与文档向量同尺度比较
    query_vector = dict(query_tf)

    scored = []
    for item in _index:
        score = _cosine_similarity(query_vector, item["vector"])
        scored.append((item["entry"], score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    dot_product = 0.0
    for token, weight in vec_a.items():
        if token in vec_b:
            dot_product += weight * vec_b[token]
    norm_a = math.sqrt(sum(weight * weight for weight in vec_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)
