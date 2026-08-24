"""知识检索核心测试：分词、相似度、排序、空输入"""

from app.core import kb

# 四条样本保证 idf = log(N/(1+df)) 在 df=1 时为正数（N=2 时恒为 0）
SAMPLE_ENTRIES = [
    {
        "id": 1,
        "category": "after_sale",
        "question": "如何申请退货",
        "answer": "答案一",
        "keywords": ["退货", "退款", "七天无理由"],
    },
    {
        "id": 2,
        "category": "consultation",
        "question": "怎么修改密码",
        "answer": "答案二",
        "keywords": ["密码", "账号"],
    },
    {
        "id": 3,
        "category": "consultation",
        "question": "快递一般几天能到",
        "answer": "答案三",
        "keywords": ["物流", "时效"],
    },
    {
        "id": 4,
        "category": "consultation",
        "question": "支持哪些支付方式",
        "answer": "答案四",
        "keywords": ["付款", "微信"],
    },
]


def _build_sample_index():
    kb.build_index(SAMPLE_ENTRIES)


def test_tokenize_removes_stopwords_and_punctuation():
    tokens = kb.tokenize("请问，怎么申请退货？的了吗")
    assert "退货" in tokens
    assert "的" not in tokens and "吗" not in tokens
    assert all(token.strip() for token in tokens)
    assert "？" not in tokens and "，" not in tokens


def test_exact_question_scores_highest():
    _build_sample_index()
    results = kb.search("如何申请退货", 4)
    assert len(results) == 4
    best_entry, best_score = results[0]
    assert best_entry["id"] == 1
    # 查询词全部命中同一条知识，分数应显著领先其他条目
    assert best_score > 0.5


def test_unrelated_question_scores_low():
    _build_sample_index()
    results = kb.search("今天天气真好啊", 4)
    assert results[0][1] < 0.30


def test_empty_input_does_not_crash():
    _build_sample_index()
    assert kb.search("", 3) == []
    assert kb.search("的了吗呢", 3) == []


def test_results_sorted_descending():
    _build_sample_index()
    results = kb.search("我想退货怎么退", 4)
    scores = [score for _entry, score in results]
    assert scores == sorted(scores, reverse=True)
    # 与退货相关的条目应排在前面
    assert results[0][0]["id"] == 1


def test_top_n_limits_result_count():
    entries = [
        {"id": i, "question": f"问题{i}", "answer": "", "keywords": []}
        for i in range(5)
    ]
    kb.build_index(entries)
    assert len(kb.search("问题1", 3)) == 3
