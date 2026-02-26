"""
快速测试故事 RAG 检索效果。不需要启动 bot。

用法：
    .venv/bin/python scripts/test_story_rag.py
    .venv/bin/python scripts/test_story_rag.py "你怎么失去色觉的"
"""
import asyncio
import sys
import os

# 把 src/ 加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env（如果有）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.memory import Memory

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

TEST_QUERIES = [
    "数据塔事故是怎么回事",
    "你怎么失去色觉的",
    "你是怎么诞生的",
    "49 发射那天什么感觉",
    "一个人的教室",
    "今天天气怎么样",       # 应该 0 条
    "你吃饭了吗",          # 应该 0 条
]


async def main():
    queries = sys.argv[1:] if len(sys.argv) > 1 else TEST_QUERIES

    memory = Memory(qdrant_url=QDRANT_URL)
    print(f"Qdrant: {QDRANT_URL}\n")

    for q in queries:
        results = await memory.search_stories(query=q, limit=2)
        print(f"Q: {q}")
        print(f"   命中: {len(results)} 条")
        for i, r in enumerate(results):
            print(f"   [{i}]")
            print(r.message)
            print()
        print()


if __name__ == "__main__":
    asyncio.run(main())
