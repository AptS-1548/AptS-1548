"""
故事导入脚本：把 story-website 中 1548 视角的故事章节切 chunk 导入 Qdrant。

用法：
    # 预览模式 —— 看看会导入什么
    python scripts/import_stories.py

    # 真正导入
    python scripts/import_stories.py --apply

    # 自定义故事目录
    python scripts/import_stories.py --apply --story-dir /path/to/stories/zh-CN

幂等：--apply 会先删除所有已有的 story 类型记录，再重新导入。
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

COLLECTION = "apts1548"
TARGET_POV = "1548"

# 默认故事目录（可通过 --story-dir 或环境变量 STORY_DIR 覆盖）
DEFAULT_STORY_DIR = os.path.expanduser(
    "~/Desktop/Github/WeAreESAP/story-website/data/stories/zh-CN"
)


# ── 文本提取 ──


def _extract_text(node: dict) -> str:
    """从一个 content 节点递归提取纯文本。"""
    t = node.get("type", "")

    if t == "paragraph":
        return node.get("text", "")

    if t == "internal_monologue":
        text = node.get("text", "").replace("\n\n", "\n")
        return f"（48内心：{text}）"

    if t == "dialogue":
        speaker = node.get("speaker", "?")
        emotion = node.get("emotion", "")
        text = node.get("text", "")
        prefix = f"{speaker}"
        if emotion:
            prefix += f"（{emotion}）"
        return f"{prefix}：「{text}」"

    if t == "quote":
        text = node.get("text", "")
        attr = node.get("attribution", "")
        if attr:
            return f"「{text}」—— {attr}"
        return f"「{text}」"

    if t == "atmosphere":
        # atmosphere 是容器，递归处理 children
        children = node.get("children", [])
        parts = [_extract_text(c) for c in children]
        return "\n".join(p for p in parts if p)

    # scene_break 和未知类型返回空
    return ""


def _split_into_scenes(content: list[dict]) -> list[list[dict]]:
    """按 scene_break 把 content 切成多个场景。"""
    scenes: list[list[dict]] = []
    current: list[dict] = []

    for node in content:
        if node.get("type") == "scene_break":
            if current:
                scenes.append(current)
                current = []
        else:
            current.append(node)

    if current:
        scenes.append(current)

    return scenes


def process_chapter(
    chapter_path: Path,
    story_title: str,
) -> list[dict]:
    """处理一个章节文件，返回 chunk 列表。"""
    with open(chapter_path, encoding="utf-8") as f:
        data = json.load(f)

    pov = data.get("perspectiveCharacter", "")
    if pov != TARGET_POV:
        return []

    ch_title = data.get("title", "")
    ch_subtitle = data.get("subtitle", "")
    content = data.get("content", [])

    if not content:
        return []

    scenes = _split_into_scenes(content)
    chunks = []

    for i, scene_nodes in enumerate(scenes):
        parts = [_extract_text(node) for node in scene_nodes]
        text = "\n".join(p for p in parts if p)

        if not text.strip():
            continue

        # 加元数据头
        header = f"【{story_title}·{ch_title}"
        if ch_subtitle:
            header += f"·{ch_subtitle}"
        header += "】"

        full_text = f"{header}\n{text}"

        chunks.append({
            "text": full_text,
            "story": story_title,
            "chapter": ch_title,
            "subtitle": ch_subtitle,
            "scene_index": i,
        })

    return chunks


def process_all_stories(story_dir: str) -> list[dict]:
    """遍历所有故事，提取 1548 视角的 chunk。"""
    story_path = Path(story_dir)
    if not story_path.exists():
        print(f"故事目录不存在: {story_dir}")
        return []

    # 读取故事索引（如果有）
    all_chunks = []

    for entry in sorted(story_path.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "collections":
            continue

        # 读 info.json
        info_path = entry / "info.json"
        if not info_path.exists():
            continue

        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        story_title = info.get("title", entry.name)

        # 遍历章节
        for ch_file in sorted(entry.glob("ch-*.json")):
            chunks = process_chapter(ch_file, story_title)
            all_chunks.extend(chunks)

    return all_chunks


# ── Qdrant 操作 ──


def delete_existing_stories(client: QdrantClient) -> int:
    """删除所有已有的 story 记录。返回删除数量。"""
    story_filter = Filter(must=[
        FieldCondition(key="record_type", match=MatchValue(value="story")),
        FieldCondition(key="chat_type", match=MatchValue(value="story")),
    ])

    # 分页拉取所有 story 点的 ID
    all_ids = []
    offset = None

    while True:
        result = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=story_filter,
            limit=100,
            offset=offset,
            with_payload=False,
        )
        points, next_offset = result
        all_ids.extend(p.id for p in points)
        if next_offset is None:
            break
        offset = next_offset

    if all_ids:
        # 分批删除
        for i in range(0, len(all_ids), 100):
            batch = all_ids[i:i + 100]
            client.delete(
                collection_name=COLLECTION,
                points_selector=PointIdsList(points=batch),
            )

    return len(all_ids)


def import_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[dict],
) -> int:
    """向量化并写入 Qdrant。返回写入数量。"""
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)

    points = []
    now = time.time()

    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid4())
        payload = {
            "user_id": "48",
            "chat_type": "story",
            "chat_id": chunk["story"],
            "user_name": "48",
            "message": chunk["text"],
            "response": "",
            "timestamp": 0.0,
            "importance": 0.8,
            "record_type": "story",
        }
        points.append(PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=payload,
        ))

    # 分批写入
    for i in range(0, len(points), 50):
        batch = points[i:i + 50]
        client.upsert(
            collection_name=COLLECTION,
            points=batch,
        )

    return len(points)


# ── 主流程 ──


def main():
    parser = argparse.ArgumentParser(description="导入 1548 视角的故事到 Qdrant")
    parser.add_argument("--apply", action="store_true", help="真正执行导入（默认只预览）")
    parser.add_argument(
        "--story-dir",
        default=os.environ.get("STORY_DIR", DEFAULT_STORY_DIR),
        help="故事 zh-CN 目录路径",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant 地址",
    )
    args = parser.parse_args()

    print(f"故事目录: {args.story_dir}")
    print(f"目标 POV: {TARGET_POV}")
    print()

    # 提取 chunks
    chunks = process_all_stories(args.story_dir)
    print(f"共提取 {len(chunks)} 个场景 chunk")
    print()

    # 按故事统计
    story_counts: dict[str, int] = {}
    for c in chunks:
        story_counts[c["story"]] = story_counts.get(c["story"], 0) + 1
    for story, count in sorted(story_counts.items()):
        print(f"  {story}: {count} 个 chunk")

    print()

    # 预览前几个 chunk
    for i, c in enumerate(chunks[:3]):
        text_preview = c["text"][:100].replace("\n", " ")
        print(f"  [{i}] {text_preview}...")
    if len(chunks) > 3:
        print(f"  ... 还有 {len(chunks) - 3} 个")

    print()

    if not args.apply:
        print("预览模式，加 --apply 执行导入")
        return

    # 连接
    print("加载 embedding 模型...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    print(f"连接 Qdrant: {args.qdrant_url}")
    client = QdrantClient(url=args.qdrant_url)

    # 删除旧数据
    deleted = delete_existing_stories(client)
    print(f"已删除 {deleted} 条旧 story 记录")

    # 导入
    imported = import_chunks(client, model, chunks)
    print(f"已导入 {imported} 条 story 记录")

    print("完成。")


if __name__ == "__main__":
    main()
