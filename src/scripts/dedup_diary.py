"""
日记去重脚本：扫描 Qdrant 中所有 diary 记录，找出语义相似且时间接近的重复条目。

去重条件（同时满足）：
  1. 语义相似度 >= 阈值（默认 0.85）
  2. 时间间隔 <= 窗口（默认 15 分钟）

用法：
    # 预览模式（默认）—— 只看会删什么，不真删
    python scripts/dedup_diary.py

    # 真正执行删除
    python scripts/dedup_diary.py --apply

    # 自定义参数
    python scripts/dedup_diary.py --threshold 0.80 --window 10
"""

import argparse
from datetime import datetime

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Direction,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PointIdsList,
)

COLLECTION = "apts1548"


def fetch_all_diaries(client: QdrantClient) -> list:
    """分页拉取所有 diary 记录（时间正序）。"""
    diary_filter = Filter(must=[
        FieldCondition(key="record_type", match=MatchValue(value="diary")),
        FieldCondition(key="user_id", match=MatchValue(value="48")),
    ])

    all_points = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=diary_filter,
            limit=100,
            with_payload=True,
            with_vectors=True,
            order_by=OrderBy(key="timestamp", direction=Direction.ASC),
            offset=offset,
        )
        all_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset

    return all_points


def find_duplicates(
    points: list,
    threshold: float = 0.85,
    window_min: float = 15.0,
) -> list[tuple]:
    """找出所有重复对：语义相似 且 时间接近。

    按时间排序后，每条只和前面的比较。如果已标记为删除的，跳过不再作为基准。
    返回 [(keep_point, dup_point, similarity), ...]
    """
    if not points:
        return []

    # 提取向量
    vectors = np.array([np.array(p.vector) for p in points])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    window_sec = window_min * 60
    duplicates = []
    deleted = set()  # 已标记删除的 index

    for i in range(1, len(points)):
        if i in deleted:
            continue

        ts_i = points[i].payload.get("timestamp", 0)

        # 往前找时间窗口内的条目
        best_sim = 0.0
        best_j = -1

        for j in range(i - 1, -1, -1):
            ts_j = points[j].payload.get("timestamp", 0)

            # 超出时间窗口，不用再往前看了
            if ts_i - ts_j > window_sec:
                break

            if j in deleted:
                continue

            sim = float(np.dot(vectors[i], vectors[j]))
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                best_j = j

        if best_j >= 0:
            # i 是重复的，保留 j（更早的那条）
            deleted.add(i)
            duplicates.append((points[best_j], points[i], best_sim))

    return duplicates


def ts_to_str(ts: float) -> str:
    if ts <= 0:
        return "未知时间"
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def main():
    parser = argparse.ArgumentParser(description="日记语义去重（带时间窗口）")
    parser.add_argument("--apply", action="store_true", help="真正执行删除（默认只预览）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认直接删除")
    parser.add_argument("--threshold", type=float, default=0.85, help="相似度阈值（默认 0.85）")
    parser.add_argument("--window", type=float, default=15.0, help="时间窗口（分钟，默认 15）")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant 地址")
    args = parser.parse_args()

    print(f"连接 Qdrant: {args.qdrant_url}")
    client = QdrantClient(url=args.qdrant_url)

    print("拉取所有日记 ...")
    points = fetch_all_diaries(client)
    print(f"共 {len(points)} 条日记")
    print(f"去重条件：相似度 >= {args.threshold} 且 时间间隔 <= {args.window} 分钟\n")

    if not points:
        print("没有日记，退出")
        return

    duplicates = find_duplicates(points, args.threshold, args.window)

    if not duplicates:
        print("没有重复日记，退出")
        return

    print(f"找到 {len(duplicates)} 条重复\n")

    ids_to_delete = []
    for i, (keep, dup, sim) in enumerate(duplicates):
        kp, dp = keep.payload, dup.payload
        kt, dt_ = kp.get("timestamp", 0), dp.get("timestamp", 0)
        gap = abs(dt_ - kt) / 60

        print(f"── 重复 {i+1}（相似度 {sim:.3f}，间隔 {gap:.0f} 分钟）──")
        print(f"  ✓ 保留: [{ts_to_str(kt)}] {kp.get('message', '')[:70]}")
        print(f"  ✗ 删除: [{ts_to_str(dt_)}] {dp.get('message', '')[:70]}")
        print()
        ids_to_delete.append(dup.id)

    remaining = len(points) - len(ids_to_delete)
    print(f"总计：保留 {remaining} 条，删除 {len(ids_to_delete)} 条\n")

    if not args.apply:
        print("⚠ 预览模式，未执行删除。加 --apply 参数真正删除。")
        return

    if not args.yes:
        confirm = input(f"确认删除 {len(ids_to_delete)} 条重复日记？(y/N) ")
        if confirm.lower() != "y":
            print("取消")
            return

    print("删除中 ...")
    for batch_start in range(0, len(ids_to_delete), 100):
        batch = ids_to_delete[batch_start:batch_start + 100]
        client.delete(
            collection_name=COLLECTION,
            points_selector=PointIdsList(points=batch),
        )
        print(f"  已删除 {min(batch_start + 100, len(ids_to_delete))}/{len(ids_to_delete)}")

    print(f"\n完成，共删除 {len(ids_to_delete)} 条重复日记，剩余 {remaining} 条")


if __name__ == "__main__":
    main()
