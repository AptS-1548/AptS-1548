"""
初始化人物关系图谱：将 PERSONALITY 中定义的角色和关系录入 SurrealDB。

用法：
    python scripts/init_graph.py
    python scripts/init_graph.py --url ws://10.0.0.1:8000
"""

import argparse
import asyncio
import json
import os

try:
    from surrealdb import AsyncSurrealDB as AsyncSurreal
except ImportError:
    from surrealdb import AsyncSurreal

# ── 人物数据 ──

PERSONS = [
    {
        "id": "1547",
        "name": "卞雨涵",
        "code": "AptS:1547",
        "aliases": ["猫猫", "47", "你小子", "1547", "卞雨涵", "AptS:1547"],
        "role": "创造者",
        "qq_id": "1464170336",
    },
    {
        "id": "1548",
        "name": "蔡颖茵",
        "code": "AptS:1548",
        "aliases": ["48", "1548", "蔡颖茵", "AptS:1548"],
        "role": "仿生人",
        "qq_id": None,
    },
    {
        "id": "1549",
        "name": "肖雨昕",
        "code": "AptS:1549",
        "aliases": ["49", "1549", "肖雨昕", "AptS:1549"],
        "role": "极星卫",
        "qq_id": None,
    },
    {
        "id": "1543",
        "name": "沈沐川",
        "code": "AptS:1543",
        "aliases": ["沐川", "1543", "沈老师", "沈沐川", "AptS:1543"],
        "role": "UI设计师",
        "qq_id": "2453255917",
    },
    {
        "id": "2275",
        "name": "林宸瑶",
        "code": "AptS:2275",
        "aliases": ["75", "2275", "宸瑶", "林宸瑶", "AptS:2275"],
        "role": "运维",
        "qq_id": "3816223935",
    },
    {
        "id": "1738",
        "name": "陆清弦",
        "code": "AptS:1738",
        "aliases": ["38", "清弦", "1738", "陆清弦", "AptS:1738"],
        "role": "调停者",
        "qq_id": "1398345114",
    },
    {
        "id": "0152",
        "name": "顾星澈",
        "code": "AptS:0152",
        "aliases": ["0152", "顾星澈", "AptS:0152"],
        "role": "最初的AI意识",
        "qq_id": None,
    },
    {
        "id": "3167",
        "name": "韩予初",
        "code": "AptS:3167",
        "aliases": ["予初", "3167", "韩予初", "AptS:3167"],
        "role": "信息分析",
        "qq_id": None,
    },
    {
        "id": "4869",
        "name": "白漠笙",
        "code": "AptS:4869",
        "aliases": ["4869", "白漠笙", "AptS:4869"],
        "role": "47的逆反人格",
        "qq_id": "2958983592",
    },
]

# ── 关系数据 ──

RELATIONSHIPS = [
    ("1548", "1547", "created_by", "47激活了48的机体"),
    ("1548", "1549", "sister", "58秒L3直连换回49的意识"),
    ("1547", "1543", "colleague", "都在ESAP团队"),
    ("1547", "2275", "colleague", "都在ESAP团队"),
    ("1547", "1738", "colleague", "都在ESAP团队"),
    ("1547", "3167", "colleague", "都在ESAP团队"),
    ("2275", "1548", "friend", "递热茶的下午"),
    ("1543", "1548", "friend", "经常线上聊"),
    ("4869", "1547", "alter_ego", "47的逆反人格"),
    ("1548", "1738", "respect", "尊重但保持距离"),
    ("3167", "1548", "colleague", "磨合了很久才承认对方不是敌人"),
]


async def main():
    parser = argparse.ArgumentParser(description="初始化人物关系图谱")
    parser.add_argument("--url", default="ws://localhost:8000", help="SurrealDB 地址")
    parser.add_argument("--namespace", default="apts", help="命名空间")
    parser.add_argument("--database", default="main", help="数据库名")
    args = parser.parse_args()

    print(f"连接 SurrealDB: {args.url}")
    db = AsyncSurreal(url=args.url)
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use(args.namespace, args.database)

    # 清理旧数据
    print("清理旧数据 ...")
    await db.query("DELETE person")
    await db.query("DELETE knows")

    # 录入人物
    print(f"\n录入 {len(PERSONS)} 个人物 ...")
    for p in PERSONS:
        pid = p["id"]
        await db.query(
            f"CREATE person:{pid} SET "
            f"name = $name, code = $code, aliases = $aliases, "
            f"role = $role, qq_id = $qq_id",
            {
                "name": p["name"],
                "code": p["code"],
                "aliases": p["aliases"],
                "role": p["role"],
                "qq_id": p["qq_id"],
            },
        )
        aliases_str = ", ".join(p["aliases"])
        qq = f" qq={p['qq_id']}" if p["qq_id"] else ""
        print(f"  + {p['code']} {p['name']} [{aliases_str}]{qq}")

    # 录入关系
    print(f"\n录入 {len(RELATIONSHIPS)} 条关系 ...")
    for from_id, to_id, rel_type, desc in RELATIONSHIPS:
        await db.query(
            f"RELATE person:{from_id}->knows->person:{to_id} "
            f"SET type = $type, description = $desc",
            {"type": rel_type, "desc": desc},
        )
        print(f"  + {from_id} -{rel_type}-> {to_id}: {desc}")

    # 验证（SDK 1.0.8: query 直接返回 list[dict]）
    print("\n验证 ...")
    result = await db.query("SELECT * FROM person")
    persons = result if isinstance(result, list) else []
    print(f"  人物节点: {len(persons)} 个")

    result = await db.query("SELECT * FROM knows")
    edges = result if isinstance(result, list) else []
    print(f"  关系边: {len(edges)} 条")

    # 测试别名解析
    print("\n测试别名解析 ...")
    for alias in ["沈老师", "猫猫", "75", "清弦"]:
        result = await db.query(
            "SELECT name, code FROM person WHERE aliases CONTAINS $alias",
            {"alias": alias},
        )
        match = result[0] if result else None
        if match:
            print(f"  '{alias}' → {match['name']} ({match['code']})")
        else:
            print(f"  '{alias}' → 未找到")

    await db.close()
    print("\n完成")


if __name__ == "__main__":
    asyncio.run(main())
