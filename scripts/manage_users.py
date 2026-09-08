#!/usr/bin/env python3
"""Manage WevisionCut accounts in ``users.yaml``.

Passwords are read from a prompt (never from argv, so they stay out of shell
history and out of ``ps``) and only their PBKDF2 hash is written to disk.

    python3 scripts/manage_users.py init                    # create the file with an admin
    python3 scripts/manage_users.py add   alice              # add a normal user
    python3 scripts/manage_users.py add   bob --admin        # add another admin
    python3 scripts/manage_users.py add   dan --no-password  # passwordless deployment
    python3 scripts/manage_users.py role  alice admin        # grant/revoke admin
    python3 scripts/manage_users.py passwd alice             # change a password
    python3 scripts/manage_users.py list
    python3 scripts/manage_users.py remove alice

Point it at a different file with ``--file /path/to/users.yaml``.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from studio.auth import ADMIN_ROLE, hash_password  # noqa: E402

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "users.yaml"
MIN_PASSWORD_LEN = 8


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"users": []}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("users"), list):
        raise SystemExit(f"{path} 格式不对：顶层需要一个 users 列表")
    return payload


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# WevisionCut 账号文件。只存 PBKDF2 哈希，不存明文密码。\n"
        "# 用 scripts/manage_users.py 维护，不要手写 password_hash。\n"
    )
    path.write_text(
        header + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    try:
        path.chmod(0o600)      # hashes are not secrets, but need not be world-readable
    except OSError:
        pass
    print(f"已写入 {path}")


def _find(payload: dict, username: str) -> dict | None:
    for item in payload["users"]:
        if isinstance(item, dict) and item.get("username") == username:
            return item
    return None


def _ask_password(username: str) -> str:
    while True:
        pw = getpass.getpass(f"为 {username} 设置密码: ")
        if len(pw) < MIN_PASSWORD_LEN:
            print(f"  密码至少 {MIN_PASSWORD_LEN} 位，请重试")
            continue
        if pw != getpass.getpass("再输入一次确认: "):
            print("  两次输入不一致，请重试")
            continue
        return pw


def cmd_init(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} 已存在。要重建请加 --force，或用 add 追加账号。")
    username = args.username or "admin"
    payload = {"users": [{
        "username": username,
        "role": ADMIN_ROLE,
        "display_name": args.display_name or "超级管理员",
        "password_hash": hash_password(_ask_password(username)),
    }]}
    _save(path, payload)
    print(f"超级管理员 {username} 已创建，可以看到所有人的作品。")


def cmd_add(args: argparse.Namespace) -> None:
    path = Path(args.file)
    payload = _load(path)
    if _find(payload, args.username):
        raise SystemExit(f"账号 {args.username} 已存在，改密码请用 passwd。")
    entry = {
        "username": args.username,
        "role": ADMIN_ROLE if args.admin else "user",
        "display_name": args.display_name or args.username,
    }
    # Passwordless deployments identify people by username alone, so forcing a
    # password here would create one that can never be used.
    if not args.no_password:
        entry["password_hash"] = hash_password(_ask_password(args.username))
    payload["users"].append(entry)
    _save(path, payload)


def cmd_role(args: argparse.Namespace) -> None:
    path = Path(args.file)
    payload = _load(path)
    entry = _find(payload, args.username)
    if entry is None:
        raise SystemExit(f"没有账号 {args.username}")
    entry["role"] = ADMIN_ROLE if args.role == "admin" else "user"
    if not any(u.get("role") == ADMIN_ROLE for u in payload["users"]):
        raise SystemExit("拒绝修改：这样就没有超级管理员了。先给别人开权限再降级。")
    _save(path, payload)
    print(f"{args.username} 现在是 {entry['role']}。重启服务后生效。")


def cmd_passwd(args: argparse.Namespace) -> None:
    path = Path(args.file)
    payload = _load(path)
    entry = _find(payload, args.username)
    if entry is None:
        raise SystemExit(f"没有账号 {args.username}")
    entry["password_hash"] = hash_password(_ask_password(args.username))
    _save(path, payload)
    print("改完后该用户的现有会话会在服务重启后失效。")


def cmd_remove(args: argparse.Namespace) -> None:
    path = Path(args.file)
    payload = _load(path)
    before = len(payload["users"])
    payload["users"] = [
        u for u in payload["users"]
        if not (isinstance(u, dict) and u.get("username") == args.username)
    ]
    if len(payload["users"]) == before:
        raise SystemExit(f"没有账号 {args.username}")
    if not any(u.get("role") == ADMIN_ROLE for u in payload["users"]):
        raise SystemExit("拒绝删除：这样就没有超级管理员了。先加一个再删。")
    _save(path, payload)
    print(f"已删除 {args.username}。他名下的作品仍在磁盘上，只有超管能看到。")


def cmd_list(args: argparse.Namespace) -> None:
    payload = _load(Path(args.file))
    if not payload["users"]:
        print("（没有账号）")
        return
    print(f"{'账号':<18}{'角色':<8}显示名")
    for u in payload["users"]:
        if isinstance(u, dict):
            print(f"{u.get('username',''):<18}{u.get('role','user'):<8}"
                  f"{u.get('display_name','')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="WevisionCut 账号管理")
    ap.add_argument("--file", default=str(DEFAULT_FILE), help="users.yaml 路径")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="新建账号文件并创建超级管理员")
    p.add_argument("username", nargs="?", default="admin")
    p.add_argument("--display-name", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("add", help="添加账号")
    p.add_argument("username")
    p.add_argument("--admin", action="store_true", help="设为超级管理员")
    p.add_argument("--no-password", action="store_true",
                   help="不设密码（免密模式下用这个）")
    p.add_argument("--display-name", default="")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("role", help="修改角色（普通用户 <-> 超级管理员）")
    p.add_argument("username")
    p.add_argument("role", choices=["admin", "user"])
    p.set_defaults(func=cmd_role)

    p = sub.add_parser("passwd", help="修改密码")
    p.add_argument("username")
    p.set_defaults(func=cmd_passwd)

    p = sub.add_parser("remove", help="删除账号")
    p.add_argument("username")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("list", help="列出账号")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
