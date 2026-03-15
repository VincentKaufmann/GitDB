"""GitDB CLI — git-like commands for your vector database.

Usage:
    gitdb init my_store --dim 1024
    gitdb add vectors.pt --documents docs.json
    gitdb commit -m "Initial ingest"
    gitdb log
    gitdb query --vector query.pt --k 10
    gitdb branch experiment
    gitdb switch experiment
    gitdb merge experiment
    gitdb diff main experiment
    gitdb blame
    gitdb stash
    gitdb stash pop
    gitdb purge --where '{"author": "claude"}' --reason "cleanup"
    gitdb reflog
    gitdb status
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def find_store(start_path: str = ".") -> str:
    """Walk up from start_path to find a directory containing .gitdb/"""
    p = Path(start_path).resolve()
    while p != p.parent:
        if (p / ".gitdb").exists():
            return str(p)
        p = p.parent
    return None


def load_db(args=None):
    """Load GitDB from current directory or --path."""
    from gitdb import GitDB
    path = getattr(args, "path", None) or find_store()
    if path is None:
        print("fatal: not a gitdb repository (or any parent)", file=sys.stderr)
        sys.exit(1)
    cfg_path = Path(path) / ".gitdb" / "config"
    cfg = json.loads(cfg_path.read_text())
    device = getattr(args, "device", None) or cfg.get("device", "cpu")
    return GitDB(path, dim=cfg["dim"], device=device)


def cmd_init(args):
    from gitdb import GitDB
    db = GitDB(args.name, dim=args.dim, device=args.device)
    print(f"Initialized empty GitDB in {Path(args.name).resolve()}/.gitdb/")
    print(f"  dim={args.dim}, device={args.device}")


def cmd_status(args):
    db = load_db(args)
    s = db.status()
    print(f"On branch {s['branch']}")
    head = s["head"]
    if head:
        print(f"HEAD: {head[:12]}...")
    else:
        print("No commits yet")

    # Vector info
    if s['active_rows'] > 0:
        print(f"\n  {s['active_rows']} vectors ({s['total_rows']} total)")
    # Document info
    if s.get('documents', 0) > 0:
        print(f"  {s['documents']} documents")

    staged = s["staged_additions"] + s["staged_deletions"] + s["staged_modifications"]
    doc_staged = s.get("staged_doc_inserts", 0) + s.get("staged_doc_updates", 0) + s.get("staged_doc_deletes", 0)
    if staged or doc_staged:
        print(f"\nChanges to be committed:")
        if s["staged_additions"]:
            print(f"  new:      {s['staged_additions']} vectors")
        if s["staged_deletions"]:
            print(f"  deleted:  {s['staged_deletions']} vectors")
        if s["staged_modifications"]:
            print(f"  modified: {s['staged_modifications']} vectors")
        if s.get("staged_doc_inserts"):
            print(f"  new:      {s['staged_doc_inserts']} documents")
        if s.get("staged_doc_updates"):
            print(f"  modified: {s['staged_doc_updates']} documents")
        if s.get("staged_doc_deletes"):
            print(f"  deleted:  {s['staged_doc_deletes']} documents")
    else:
        print("\nnothing to commit, working tree clean")


def cmd_add(args):
    db = load_db(args)

    # Text-based embedding mode
    if args.text:
        texts = args.text if isinstance(args.text, list) else [args.text]
        model = getattr(args, "embed_model", None)
        indices = db.add(texts=texts, embed_model=model)
        print(f"Embedded and added {len(indices)} vectors ({db.dim}d)")
        if args.autocommit:
            h = db.commit(f"Add {len(indices)} embedded texts")
            print(f"Committed: {h[:12]}...")
        return

    if args.text_file:
        text_path = Path(args.text_file)
        texts = [line.strip() for line in text_path.read_text().splitlines() if line.strip()]
        model = getattr(args, "embed_model", None)
        indices = db.add(texts=texts, embed_model=model)
        print(f"Embedded and added {len(indices)} vectors from {text_path.name}")
        if args.autocommit:
            h = db.commit(f"Add {len(indices)} embedded texts from {text_path.name}")
            print(f"Committed: {h[:12]}...")
        return

    # Load tensor from file
    tensor_path = Path(args.file) if args.file else None
    if tensor_path is None:
        print("fatal: must specify FILE, --text, or --text-file", file=sys.stderr)
        sys.exit(1)

    if tensor_path.suffix == ".pt":
        embeddings = torch.load(tensor_path, weights_only=True)
    elif tensor_path.suffix == ".npy":
        import numpy as np
        embeddings = torch.from_numpy(np.load(tensor_path))
    elif tensor_path.suffix in (".json", ".jsonl"):
        # JSONL: each line has {"embedding": [...], "document": "...", "metadata": {...}}
        docs = []
        metas = []
        embs = []
        with open(tensor_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                embs.append(obj["embedding"])
                docs.append(obj.get("document"))
                metas.append(obj.get("metadata", {}))
        embeddings = torch.tensor(embs, dtype=torch.float32)
        indices = db.add(embeddings, documents=docs, metadata=metas)
        print(f"Added {len(indices)} vectors from {tensor_path.name}")
        if args.autocommit:
            h = db.commit(f"Add {len(indices)} vectors from {tensor_path.name}")
            print(f"Committed: {h[:12]}...")
        return
    else:
        print(f"Unsupported file format: {tensor_path.suffix}", file=sys.stderr)
        print("Supported: .pt, .npy, .json, .jsonl", file=sys.stderr)
        sys.exit(1)

    # Load optional documents
    docs = None
    if args.documents:
        docs_path = Path(args.documents)
        docs = json.loads(docs_path.read_text())

    # Load optional metadata
    metas = None
    if args.metadata:
        meta_path = Path(args.metadata)
        metas = json.loads(meta_path.read_text())

    indices = db.add(embeddings, documents=docs, metadata=metas)
    print(f"Added {len(indices)} vectors ({embeddings.shape[1]}d)")

    if args.autocommit:
        h = db.commit(f"Add {len(indices)} vectors from {tensor_path.name}")
        print(f"Committed: {h[:12]}...")


def cmd_commit(args):
    db = load_db(args)
    try:
        h = db.commit(args.message)
        s = db.status()
        print(f"[{s['branch']} {h[:8]}] {args.message}")
        log = db.log(limit=1)
        if log:
            stats = log[0].stats
            print(f" {stats}")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_log(args):
    db = load_db(args)
    entries = db.log(limit=args.limit, source=args.source)
    if not entries:
        print("No commits yet")
        return
    for c in entries:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.timestamp))
        print(f"\033[33m{c.hash[:8]}\033[0m {c.message}")
        print(f"  {ts}  {c.stats}")
        print()


def cmd_diff(args):
    db = load_db(args)
    try:
        d = db.diff(args.ref_a, args.ref_b)
        if not (d.added_count or d.removed_count or d.modified_count):
            print("No differences")
            return
        # Print unified diff with colors
        unified = d.unified(args.ref_a, args.ref_b)
        for line in unified.splitlines():
            if line.startswith("diff --gitdb"):
                print(f"\033[1m{line}\033[0m")  # bold
            elif line.startswith("---"):
                print(f"\033[1;31m{line}\033[0m")
            elif line.startswith("+++"):
                print(f"\033[1;32m{line}\033[0m")
            elif line.startswith("@@"):
                print(f"\033[36m{line}\033[0m")  # cyan
            elif line.startswith("-"):
                print(f"\033[31m{line}\033[0m")  # red
            elif line.startswith("+"):
                print(f"\033[32m{line}\033[0m")  # green
            elif line.startswith("new vector") or line.startswith("deleted vector") or line.startswith("modified vector"):
                print(f"\033[33m{line}\033[0m")  # yellow
            else:
                print(line)
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_branch(args):
    db = load_db(args)
    if args.name:
        try:
            db.branch(args.name)
            print(f"Created branch '{args.name}'")
        except ValueError as e:
            print(f"fatal: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        branches = db.branches()
        current = db.refs.current_branch
        for name, h in sorted(branches.items()):
            marker = "* " if name == current else "  "
            print(f"{marker}\033[32m{name}\033[0m  {h[:8]}")


def cmd_switch(args):
    db = load_db(args)
    try:
        db.switch(args.branch)
        print(f"Switched to branch '{args.branch}'")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_checkout(args):
    db = load_db(args)
    try:
        db.checkout(args.ref)
        print(f"HEAD now at {args.ref}")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_merge(args):
    db = load_db(args)
    try:
        result = db.merge(args.branch, strategy=args.strategy)
        print(f"Merge commit {result.commit_hash[:12]} ({result.strategy})")
        if result.added:
            print(f"  \033[32m+{result.added} vectors added\033[0m")
        if result.removed:
            print(f"  \033[31m-{result.removed} vectors removed\033[0m")
        if result.has_conflicts:
            print(f"  \033[31m{len(result.conflicts)} conflicts: {', '.join(c[:8] for c in result.conflicts)}\033[0m")
        if result.diff and result.diff.entries:
            print()
            unified = result.diff.unified("ours", args.branch)
            for line in unified.splitlines():
                if line.startswith("diff --gitdb"):
                    print(f"\033[1m{line}\033[0m")
                elif line.startswith("---"):
                    print(f"\033[1;31m{line}\033[0m")
                elif line.startswith("+++"):
                    print(f"\033[1;32m{line}\033[0m")
                elif line.startswith("@@"):
                    print(f"\033[36m{line}\033[0m")
                elif line.startswith("-"):
                    print(f"\033[31m{line}\033[0m")
                elif line.startswith("+"):
                    print(f"\033[32m{line}\033[0m")
                else:
                    print(line)
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_cherry_pick(args):
    db = load_db(args)
    try:
        source_info = db.show(args.ref)
        h = db.cherry_pick(args.ref)
        branch = db.refs.current_branch
        print(f"\033[33m[{branch} {h[:8]}]\033[0m {source_info['message']}")
        s = source_info['stats']
        parts = []
        if s['added']: parts.append(f"{s['added']} insertion(+)")
        if s['removed']: parts.append(f"{s['removed']} deletion(-)")
        if s['modified']: parts.append(f"{s['modified']} modification(~)")
        if parts:
            print(f" {', '.join(parts)}")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_revert(args):
    db = load_db(args)
    try:
        source_info = db.show(args.ref)
        h = db.revert(args.ref)
        branch = db.refs.current_branch
        print(f"\033[33m[{branch} {h[:8]}]\033[0m Revert \"{source_info['message']}\"")
        print(f"This reverts commit {args.ref}.")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_stash(args):
    db = load_db(args)
    if args.action == "pop":
        try:
            entry = db.stash_pop()
            branch = db.refs.current_branch
            print(f"On branch {branch}")
            print(f"Dropped stash@{{{entry.index}}} ({entry.message})")
        except (ValueError, IndexError) as e:
            print(f"fatal: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "list":
        entries = db.stash_list()
        if not entries:
            print("No stashes")
        for e in entries:
            print(f"stash@{{{e.index}}}: On {db.refs.current_branch}: {e.message}")
    else:
        try:
            branch = db.refs.current_branch
            head = db.refs.head
            head_short = head[:7] if head else "0000000"
            # Get last commit message for display
            msg = args.message or "WIP"
            try:
                info = db.show("HEAD")
                head_msg = info.get("message", "")
            except Exception:
                head_msg = ""
            db.stash(msg)
            stashes = db.stash_list()
            idx = stashes[-1].index if stashes else 0
            print(f"Saved working directory and index state On {branch}: {msg}")
            print(f"HEAD is now at {head_short} {head_msg}")
        except ValueError as e:
            print(f"fatal: {e}", file=sys.stderr)
            sys.exit(1)


def cmd_blame(args):
    db = load_db(args)
    entries = db.blame()
    if not entries:
        print("No vectors to blame")
        return
    for b in entries:
        ts = time.strftime("%Y-%m-%d", time.localtime(b.timestamp))
        print(f"\033[33m{b.commit_hash[:8]}\033[0m  {ts}  row {b.row_index:>4d}  {b.vector_id[:12]}  {b.commit_message}")


def cmd_reflog(args):
    db = load_db(args)
    entries = db.reflog(limit=args.limit)
    if not entries:
        print("Reflog is empty")
        return
    for i, e in enumerate(entries):
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e["timestamp"]))
        commit = e.get("commit", "?")[:8]
        print(f"  {commit}  {ts}  {e['action']}")


def cmd_purge(args):
    db = load_db(args)
    where = json.loads(args.where) if args.where else None
    ids = args.ids.split(",") if args.ids else None
    if not where and not ids:
        print("fatal: must specify --where or --ids", file=sys.stderr)
        sys.exit(1)
    result = db.purge(ids=ids, where=where, reason=args.reason or "")
    print(f"Purged {result['vectors_purged']} vectors from {result['commits_rewritten']} commits")
    if result["vectors_purged"]:
        print(f"  Old HEAD: {result['old_head'][:12]}...")
        print(f"  New HEAD: {result['new_head'][:12]}...")


def cmd_query(args):
    db = load_db(args)

    # Load query vector
    if args.text:
        results = db.query_text(
            args.text, k=args.k,
            where=json.loads(args.where) if args.where else None,
            at=args.at,
            embed_model=getattr(args, "embed_model", None),
        )
    elif args.vector:
        v = torch.load(args.vector, weights_only=True)
        where = json.loads(args.where) if args.where else None
        results = db.query(v, k=args.k, where=where, at=args.at)
    elif args.random:
        v = torch.randn(db.dim)
        print(f"(using random query vector)")
        where = json.loads(args.where) if args.where else None
        results = db.query(v, k=args.k, where=where, at=args.at)
    else:
        print("fatal: must specify --text, --vector FILE, or --random", file=sys.stderr)
        sys.exit(1)

    if not results.ids:
        print("No results")
        return

    for i in range(len(results)):
        score = f"{results.scores[i]:.4f}"
        vid = results.ids[i][:12]
        doc = results.documents[i] or ""
        if len(doc) > 60:
            doc = doc[:57] + "..."
        meta = json.dumps(results.metadata[i]) if results.metadata[i] else ""
        print(f"  {score}  {vid}  {doc}  {meta}")


def cmd_tag(args):
    db = load_db(args)
    try:
        db.tag(args.name, ref=args.ref or "HEAD")
        print(f"Tagged '{args.name}'")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_reset(args):
    db = load_db(args)
    db.reset(args.ref or "HEAD")
    print(f"Reset to {args.ref or 'HEAD'}")


def cmd_gc(args):
    db = load_db(args)
    db.gc(keep_last=args.keep)
    print(f"GC complete (checkpoint interval: {args.keep})")


def cmd_rebase(args):
    db = load_db(args)
    try:
        new = db.rebase(args.onto)
        for h in new:
            try:
                info = db.show(h)
                print(f"Applying: {info['message']}")
            except Exception:
                print(f"Applying: {h[:8]}")
        print(f"Successfully rebased and updated refs/heads/{db.refs.current_branch}.")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_filter_branch(args):
    import torch.nn.functional as F
    db = load_db(args)

    transforms = {
        "normalize": lambda t: F.normalize(t, dim=1),
        "scale2x": lambda t: t * 2,
        "zero-mean": lambda t: t - t.mean(dim=0),
    }

    if args.transform not in transforms:
        print(f"Unknown transform: {args.transform}", file=sys.stderr)
        print(f"Available: {', '.join(transforms.keys())}", file=sys.stderr)
        sys.exit(1)

    h = db.filter_branch(transforms[args.transform], args.transform)
    branch = db.refs.current_branch
    print(f"Rewrite {h[:12]} ({branch})")
    print(f"Ref 'refs/heads/{branch}' was rewritten")


def cmd_show(args):
    db = load_db(args)
    try:
        info = db.show(args.ref or "HEAD")
        print(f"\033[33mcommit {info['hash']}\033[0m")
        if info["parent"]:
            print(f"Parent: {info['parent'][:12]}...")
        if info["parent2"]:
            print(f"Merge:  {info['parent2'][:12]}...")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["timestamp"]))
        print(f"Date:   {ts}")
        print(f"\n    {info['message']}\n")
        print(f"  +{info['stats']['added']} -{info['stats']['removed']} ~{info['stats']['modified']}")
        print(f"  {info['tensor_rows']} rows | delta {info['delta_size_bytes']} bytes")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_amend(args):
    db = load_db(args)
    try:
        h = db.amend(message=args.message)
        info = db.show(h)
        branch = db.refs.current_branch
        print(f"\033[33m[{branch} {h[:8]}]\033[0m {info['message']}")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_squash(args):
    db = load_db(args)
    try:
        h = db.squash(args.n, message=args.message)
        info = db.show(h)
        branch = db.refs.current_branch
        print(f"\033[33m[{branch} {h[:8]}]\033[0m {info['message']}")
        print(f" Squashed {args.n} commits into 1.")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_fork(args):
    db = load_db(args)
    try:
        forked = db.fork(args.dest, branch=args.branch)
        print(f"Cloning into '{Path(args.dest).resolve()}'...")
        print(f"done.")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_note(args):
    db = load_db(args)
    if args.message:
        db.note(args.ref or "HEAD", args.message)
        print(f"Note added to {(args.ref or 'HEAD')}")
    else:
        notes = db.notes(args.ref or "HEAD")
        if not notes:
            print("No notes")
        for n in notes:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(n["timestamp"]))
            print(f"  {ts}  {n['message']}")


def cmd_delete_branch(args):
    db = load_db(args)
    try:
        db.delete_branch(args.name)
        print(f"Deleted branch '{args.name}'")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_rename_branch(args):
    db = load_db(args)
    try:
        db.rename_branch(args.old, args.new)
        print(f"Renamed '{args.old}' → '{args.new}'")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_head(args):
    db = load_db(args)
    head = db.HEAD
    if head:
        print(f"{head}")
        if not args.quiet:
            info = db.show()
            print(f"  branch: {db.current_branch}")
            print(f"  {info['message']}")
            print(f"  {info['tensor_rows']} vectors")
    else:
        print("No commits yet")


def cmd_remote(args):
    db = load_db(args)
    if args.action == "add":
        if not args.name or not args.url:
            print("Usage: gitdb remote add <name> <url>", file=sys.stderr)
            sys.exit(1)
        db.remote_add(args.name, args.url)
        print(f"Remote '{args.name}' → {args.url}")
    elif args.action == "remove":
        db.remote_remove(args.name)
        print(f"Removed remote '{args.name}'")
    else:
        remotes = db.remotes()
        if not remotes:
            print("No remotes configured")
        for name, url in remotes.items():
            print(f"  {name}\t{url}")


def cmd_push(args):
    db = load_db(args)
    try:
        result = db.push(args.remote, branch=args.branch)
        if result["status"] == "up-to-date":
            print(f"Everything up-to-date")
        else:
            print(f"Pushed {result['objects_pushed']} objects to {args.remote}/{result['branch']}")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_pull(args):
    db = load_db(args)
    try:
        result = db.pull(args.remote, branch=args.branch)
        if result["status"] == "up-to-date":
            print("Already up-to-date")
        else:
            print(f"Pulled {result.get('objects_fetched', 0)} objects from {args.remote}")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_fetch(args):
    db = load_db(args)
    try:
        result = db.fetch(args.remote, branch=args.branch)
        if result["status"] == "up-to-date":
            print("Already up-to-date")
        else:
            print(f"Fetched {result['objects_fetched']} objects from {args.remote}")
    except ValueError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_pr(args):
    db = load_db(args)
    if args.action == "create":
        try:
            pr = db.pr_create(
                title=args.title,
                source_branch=args.source,
                target_branch=args.target or "main",
                description=args.description or "",
                author=args.author or "anonymous",
            )
            print(f"Created {pr}")
        except ValueError as e:
            print(f"fatal: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "list":
        prs = db.pr_list(status=args.status)
        if not prs:
            print("No pull requests")
        for pr in prs:
            status_color = {"open": "32", "merged": "35", "closed": "31"}.get(pr.status, "0")
            print(f"  #{pr.id}  \033[{status_color}m[{pr.status}]\033[0m  {pr.title}")
            print(f"       {pr.source_branch} → {pr.target_branch}  ({len(pr.comments)} comments)")
    elif args.action == "show":
        try:
            pr = db.pr_show(args.pr_id)
            status_color = {"open": "32", "merged": "35", "closed": "31"}.get(pr.status, "0")
            print(f"\033[{status_color}m#{pr.id} [{pr.status}]\033[0m {pr.title}")
            print(f"  {pr.source_branch} → {pr.target_branch}")
            print(f"  Author: {pr.author}")
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(pr.created_at))
            print(f"  Created: {ts}")
            if pr.description:
                print(f"\n  {pr.description}\n")
            # Show diff preview if PR is open
            if pr.status == "open":
                try:
                    d = db.diff(pr.target_branch, pr.source_branch)
                    if d.added_count or d.removed_count or d.modified_count:
                        print(f"\n  Changes: \033[32m+{d.added_count}\033[0m \033[31m-{d.removed_count}\033[0m \033[33m~{d.modified_count}\033[0m")
                        print()
                        unified = d.unified(pr.target_branch, pr.source_branch)
                        for line in unified.splitlines():
                            if line.startswith("diff --gitdb"):
                                print(f"  \033[1m{line}\033[0m")
                            elif line.startswith("---"):
                                print(f"  \033[1;31m{line}\033[0m")
                            elif line.startswith("+++"):
                                print(f"  \033[1;32m{line}\033[0m")
                            elif line.startswith("@@"):
                                print(f"  \033[36m{line}\033[0m")
                            elif line.startswith("-"):
                                print(f"  \033[31m{line}\033[0m")
                            elif line.startswith("+"):
                                print(f"  \033[32m{line}\033[0m")
                            else:
                                print(f"  {line}")
                except Exception:
                    pass
            if pr.comments:
                print(f"\n  Comments ({len(pr.comments)}):")
                for i, c in enumerate(pr.comments):
                    cts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.timestamp))
                    print(f"    [{cts}] {c.author}: {c.message}")
            if pr.merge_commit:
                print(f"\n  Merged: {pr.merge_commit[:12]}...")
        except ValueError as e:
            print(f"fatal: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "merge":
        try:
            result = db.pr_merge(args.pr_id, strategy=args.strategy or "union")
            branch = db.refs.current_branch
            print(f"Merge made by the '{result.strategy}' strategy.")
            parts = []
            if result.added: parts.append(f"{result.added} insertion(+)")
            if result.removed: parts.append(f"{result.removed} deletion(-)")
            if parts:
                print(f" {', '.join(parts)}")
            if result.has_conflicts:
                print(f"CONFLICT: {len(result.conflicts)} conflicting entries")
        except ValueError as e:
            print(f"fatal: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.action == "close":
        db.pr_close(args.pr_id)
        print(f"Closed PR #{args.pr_id}")
    elif args.action == "comment":
        db.pr_comment(args.pr_id, author=args.author or "anonymous", message=args.message)
        print(f"Comment added to PR #{args.pr_id}")


def cmd_insert(args):
    """Insert a JSON document."""
    db = load_db(args)
    try:
        doc = json.loads(args.document)
    except json.JSONDecodeError as e:
        print(f"fatal: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    _id = db.insert(doc)
    print(f"Inserted 1 document ({_id})")


def cmd_find(args):
    """Find documents matching a query."""
    db = load_db(args)
    where = json.loads(args.where) if args.where else None
    results = db.find(where=where, limit=args.limit, skip=args.skip or 0)
    if not results:
        print("No matching documents")
        return
    for doc in results:
        print(json.dumps(doc, default=str))


def cmd_doc_find(args):
    """Find documents matching a query (MongoDB-style)."""
    db = load_db(args)
    where = json.loads(args.where) if args.where else None
    columns = args.columns.split(",") if args.columns else None
    results = db.docs.select(
        columns=columns,
        where=where,
        order_by=getattr(args, "order_by", None),
        desc=getattr(args, "desc", False),
        limit=args.limit,
        offset=getattr(args, "offset", 0) or 0,
    )
    if not results:
        print("No matching documents")
        return

    if columns and results:
        # Table format
        widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in results)) for c in columns}
        header = "  ".join(c.ljust(widths[c]) for c in columns)
        print(header)
        print("  ".join("-" * widths[c] for c in columns))
        for row in results:
            print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    else:
        for doc in results:
            print(json.dumps(doc, default=str))
    print(f"\n{len(results)} row(s)")


def cmd_doc_update(args):
    """Update documents matching a query."""
    db = load_db(args)
    where = json.loads(args.where)
    set_fields = json.loads(args.set)
    count = db.update_docs(where, set_fields)
    print(f"Updated {count} document(s)")


def cmd_doc_delete(args):
    """Delete documents matching a query."""
    db = load_db(args)
    where = json.loads(args.where)
    count = db.delete_docs(where)
    print(f"Deleted {count} document(s)")


def cmd_embed(args):
    """Embedding model management and one-shot embedding."""
    if args.action == "list":
        from gitdb.embed import list_models
        models = list_models()
        for name, spec in models.items():
            default = " (default)" if name == "arctic" else ""
            print(f"  {name}{default}")
            print(f"    {spec['description']}")
            print(f"    {spec['params']} params, {spec['dim']}d, model: {spec['hf_name']}")
    elif args.action == "text":
        if not args.text:
            print("fatal: --text required", file=sys.stderr)
            sys.exit(1)
        from gitdb.embed import embed
        model = getattr(args, "model", None) or "arctic"
        vectors = embed(args.text, model_name=model)
        print(f"Embedded {len(args.text)} text(s) → {vectors.shape}")
        if args.output:
            torch.save(vectors, args.output)
            print(f"Saved to {args.output}")
        else:
            print(vectors)
    elif args.action == "info":
        from gitdb.embed import get_model_info
        model = getattr(args, "model", None) or "arctic"
        info = get_model_info(model)
        for k, v in info.items():
            print(f"  {k}: {v}")


def cmd_re_embed(args):
    """Re-embed all vectors using their document text."""
    db = load_db(args)
    model = getattr(args, "model", None) or "arctic"
    msg = db.re_embed(embed_model=model, embed_dim=getattr(args, "dim", None))
    print(msg)
    if args.autocommit:
        h = db.commit(f"Re-embed: {msg}")
        print(f"Committed: {h[:12]}...")


def cmd_semantic_blame(args):
    """Blame by concept — find which commits introduced matching vectors."""
    db = load_db(args)
    model = getattr(args, "model", None)
    results = db.semantic_blame(
        args.query,
        threshold=args.threshold,
        embed_model=model,
    )
    if not results:
        print("No matching vectors found")
        return
    for r in results:
        doc = r["document"] or ""
        if len(doc) > 50:
            doc = doc[:47] + "..."
        print(f"  {r['similarity']:.4f}  {r['commit'][:8]}  {doc}  {r['message']}")


def cmd_semantic_cherry_pick(args):
    """Cherry-pick commits matching a semantic query."""
    db = load_db(args)
    model = getattr(args, "model", None)
    result = db.semantic_cherry_pick(
        args.branch,
        args.query,
        threshold=args.threshold,
        embed_model=model,
    )
    if result:
        print(f"Cherry-picked matching commits → {result[:8]}")
    else:
        print("No commits matched the query")


def cmd_select(args):
    """SQL-style SELECT over vector metadata."""
    db = load_db(args)
    fields = args.fields.split(",") if args.fields else None
    where = json.loads(args.where) if args.where else None
    rows = db.select(
        fields=fields,
        where=where,
        order_by=args.order_by,
        limit=args.limit,
        reverse=args.desc,
    )
    if not rows:
        print("No results")
        return
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    elif args.format == "jsonl":
        for row in rows:
            print(json.dumps(row))
    else:
        # Table format
        if rows:
            headers = list(rows[0].keys())
            widths = [max(len(str(h)), max(len(str(r.get(h, ""))) for r in rows)) for h in headers]
            widths = [min(w, 40) for w in widths]
            header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
            print(f"  {header_line}")
            print(f"  {'  '.join('-' * w for w in widths)}")
            for row in rows:
                vals = []
                for h, w in zip(headers, widths):
                    v = str(row.get(h, ""))
                    if len(v) > w:
                        v = v[:w-3] + "..."
                    vals.append(v.ljust(w))
                print(f"  {'  '.join(vals)}")
        print(f"\n  {len(rows)} rows")


def cmd_group_by(args):
    """GROUP BY aggregation."""
    db = load_db(args)
    where = json.loads(args.where) if args.where else None
    result = db.group_by(
        field=args.field,
        agg_field=args.agg_field,
        agg_fn=args.agg_fn or "count",
        where=where,
    )
    for key, val in sorted(result.items(), key=lambda x: str(x[0])):
        print(f"  {key}: {val}")


def cmd_export(args):
    """Export store to JSONL."""
    db = load_db(args)
    db.export_jsonl(args.output, include_embeddings=args.embeddings)
    print(f"Exported to {args.output}")


def cmd_import(args):
    """Import from JSONL."""
    db = load_db(args)
    indices = db.import_jsonl(
        args.file,
        embed_texts=args.embed,
        embed_model=getattr(args, "embed_model", None),
    )
    print(f"Imported {len(indices)} rows from {args.file}")
    if args.autocommit:
        h = db.commit(f"Import {len(indices)} rows from {Path(args.file).name}")
        print(f"Committed: {h[:12]}...")


def cmd_ac(args):
    """Emirati AC — ambient activation engine."""
    db = load_db(args)
    if args.action == "status":
        stats = db.ac.stats()
        running = "\033[32mrunning\033[0m" if stats["running"] else "\033[31mstopped\033[0m"
        print(f"  Emirati AC: {running}")
        print(f"  Active vectors: {stats['active_vectors']}")
        print(f"  Hot cache: {stats['hot_cache_size']}")
        print(f"  Cycles: {stats['cycles']}")
        print(f"  Last cycle: {stats['last_cycle_ms']}ms")
        print(f"  Peak active: {stats['peak_active']}")
        print(f"  Spread hops: {stats['total_spread_hops']}")
        print(f"  Top activation: {stats['top_score']}")
        drift = stats.get("drift")
        if drift:
            sev = drift["severity"]
            color = "31" if sev == "high" else "33"
            print(f"  \033[{color}mDrift: {drift['magnitude']} ({sev})\033[0m")
    elif args.action == "primed":
        k = args.k or 10
        results = db.ac.primed(k)
        if not results.ids:
            print("No activated vectors (AC may not be running)")
            return
        print(f"  Hot cache ({len(results.ids)} vectors):")
        for i in range(len(results)):
            score = f"{results.scores[i]:.4f}"
            doc = results.documents[i] or ""
            if len(doc) > 60:
                doc = doc[:57] + "..."
            print(f"    {score}  {doc}")
    elif args.action == "drift":
        drift = db.ac.drift()
        if drift:
            print(f"  Drift magnitude: {drift['magnitude']}")
            print(f"  Similarity to centroid: {drift['similarity_to_centroid']}")
            print(f"  Recent additions: {drift['recent_count']}")
            print(f"  Severity: {drift['severity']}")
        else:
            print("  No significant drift detected")
    elif args.action == "start":
        db.ac.start()
        print("  Emirati AC started — engine running, AC on")
    elif args.action == "stop":
        db.ac.stop()
        print("  Emirati AC stopped")


def cmd_comment(args):
    db = load_db(args)
    if args.message:
        db.comment(args.ref or "HEAD", author=args.author or "anonymous", message=args.message)
        print(f"Comment added to {args.ref or 'HEAD'}")
    else:
        comments = db.comments(args.ref or "HEAD")
        if not comments:
            print("No comments")
        for c in comments:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["timestamp"]))
            print(f"  {ts}  {c['message']}")


def cmd_watch(args):
    db = load_db(args)
    if args.action == "list":
        watches = db.watches.list_watches()
        if not watches:
            print("No active watches")
            return
        for w in watches:
            if w["type"] == "branch":
                print(f"  #{w['id']}  branch={w['branch']}")
            else:
                print(f"  #{w['id']}  where={json.dumps(w['where'])}")
    elif args.action == "clear":
        # Unwatch all
        for w in db.watches.list_watches():
            db.watches.unwatch(w["id"])
        print("All watches cleared")


def cmd_index(args):
    db = load_db(args)
    if args.action == "create":
        idx_type = args.type or "hash"
        db.create_index(args.field, idx_type)
        print(f"Created {idx_type} index on '{args.field}'")
    elif args.action == "drop":
        db.drop_index(args.field)
        print(f"Dropped index on '{args.field}'")
    elif args.action == "list":
        indexes = db.list_indexes()
        if not indexes:
            print("No indexes")
            return
        for idx in indexes:
            print(f"  {idx['field']:20s}  {idx['type']:6s}  {idx['entries']:>6d} entries")
    elif args.action == "lookup":
        if not args.field or not args.value:
            print("Provide --field and --value", file=sys.stderr)
            sys.exit(1)
        # Try to parse value as JSON, fallback to string
        try:
            val = json.loads(args.value)
        except (json.JSONDecodeError, TypeError):
            val = args.value
        results = db.indexes.lookup(args.field, val)
        print(f"{len(results)} matches: {results}")


def cmd_backup(args):
    db = load_db(args)
    if args.action == "full":
        output = args.output or f"backup_{int(time.time())}.gitdb-backup"
        manifest = db.backup(output)
        print(f"Full backup: {output}")
        print(f"  Objects: {manifest['object_count']}, Size: {manifest['backup_size_bytes']} bytes")
    elif args.action == "incremental":
        output = args.output or f"backup_{int(time.time())}.gitdb-incr"
        manifest = db.backup_incremental(output)
        print(f"Incremental backup: {output}")
        print(f"  New objects: {manifest.get('new_objects', 0)}, Size: {manifest['backup_size_bytes']} bytes")
    elif args.action == "list":
        backups = db.backup_list()
        if not backups:
            print("No backups recorded")
            return
        for b in backups:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b.get("timestamp", 0)))
            print(f"  {ts}  {b.get('type', '?'):12s}  {b.get('size', 0):>10d} bytes  {b.get('path', '?')}")
    elif args.action == "verify":
        result = db.backup_verify()
        if result["valid"]:
            print(f"Store is valid. {result['commits_verified']} commits verified, {result['total_objects_on_disk']} objects on disk.")
        else:
            print("ISSUES FOUND:")
            for issue in result["issues"]:
                print(f"  - {issue}")


def cmd_restore(args):
    from gitdb import GitDB
    from gitdb.backup import restore
    manifest = restore(args.backup, args.dest, overwrite=args.force)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(manifest.get("timestamp", 0)))
    print(f"Restored {manifest.get('type', '?')} backup from {ts}")
    print(f"  Objects: {manifest.get('object_count', '?')}, Dest: {args.dest}")


def cmd_snapshot(args):
    db = load_db(args)
    if args.action == "create":
        snap = db.snapshot(args.name)
        print(f"Snapshot '{snap.name}': {snap.size} vectors")
    elif args.action == "list":
        snaps = db.snapshots()
        if not snaps:
            print("No snapshots")
            return
        for s in snaps:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s["timestamp"]))
            print(f"  {s['name']:20s}  {s['size']:>6d} vectors  {ts}")
    elif args.action == "query":
        snap = db.get_snapshot(args.name)
        if args.text:
            results = snap.query_text(args.text, k=args.k or 10)
        elif args.vector:
            vec = torch.load(args.vector, weights_only=True)
            results = snap.query(vec, k=args.k or 10)
        else:
            print("Provide --text or --vector", file=sys.stderr)
            sys.exit(1)
        for i, (vid, score) in enumerate(zip(results.ids, results.scores)):
            doc = results.documents[i] or ""
            print(f"  {score:.4f}  {vid[:16]}  {doc[:60]}")


def cmd_hook(args):
    db = load_db(args)
    if args.action == "list":
        hooks = db.hooks.list_hooks()
        if not hooks:
            print("No hooks registered")
            return
        for event, callbacks in hooks.items():
            for cb in callbacks:
                print(f"  {event:16s}  {cb}")
    elif args.action == "clear":
        event = args.event if hasattr(args, 'event') and args.event else None
        db.hooks.clear(event)
        print(f"Cleared hooks" + (f" for {event}" if event else ""))


def cmd_ingest(args):
    db = load_db(args)
    from gitdb.ingest import ingest_file, ingest_directory, detect_type, SUPPORTED_EXTENSIONS
    from gitdb.cloud_ingest import is_cloud_uri, ingest_cloud

    target = args.target

    if is_cloud_uri(target):
        result = ingest_cloud(
            db, target,
            embed=not args.no_embed,
            autocommit=not args.no_commit,
        )
        print(f"Ingested {result['files_ingested']} files from {result['scheme']}://, {result['total_rows']} total rows")
        for r in result.get("results", []):
            key = r.get("_cloud_key", "?")
            count = r.get("rows", r.get("chunks", r.get("documents", 0)))
            print(f"  ✓ {key}: {count}")
        return

    if os.path.isdir(target):
        result = ingest_directory(
            db, target,
            pattern=args.pattern or "*",
            recursive=not args.no_recursive,
            embed=not args.no_embed,
            autocommit=not args.no_commit,
        )
        print(f"Ingested {result['total_files']} files, {result['total_rows']} total rows")
        for fname, r in result["files"].items():
            if "error" in r:
                print(f"  ✗ {fname}: {r['error']}")
            else:
                count = r.get("rows", r.get("chunks", r.get("documents", 0)))
                print(f"  ✓ {fname}: {count}")
    else:
        file_type = detect_type(target)
        kwargs = {}
        if args.text_column:
            if file_type == "csv":
                kwargs["text_column"] = args.text_column
            elif file_type == "sqlite":
                # For SQLite, text_columns is a dict {table: column}
                # Simple case: use same column for all tables
                kwargs["text_columns"] = {"*": args.text_column}
        if args.chunk_size:
            kwargs["chunk_size"] = args.chunk_size
        if args.chunk_overlap:
            kwargs["chunk_overlap"] = args.chunk_overlap

        result = ingest_file(
            db, target,
            embed=not args.no_embed,
            autocommit=not args.no_commit,
            **kwargs,
        )

        # Print results based on type
        if "tables" in result:
            print(f"Ingested SQLite: {result['tables']} tables, {result['total_rows']} rows")
        elif "chunks" in result:
            print(f"Ingested: {result['chunks']} chunks ({result.get('total_chars', 0)} chars)")
        elif "documents" in result:
            print(f"Ingested MongoDB: {result['documents']} documents")
        elif "rows" in result:
            print(f"Ingested: {result['rows']} rows")
        else:
            print(f"Ingested: {json.dumps(result, indent=2)}")


def cmd_schema(args):
    db = load_db(args)
    if args.action == "show":
        schema = db.get_schema()
        if schema:
            print(json.dumps(schema, indent=2))
        else:
            print("No schema set")
    elif args.action == "set":
        if not args.file:
            print("Provide --file with JSON schema", file=sys.stderr)
            sys.exit(1)
        definition = json.loads(Path(args.file).read_text())
        db.set_schema(definition)
        print(f"Schema set from {args.file}")
    elif args.action == "clear":
        db.set_schema(None)
        print("Schema cleared")
    elif args.action == "validate":
        schema = db.get_schema()
        if not schema:
            print("No schema set")
            return
        from gitdb.schema import Schema
        s = Schema(schema)
        errors_total = 0
        for i, m in enumerate(db.tree.metadata):
            errors = s.validate(m.metadata)
            if errors:
                errors_total += len(errors)
                print(f"  Row {i} ({m.id}): {'; '.join(errors)}")
        if errors_total == 0:
            print(f"All {len(db.tree.metadata)} rows pass schema validation")


def main():
    parser = argparse.ArgumentParser(
        prog="gitdb",
        description="GPU-accelerated version-controlled vector database",
    )
    parser.add_argument("--path", help="Path to GitDB store")
    parser.add_argument("--device", help="Override device (cpu/mps/cuda)")
    sub = parser.add_subparsers(dest="command")

    # init
    p = sub.add_parser("init", help="Create a new GitDB store")
    p.add_argument("name", help="Store name/path")
    p.add_argument("--dim", type=int, default=1024, help="Embedding dimension")
    p.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")

    # status
    sub.add_parser("status", help="Show working tree status")

    # add
    p = sub.add_parser("add", help="Add vectors from file or text")
    p.add_argument("file", nargs="?", help="Vectors file (.pt, .npy, .jsonl)")
    p.add_argument("--text", nargs="+", help="Text(s) to embed and add")
    p.add_argument("--text-file", help="File with one text per line to embed")
    p.add_argument("--documents", help="Documents JSON file")
    p.add_argument("--metadata", help="Metadata JSON file")
    p.add_argument("--embed-model", default=None, help="Embedding model (arctic, nv-embed-qa)")
    p.add_argument("-c", "--autocommit", action="store_true", help="Auto-commit after add")

    # commit
    p = sub.add_parser("commit", help="Commit staged changes")
    p.add_argument("-m", "--message", required=True, help="Commit message")

    # log
    p = sub.add_parser("log", help="Show commit history")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.add_argument("--source", help="Filter by source keyword")

    # diff
    p = sub.add_parser("diff", help="Compare two refs")
    p.add_argument("ref_a")
    p.add_argument("ref_b")

    # query
    p = sub.add_parser("query", help="Similarity search")
    p.add_argument("--text", help="Text query (auto-embeds)")
    p.add_argument("--vector", help="Query vector file (.pt)")
    p.add_argument("--random", action="store_true", help="Random query vector")
    p.add_argument("-k", type=int, default=10, help="Top-K results")
    p.add_argument("--where", help="Metadata filter JSON")
    p.add_argument("--at", help="Time-travel ref")
    p.add_argument("--embed-model", default=None, help="Embedding model")

    # branch
    p = sub.add_parser("branch", help="List or create branches")
    p.add_argument("name", nargs="?", help="Branch name to create")

    # switch
    p = sub.add_parser("switch", help="Switch branch")
    p.add_argument("branch")

    # checkout
    p = sub.add_parser("checkout", help="Checkout a ref")
    p.add_argument("ref")

    # merge
    p = sub.add_parser("merge", help="Merge a branch")
    p.add_argument("branch")
    p.add_argument("--strategy", default="union", choices=["union", "ours", "theirs"])

    # cherry-pick
    p = sub.add_parser("cherry-pick", help="Cherry-pick a commit")
    p.add_argument("ref")

    # revert
    p = sub.add_parser("revert", help="Revert a commit")
    p.add_argument("ref")

    # stash
    p = sub.add_parser("stash", help="Stash/pop/list")
    p.add_argument("action", nargs="?", default="save", choices=["save", "pop", "list"])
    p.add_argument("-m", "--message", help="Stash message")

    # tag
    p = sub.add_parser("tag", help="Create a tag")
    p.add_argument("name")
    p.add_argument("--ref", help="Ref to tag (default: HEAD)")

    # reset
    p = sub.add_parser("reset", help="Reset to ref")
    p.add_argument("ref", nargs="?", default="HEAD")

    # blame
    sub.add_parser("blame", help="Show vector provenance")

    # reflog
    p = sub.add_parser("reflog", help="Show reflog")
    p.add_argument("-n", "--limit", type=int, default=20)

    # purge
    p = sub.add_parser("purge", help="Purge vectors from all history")
    p.add_argument("--where", help="Metadata filter JSON")
    p.add_argument("--ids", help="Comma-separated vector IDs")
    p.add_argument("--reason", help="Audit reason")

    # gc
    p = sub.add_parser("gc", help="Garbage collect")
    p.add_argument("--keep", type=int, default=10, help="Checkpoint interval")

    # rebase
    p = sub.add_parser("rebase", help="Rebase onto ref")
    p.add_argument("onto")

    # filter-branch
    p = sub.add_parser("filter-branch", help="Apply transform to all embeddings")
    p.add_argument("transform", choices=["normalize", "scale2x", "zero-mean"])

    # show
    p = sub.add_parser("show", help="Show commit details")
    p.add_argument("ref", nargs="?", default="HEAD")

    # amend
    p = sub.add_parser("amend", help="Amend last commit")
    p.add_argument("-m", "--message", help="New commit message")

    # squash
    p = sub.add_parser("squash", help="Squash last N commits into one")
    p.add_argument("n", type=int, help="Number of commits to squash")
    p.add_argument("-m", "--message", help="Squash commit message")

    # fork / clone
    p = sub.add_parser("fork", help="Fork database to new location")
    p.add_argument("dest", help="Destination path")
    p.add_argument("--branch", help="Branch to checkout after fork")

    p = sub.add_parser("clone", help="Clone database (alias for fork)")
    p.add_argument("dest", help="Destination path")
    p.add_argument("--branch", help="Branch to checkout after clone")

    # note
    p = sub.add_parser("note", help="Add or show notes on a commit")
    p.add_argument("--ref", default="HEAD", help="Commit ref")
    p.add_argument("-m", "--message", help="Note message (omit to show notes)")

    # branch -d
    p = sub.add_parser("branch-delete", help="Delete a branch")
    p.add_argument("name")

    # branch -m
    p = sub.add_parser("branch-rename", help="Rename a branch")
    p.add_argument("old")
    p.add_argument("new")

    # head
    p = sub.add_parser("head", help="Show HEAD commit")
    p.add_argument("-q", "--quiet", action="store_true", help="Just print hash")

    # remote
    p = sub.add_parser("remote", help="Manage remotes")
    p.add_argument("action", nargs="?", default="list", choices=["add", "remove", "list"])
    p.add_argument("name", nargs="?")
    p.add_argument("url", nargs="?")

    # push
    p = sub.add_parser("push", help="Push to remote")
    p.add_argument("remote", help="Remote name")
    p.add_argument("branch", nargs="?", help="Branch (default: current)")

    # pull
    p = sub.add_parser("pull", help="Pull from remote")
    p.add_argument("remote", help="Remote name")
    p.add_argument("branch", nargs="?", help="Branch (default: current)")

    # fetch
    p = sub.add_parser("fetch", help="Fetch from remote")
    p.add_argument("remote", help="Remote name")
    p.add_argument("branch", nargs="?", help="Branch (default: current)")

    # pr
    p = sub.add_parser("pr", help="Pull requests")
    p.add_argument("action", choices=["create", "list", "show", "merge", "close", "comment"])
    p.add_argument("--title", help="PR title")
    p.add_argument("--source", help="Source branch")
    p.add_argument("--target", help="Target branch (default: main)")
    p.add_argument("--description", help="PR description")
    p.add_argument("--author", help="Author name")
    p.add_argument("--pr-id", type=int, help="PR ID for show/merge/close/comment")
    p.add_argument("--status", help="Filter by status (open/merged/closed)")
    p.add_argument("--strategy", help="Merge strategy")
    p.add_argument("-m", "--message", help="Comment message")

    # comment
    p = sub.add_parser("comment", help="Comment on a commit")
    p.add_argument("--ref", default="HEAD", help="Commit ref")
    p.add_argument("--author", help="Author name")
    p.add_argument("-m", "--message", help="Comment (omit to show)")

    # select (SQL-style query)
    p = sub.add_parser("select", help="SQL-style SELECT over metadata")
    p.add_argument("--fields", help="Comma-separated fields to return")
    p.add_argument("--where", help="JSON filter (supports $gt, $in, $regex, etc)")
    p.add_argument("--order-by", help="Sort by field")
    p.add_argument("--limit", type=int, help="Max rows")
    p.add_argument("--desc", action="store_true", help="Sort descending")
    p.add_argument("--format", choices=["table", "json", "jsonl"], default="table")

    # group-by
    p = sub.add_parser("group-by", help="GROUP BY aggregation")
    p.add_argument("field", help="Field to group by")
    p.add_argument("--agg-field", help="Field to aggregate")
    p.add_argument("--agg-fn", choices=["count", "sum", "avg", "min", "max", "collect"])
    p.add_argument("--where", help="Pre-filter JSON")

    # export
    p = sub.add_parser("export", help="Export store to JSONL")
    p.add_argument("output", help="Output file path (.jsonl)")
    p.add_argument("--embeddings", action="store_true", help="Include embedding vectors")

    # import
    p = sub.add_parser("import", help="Import from JSONL")
    p.add_argument("file", help="Input JSONL file")
    p.add_argument("--embed", action="store_true", help="Auto-embed from document text")
    p.add_argument("--embed-model", help="Embedding model")
    p.add_argument("-c", "--autocommit", action="store_true")

    # ac (Emirati AC)
    p = sub.add_parser("ac", help="Emirati AC — ambient activation engine")
    p.add_argument("action", choices=["status", "primed", "drift", "start", "stop"])
    p.add_argument("-k", type=int, help="Top-K primed results")

    # embed
    p = sub.add_parser("embed", help="Embedding model tools")
    p.add_argument("action", choices=["list", "text", "info"], help="Action")
    p.add_argument("--text", nargs="+", help="Text(s) to embed")
    p.add_argument("--model", default="arctic", help="Model name")
    p.add_argument("--output", help="Save tensor to file (.pt)")

    # re-embed
    p = sub.add_parser("re-embed", help="Re-embed all vectors from document text")
    p.add_argument("--model", default="arctic", help="Embedding model")
    p.add_argument("--dim", type=int, help="Target dimension")
    p.add_argument("-c", "--autocommit", action="store_true", help="Auto-commit")

    # semantic-blame
    p = sub.add_parser("semantic-blame", help="Blame by concept")
    p.add_argument("query", help="Semantic query text")
    p.add_argument("--threshold", type=float, default=0.5, help="Min similarity")
    p.add_argument("--model", help="Embedding model")

    # semantic-cherry-pick
    p = sub.add_parser("semantic-cherry-pick", help="Cherry-pick by semantic match")
    p.add_argument("branch", help="Source branch")
    p.add_argument("query", help="Semantic query text")
    p.add_argument("--threshold", type=float, default=0.5, help="Min similarity")
    p.add_argument("--model", help="Embedding model")

    # backup
    p = sub.add_parser("backup", help="Backup operations")
    p.add_argument("action", choices=["full", "incremental", "list", "verify"])
    p.add_argument("--output", "-o", help="Output file path")

    # restore
    p = sub.add_parser("restore", help="Restore from backup")
    p.add_argument("backup", help="Backup file path")
    p.add_argument("dest", help="Destination directory")
    p.add_argument("--force", "-f", action="store_true", help="Overwrite existing")

    # snapshot
    p = sub.add_parser("snapshot", help="In-memory snapshots")
    p.add_argument("action", choices=["create", "list", "query"])
    p.add_argument("--name", help="Snapshot name")
    p.add_argument("--text", help="Query text (for query action)")
    p.add_argument("--vector", help="Query vector file (for query action)")
    p.add_argument("-k", type=int, help="Top-K results")

    # hook
    p = sub.add_parser("hook", help="Event hooks")
    p.add_argument("action", choices=["list", "clear"])
    p.add_argument("--event", help="Event name (for clear)")

    # watch
    p = sub.add_parser("watch", help="Change watches")
    p.add_argument("action", choices=["list", "clear"])

    # index
    p = sub.add_parser("index", help="Secondary indexes")
    p.add_argument("action", choices=["create", "drop", "list", "lookup"])
    p.add_argument("--field", help="Field name")
    p.add_argument("--type", choices=["hash", "range"], help="Index type")
    p.add_argument("--value", help="Lookup value")

    # schema
    p = sub.add_parser("schema", help="Schema enforcement")
    p.add_argument("action", choices=["show", "set", "clear", "validate"])
    p.add_argument("--file", help="Schema JSON file (for set)")

    # serve
    p = sub.add_parser("serve", help="Launch web dashboard")
    p.add_argument("--port", type=int, default=7474, help="Port (default 7474)")
    p.add_argument("--no-browser", action="store_true", help="Don't open browser")

    # ingest
    p = sub.add_parser("ingest", help="Ingest files (SQLite, MongoDB, CSV, Parquet, PDF, text)")
    p.add_argument("target", help="File or directory to ingest")
    p.add_argument("--text-column", help="Column to embed (for CSV/SQLite)")
    p.add_argument("--chunk-size", type=int, help="Chunk size for text/PDF (default 500)")
    p.add_argument("--chunk-overlap", type=int, help="Chunk overlap (default 50)")
    p.add_argument("--pattern", help="Glob pattern for directory ingest (default *)")
    p.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirs")
    p.add_argument("--no-embed", action="store_true", help="Skip embedding (store raw text)")
    p.add_argument("--no-commit", action="store_true", help="Don't auto-commit after ingest")

    # insert (document)
    p = sub.add_parser("insert", help="Insert a JSON document")
    p.add_argument("document", help="JSON document string")

    # find (document)
    p = sub.add_parser("find", help="Find documents (MongoDB-style)")
    p.add_argument("--where", help="Query filter JSON")
    p.add_argument("--columns", help="Comma-separated fields to return")
    p.add_argument("--order-by", help="Sort by field")
    p.add_argument("--desc", action="store_true", help="Sort descending")
    p.add_argument("--limit", type=int, help="Max documents")
    p.add_argument("--offset", type=int, default=0, help="Skip documents")
    p.add_argument("--skip", type=int, default=0, help="Skip documents (alias)")

    # update (document)
    p = sub.add_parser("update", help="Update documents matching query")
    p.add_argument("--where", required=True, help="Query filter JSON")
    p.add_argument("--set", required=True, help="Fields to update (JSON)")

    # delete (document — careful: different from branch-delete)
    p = sub.add_parser("delete", help="Delete documents matching query")
    p.add_argument("--where", required=True, help="Query filter JSON")

    # count (document)
    p = sub.add_parser("count", help="Count documents")
    p.add_argument("--where", help="Query filter JSON")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "init": cmd_init,
        "status": cmd_status,
        "add": cmd_add,
        "commit": cmd_commit,
        "log": cmd_log,
        "diff": cmd_diff,
        "query": cmd_query,
        "branch": cmd_branch,
        "switch": cmd_switch,
        "checkout": cmd_checkout,
        "merge": cmd_merge,
        "cherry-pick": cmd_cherry_pick,
        "revert": cmd_revert,
        "stash": cmd_stash,
        "tag": cmd_tag,
        "reset": cmd_reset,
        "blame": cmd_blame,
        "reflog": cmd_reflog,
        "purge": cmd_purge,
        "gc": cmd_gc,
        "rebase": cmd_rebase,
        "filter-branch": cmd_filter_branch,
        "show": cmd_show,
        "amend": cmd_amend,
        "squash": cmd_squash,
        "fork": cmd_fork,
        "clone": cmd_fork,
        "note": cmd_note,
        "branch-delete": cmd_delete_branch,
        "branch-rename": cmd_rename_branch,
        "head": cmd_head,
        "remote": cmd_remote,
        "push": cmd_push,
        "pull": cmd_pull,
        "fetch": cmd_fetch,
        "pr": cmd_pr,
        "comment": cmd_comment,
        "ac": cmd_ac,
        "select": cmd_select,
        "group-by": cmd_group_by,
        "export": cmd_export,
        "import": cmd_import,
        "embed": cmd_embed,
        "re-embed": cmd_re_embed,
        "semantic-blame": cmd_semantic_blame,
        "semantic-cherry-pick": cmd_semantic_cherry_pick,
        "backup": cmd_backup,
        "restore": cmd_restore,
        "snapshot": cmd_snapshot,
        "hook": cmd_hook,
        "watch": cmd_watch,
        "index": cmd_index,
        "schema": cmd_schema,
        "ingest": cmd_ingest,
        "serve": lambda args: __import__('gitdb.server', fromlist=['cmd_serve']).cmd_serve(args),
        "insert": cmd_insert,
        "find": cmd_doc_find,
        "update": cmd_doc_update,
        "delete": cmd_doc_delete,
        "count": lambda args: print(f"{load_db(args).count_docs(where=json.loads(args.where) if args.where else None)} document(s)"),
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
