#!/usr/bin/env python3
"""
Lightweight server-side daemon for Hand Motion Viewer.
Communicates via standard input/output with the remote desktop viewer.

Protocol:
- Requests arrive as JSON lines on stdin: {"id": <int>, "cmd": "<command>", ...}
- Text responses are written as JSON lines on stdout: {"id": <int>, "status": "ok"|"error", ...}\n
- Binary responses (get_file, bundle_frames, get_frame) write a JSON header line:
    {"id": <int>, "status": "ok", "size": <byte_length>, ...}\n
  followed immediately by exactly <byte_length> raw bytes.
"""

import io
import json
import os
import re
import sys
import traceback
import zipfile
from pathlib import Path


def natural_sort_key(s: str):
    """Sort strings with embedded numbers naturally (e.g. T1 before T10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def scan_tree(root_path_str: str):
    """Recursively scan directory, returning a pruned tree containing only .hexport files."""
    root_path = os.path.expanduser(root_path_str)
    if not os.path.isdir(root_path):
        return None

    def _scan(dir_path: str):
        children = []
        has_hexport = False
        try:
            with os.scandir(dir_path) as it:
                for entry in it:
                    if entry.name.startswith("."):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            sub = _scan(entry.path)
                            if sub:
                                children.append(sub)
                        elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".hexport"):
                            children.append({
                                "name": entry.name,
                                "path": entry.path,
                                "is_file": True,
                                "children": []
                            })
                            has_hexport = True
                    except OSError:
                        continue
        except (PermissionError, OSError):
            pass

        if children or has_hexport:
            # Sort: folders first, then natural order by name
            children.sort(key=lambda c: (c["is_file"], natural_sort_key(c["name"])))
            folder_name = os.path.basename(dir_path) or dir_path
            return {
                "name": folder_name,
                "path": dir_path,
                "is_file": False,
                "children": children
            }
        return None

    tree = _scan(root_path)
    if tree is None:
        folder_name = os.path.basename(root_path) or root_path
        tree = {
            "name": folder_name,
            "path": root_path,
            "is_file": False,
            "children": []
        }
    return tree


def find_frames_dir(export_path_str: str):
    """
    Locate the frames directory for an export file.
    Supports both:
    1. Standard sibling layout: <dir>/frames/<export_stem>/
    2. infant_grasp_pipeline cache layout:
       <cache_root>/hand_export/<hash>__<fps>__<params>.hexport ->
       <cache_root>/frames/<hash>__<fps>/
    """
    export_path = Path(os.path.expanduser(export_path_str))
    stem = export_path.stem

    # 1. Standard sibling layout
    candidate1 = export_path.parent / "frames" / stem
    if candidate1.is_dir():
        return candidate1

    # 2. Pipeline cache layout: hand_export -> frames/<hash>__<fps>
    if export_path.parent.name == "hand_export":
        cache_root = export_path.parent.parent
        if "__" in stem:
            frames_key = stem.rsplit("__", 1)[0]
            candidate2 = cache_root / "frames" / frames_key
            if candidate2.is_dir():
                return candidate2

    return None


def handle_scan_tree(req):
    root = req.get("root", "")
    tree = scan_tree(root)
    if tree is None:
        return {"status": "error", "message": f"Directory not found or unreadable: {root}"}
    return {"status": "ok", "tree": tree}


def handle_get_file(req, out):
    path_str = os.path.expanduser(req.get("path", ""))
    if not os.path.isfile(path_str):
        write_json_line(out, {"id": req["id"], "status": "error", "message": f"File not found: {path_str}"})
        return

    try:
        with open(path_str, "rb") as f:
            data = f.read()
        write_json_line(out, {"id": req["id"], "status": "ok", "size": len(data)})
        out.write(data)
        out.flush()
    except Exception as e:
        write_json_line(out, {"id": req["id"], "status": "error", "message": str(e)})


def handle_get_frame(req, out):
    export_path_str = req.get("export_path", "")
    frame_number = req.get("frame_number", 1)

    frames_dir = find_frames_dir(export_path_str)
    if not frames_dir:
        write_json_line(out, {"id": req["id"], "status": "not_found"})
        return

    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = frames_dir / f"frame_{frame_number:05d}{ext}"
        if candidate.exists():
            try:
                resolved = candidate.resolve()
                with open(resolved, "rb") as f:
                    data = f.read()
                write_json_line(out, {"id": req["id"], "status": "ok", "size": len(data)})
                out.write(data)
                out.flush()
                return
            except Exception as e:
                write_json_line(out, {"id": req["id"], "status": "error", "message": str(e)})
                return

    write_json_line(out, {"id": req["id"], "status": "not_found"})


def handle_bundle_frames(req, out):
    export_path_str = req.get("export_path", "")
    frames_dir = find_frames_dir(export_path_str)

    if not frames_dir:
        # No frames folder; send 0 size
        write_json_line(out, {"id": req["id"], "status": "ok", "size": 0, "frame_count": 0})
        return

    try:
        # Find all frame images
        frame_files = []
        for p in frames_dir.iterdir():
            if p.name.startswith("frame_") and p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                frame_files.append(p)

        frame_files.sort(key=lambda p: natural_sort_key(p.name))

        if not frame_files:
            write_json_line(out, {"id": req["id"], "status": "ok", "size": 0, "frame_count": 0})
            return

        # Pack into an uncompressed ZIP archive (JPEGs are already compressed)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for p in frame_files:
                resolved = p.resolve()
                if resolved.exists():
                    zf.write(resolved, arcname=p.name)

        zip_bytes = buf.getvalue()
        write_json_line(out, {
            "id": req["id"],
            "status": "ok",
            "size": len(zip_bytes),
            "frame_count": len(frame_files)
        })
        out.write(zip_bytes)
        out.flush()
    except Exception as e:
        write_json_line(out, {"id": req["id"], "status": "error", "message": str(e)})


def write_json_line(out, obj):
    line = json.dumps(obj) + "\n"
    out.write(line.encode("utf-8"))
    out.flush()


def main():
    stdin = sys.stdin
    stdout = sys.stdout.buffer

    # Optional handshake banner to verify connection
    sys.stderr.write("hand_motion_viewer daemon started\n")
    sys.stderr.flush()

    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            write_json_line(stdout, {"status": "error", "message": f"Invalid JSON: {e}"})
            continue

        req_id = req.get("id", 0)
        cmd = req.get("cmd", "")

        try:
            if cmd == "ping":
                write_json_line(stdout, {"id": req_id, "status": "ok", "version": 1})
            elif cmd == "scan_tree":
                res = handle_scan_tree(req)
                res["id"] = req_id
                write_json_line(stdout, res)
            elif cmd == "get_file":
                handle_get_file(req, stdout)
            elif cmd == "get_frame":
                handle_get_frame(req, stdout)
            elif cmd == "bundle_frames":
                handle_bundle_frames(req, stdout)
            elif cmd == "quit" or cmd == "exit":
                write_json_line(stdout, {"id": req_id, "status": "ok"})
                break
            else:
                write_json_line(stdout, {"id": req_id, "status": "error", "message": f"Unknown command: {cmd}"})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            write_json_line(stdout, {"id": req_id, "status": "error", "message": str(e)})


if __name__ == "__main__":
    main()
