#!/usr/bin/env python3
"""Sanitize textual image-placeholder tokens instead of rejecting the request.

``encoding_dsv41`` raises ``ValueError`` in two places when its reserved image
placeholder token appears in message text. Both guards are rewritten to
sanitize-and-warn, while real image blocks remain unchanged. The patch is
idempotent and fails closed when the pinned upstream anchors drift.
"""

from __future__ import annotations

import pathlib
import sys

DEFAULT_TARGET = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv41.py"
TAG = "# [encoding_dsv41-placeholder-sanitize]"
SENT1 = "sanitizing image placeholder token in message content"
SENT2 = "sanitizing image placeholder token in a text block"
RAISE1 = "Message content contains image special token"
RAISE2 = "Text block contains image placeholder"
LOGGER_BOOTSTRAP = "logger = logging.getLogger(__name__)  " + TAG + "\n"

OLD1 = """def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    \"\"\"Reject user-supplied image placeholder tokens in textual fields.\"\"\"
    content = msg.get(\"content\")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError(
            f\"Message content contains image special token '{IMAGE_PLACEHOLDER}'. \"
            \"Images should be provided as image content blocks.\"
        )
    reasoning_content = msg.get(\"reasoning_content\")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError(
            f\"reasoning_content contains image special token '{IMAGE_PLACEHOLDER}'\"
        )
"""

NEW1 = """def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    \"\"\"Sanitize textual image placeholder tokens instead of rejecting the request.

    Raising here turns a replayed placeholder into a self-sustaining request
    failure. Rewrite it into plain text so the conversation can proceed.
    # [encoding_dsv41-placeholder-sanitize]
    \"\"\"
    content = msg.get(\"content\")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        logger.warning(
            \"encoding_dsv41: sanitizing image placeholder token in message content \"
            \"(upstream would raise) # [encoding_dsv41-placeholder-sanitize]\"
        )
        msg[\"content\"] = content.replace(IMAGE_PLACEHOLDER, \"[image]\")
    reasoning_content = msg.get(\"reasoning_content\")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        logger.warning(
            \"encoding_dsv41: sanitizing image placeholder token in reasoning_content \"
            \"(upstream would raise) # [encoding_dsv41-placeholder-sanitize]\"
        )
        msg[\"reasoning_content\"] = reasoning_content.replace(IMAGE_PLACEHOLDER, \"[image]\")
"""


def ensure_logger(source: str) -> str:
    """插入模块 logger，保持补丁在不同上游导入布局下可重放。"""
    if LOGGER_BOOTSTRAP in source:
        return source
    if "import logging" not in source:
        anchor = "from typing import Any, Dict, List, Optional, Tuple, Union\n"
        if source.count(anchor) != 1:
            raise SystemExit("logger: cannot find the typing import anchor")
        source = source.replace(anchor, "import logging\n" + anchor)
    anchor = "import logging\n"
    index = source.index(anchor) + len(anchor)
    lines = source[index:].splitlines(keepends=True)
    import_count = 0
    for line in lines:
        if line.startswith(("import ", "from ")) or not line.strip():
            import_count += 1
        else:
            break
    insert_at = index + sum(len(line) for line in lines[:import_count])
    return source[:insert_at] + "\n" + LOGGER_BOOTSTRAP + source[insert_at:]


OLD2 = """                raise ValueError(
                    f\"Text block contains image placeholder '{IMAGE_PLACEHOLDER}': \"
                    f\"'{text[:100]}'. Images should be separate content blocks.\"
                )
"""
NEW2 = """                logger.warning(
                    \"encoding_dsv41: sanitizing image placeholder token in a text block \"
                    \"(upstream would raise) # [encoding_dsv41-placeholder-sanitize]\"
                )
                block = dict(block)
                block[\"text\"] = text.replace(IMAGE_PLACEHOLDER, \"[image]\")
"""


def patch_text_block(source: str) -> tuple[str, bool]:
    if source.count(OLD2) != 1:
        return source, False
    return source.replace(OLD2, NEW2), True


def verify(source: str) -> list[str]:
    problems: list[str] = []
    if source.count(NEW1) != 1 or source.count(NEW2) != 1:
        problems.append("complete sanitize anchors do not match; re-derive the patch")
    if RAISE1 in source:
        problems.append("anchor 1 still raises")
    if RAISE2 in source:
        problems.append("anchor 2 still raises")
    if SENT1 not in source:
        problems.append("anchor 1 sanitize missing")
    if SENT2 not in source:
        problems.append("anchor 2 sanitize missing")
    if LOGGER_BOOTSTRAP not in source:
        problems.append("logger bootstrap missing")
    if "import logging\n" not in source:
        problems.append("import logging missing")
    return problems


def main() -> int:
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
    if not target.is_file():
        print(f"ERROR: target not found: {target} (base image changed?)", file=sys.stderr)
        return 1
    source = target.read_text()
    changed: list[str] = []
    source = ensure_logger(source)
    if SENT1 not in source:
        if source.count(OLD1) != 1:
            print(
                f"ERROR: expected exactly 1 match for anchor 1, found {source.count(OLD1)} -- re-derive the patch",
                file=sys.stderr,
            )
            return 1
        source = source.replace(OLD1, NEW1)
        changed.append("anchor1(string content)")
    if SENT2 not in source:
        source, applied = patch_text_block(source)
        if not applied:
            print("ERROR: anchor 2 not found -- base image changed, re-derive the patch", file=sys.stderr)
            return 1
        changed.append("anchor2(text block / tool_result)")
    problems = verify(source)
    if problems:
        print(f"ERROR: post-patch verification failed: {'; '.join(problems)}", file=sys.stderr)
        return 1
    target.write_text(source)
    print(f"patched {target}: {', '.join(changed) or 'logger only'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
