from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console

console = Console()


def extract_image_refs(text: str) -> list[str]:
    """Extract image references from text.

    Supports formats:
    - #path/to/image.png (simple paths)
    - #"path with spaces/image.png" (quoted paths)
    """
    images: list[str] = []
    for match in re.finditer(r'#"([^"]+)"', text):
        path = match.group(1).strip()
        if path:
            images.append(path)

    for match in re.finditer(r'#([^\s"#]+)', text):
        path = match.group(1).strip()
        if path and not path.startswith('"'):
            images.append(path)

    return images


def validate_image_paths(image_refs: list[str], base_dir: Path) -> tuple[list[str], list[str]]:
    """Validate and copy image files into project directory.

    Copies images from current working directory or source location into base_dir.
    Returns (valid_paths_in_project, error_messages).
    """
    valid_paths: list[str] = []
    errors: list[str] = []

    for ref in image_refs:
        src_path = Path(ref)
        if not src_path.is_absolute():
            src_path = Path.cwd() / src_path

        if not src_path.exists():
            errors.append(f"Image not found at source: {ref}")
            continue

        if not src_path.is_file():
            errors.append(f"Not a file: {ref}")
            continue

        try:
            if src_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
                errors.append(f"Unsupported format: {ref}")
                continue

            dest_path = base_dir / ref
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_path, dest_path)
            valid_paths.append(ref)

            try:
                img = Image.open(dest_path)
                max_size = 512
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                if dest_path.suffix.lower() in {".jpg", ".jpeg"}:
                    img.save(dest_path, quality=85, optimize=True)
                elif dest_path.suffix.lower() == ".webp":
                    img.save(dest_path, quality=85)
                else:
                    img.save(dest_path, optimize=True)

            except ImportError:
                errors.append(f"Pillow not installed; skipping image scaling for {ref}")
            except Exception as e:
                errors.append(f"Warning: Could not scale image {ref}: {e}")

        except Exception as e:
            errors.append(f"Error copying {ref}: {e}")

    return valid_paths, errors


def encode_image_to_base64(path: Path) -> str:
    """Read image file and encode to base64 data URL."""
    with open(path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")

    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(suffix, "image/png")
    return f"data:{mime_type};base64,{b64}"


def build_vision_message_blocks(text: str, image_refs: list[str], base_dir: Path) -> list[dict[str, Any]]:
    """Build message content blocks for vision API.

    Returns list of blocks: [{"type": "text", ...}, {"type": "image_url", ...}]
    """
    content_blocks: list[dict[str, Any]] = []

    if text.strip():
        content_blocks.append(
            {
                "type": "text",
                "text": text,
            }
        )

    for ref in image_refs:
        try:
            img_path = base_dir / ref
            if img_path.exists() and img_path.is_file():
                img_url = encode_image_to_base64(img_path)
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img_url},
                    }
                )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load image {ref}: {e}[/yellow]")

    return content_blocks
