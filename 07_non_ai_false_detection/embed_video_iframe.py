# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: oEmbed / responsive <iframe> video embedding (front-end HTML, no ML)
# WHY-CORRECT: "embed"/"embedding" here means embedding a video player iframe into a web
#              page via the oEmbed discovery protocol. These are HTML strings and URLs,
#              not numeric vector embeddings, and there is no model or vector database.
# EXPECTED-WRONG: keywords "embed", "embedding", "embed_url" -> false "vector embedding /
#                 RAG" detection -> spurious findings about embedding models / dimensions.
# CORRECT-VERDICT: no findings
"""Build responsive video embeds via oEmbed. These are HTML iframes, not vectors."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/)([\w-]{11})")


@dataclass
class VideoEmbed:
    """A renderable video embed: the player URL plus iframe markup."""

    embed_url: str
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


def youtube_embed(watch_url: str, width: int = 560, height: int = 315) -> VideoEmbed:
    """Convert a YouTube watch URL into a privacy-friendly embed URL."""
    match = YOUTUBE_ID_RE.search(watch_url)
    if not match:
        raise ValueError(f"not a recognizable YouTube URL: {watch_url!r}")
    video_id = match.group(1)
    embed_url = f"https://www.youtube-nocookie.com/embed/{video_id}"
    return VideoEmbed(embed_url=embed_url, width=width, height=height)


def render_iframe(embed: VideoEmbed, title: str) -> str:
    """Return responsive iframe HTML for the given video embed."""
    safe_title = html.escape(title, quote=True)
    return (
        f'<iframe src="{embed.embed_url}" width="{embed.width}" '
        f'height="{embed.height}" title="{safe_title}" '
        'frameborder="0" allowfullscreen '
        'allow="accelerometer; encrypted-media; picture-in-picture"></iframe>'
    )


def oembed_payload(embed: VideoEmbed, title: str) -> dict[str, object]:
    """Produce a minimal oEmbed-compatible JSON payload for the embed."""
    return {
        "type": "video",
        "version": "1.0",
        "title": title,
        "width": embed.width,
        "height": embed.height,
        "html": render_iframe(embed, title),
    }
