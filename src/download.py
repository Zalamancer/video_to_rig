"""Download video from YouTube using yt-dlp."""
import subprocess
import os
import tempfile


def download_video(url, output_dir=None):
    """Download a YouTube video and return the local file path."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="v2r_")

    output_path = os.path.join(output_dir, "video.mp4")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    if not os.path.exists(output_path):
        # yt-dlp may add extension — look for any mp4
        for f in os.listdir(output_dir):
            if f.endswith(".mp4"):
                return os.path.join(output_dir, f)
        raise FileNotFoundError("Downloaded video not found")

    return output_path
