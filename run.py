#!/usr/bin/env python3
"""
Video to Rig — Extract human motion from video and save as BVH animation.

Usage:
    python run.py "https://youtube.com/watch?v=..."           # YouTube URL
    python run.py my_video.mp4                                 # local file
    python run.py video.mp4 -o output/dance.bvh --fps 30      # options
"""
import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Extract humanoid motion from video and export as BVH"
    )
    parser.add_argument("input", help="YouTube URL or path to a local video file")
    parser.add_argument("-o", "--output", default="output/animation.bvh",
                        help="Output BVH path (default: output/animation.bvh)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS for the animation (default: 30)")
    parser.add_argument("--smooth", type=float, default=0.5,
                        help="Smoothing 0-1, higher = smoother (default: 0.5)")
    parser.add_argument("--start", type=float, default=0,
                        help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (0 = whole video)")
    parser.add_argument("--scale", type=float, default=100,
                        help="Scale factor, 100 = metres to cm (default: 100)")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: obtain video ──────────────────────────────────────────────
    is_url = args.input.startswith(("http://", "https://", "www."))
    if is_url:
        from src.download import download_video
        print("[1/5] Downloading video …")
        t0 = time.time()
        video_path = download_video(args.input)
        print(f"       Done in {time.time() - t0:.1f}s  →  {video_path}")
    else:
        video_path = args.input
        if not os.path.isfile(video_path):
            sys.exit(f"Error: file not found: {video_path}")
        print(f"[1/5] Using local video: {video_path}")

    # ── Step 2: extract poses (2D + 3D) ──────────────────────────────────
    from src.extract import extract_poses
    print(f"[2/5] Extracting poses (MediaPipe, target {args.fps} FPS) …")
    t0 = time.time()
    lm_3d, lm_2d, img_dims, actual_fps, n_frames = extract_poses(
        video_path,
        target_fps=args.fps,
        start_time=args.start,
        duration=args.duration,
    )
    print(f"       {n_frames} frames extracted in {time.time() - t0:.1f}s")

    if n_frames == 0:
        sys.exit("Error: no poses detected — is there a person visible in the video?")

    # ── Step 3: recover global body orientation via PnP ──────────────────
    from src.global_orient import (estimate_global_rotations,
                                   smooth_rotations,
                                   apply_global_rotations)
    print("[3/5] Recovering global body orientation (PnP) …")
    rotations = estimate_global_rotations(lm_3d, lm_2d, img_dims)
    rotations = smooth_rotations(rotations, window=7)
    lm_oriented = apply_global_rotations(lm_3d, rotations)
    print(f"       Global orientation recovered for {len(rotations)} frames")

    # ── Step 4: smooth ────────────────────────────────────────────────────
    from src.smooth import smooth_landmarks
    print(f"[4/5] Smoothing (factor={args.smooth}) …")
    lm_oriented = smooth_landmarks(lm_oriented, factor=args.smooth)
    lm_3d_smooth = smooth_landmarks(lm_3d, factor=args.smooth)

    # ── Step 5: export ──────────────────────────────────────────────────
    from src.bvh import landmarks_to_bvh
    from src.json_export import landmarks_to_json

    print("[5/5] Exporting …")
    landmarks_to_bvh(lm_3d_smooth, args.output, fps=args.fps, scale=args.scale)

    json_path = args.output.rsplit(".", 1)[0] + ".json"
    landmarks_to_json(lm_oriented, json_path, fps=args.fps, scale=args.scale)

    print(f"\n  BVH saved to: {args.output}")
    print(f"  JSON saved to: {json_path}  (globally oriented 3D)")
    print(f"  Open viewer/index.html to preview.")


if __name__ == "__main__":
    main()
