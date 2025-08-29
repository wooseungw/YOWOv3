#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SequenceKey:
    participant: str  # e.g., P01
    session: str      # e.g., 20201102PM_S02_H120_P01
    action_id: str    # e.g., A001
    base_name: str    # e.g., A001_P001_C0li_C0li


def iter_skeleton_csvs(joint_root: str, participants: Optional[List[str]] = None):
    for p in sorted(os.listdir(joint_root)):
        if not p.startswith("P"):
            continue
        if participants and p not in participants:
            continue
        p_dir = os.path.join(joint_root, p)
        if not os.path.isdir(p_dir):
            continue
        for session in sorted(os.listdir(p_dir)):
            s_dir = os.path.join(p_dir, session)
            if not os.path.isdir(s_dir):
                continue
            for fname in sorted(os.listdir(s_dir)):
                if not fname.lower().endswith(".csv"):
                    continue
                base = fname[:-4]
                if "_" in base:
                    action = base.split("_", 1)[0]
                else:
                    action = base
                yield (
                    SequenceKey(participant=p, session=session, action_id=action, base_name=base),
                    os.path.join(s_dir, fname),
                )


def safe_count_lines(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # subtract 1 for header if present; we'll correct to non-negative
            count = sum(1 for _ in f)
        return max(0, count - 1)
    except Exception as e:
        print(f"WARN: failed counting lines for {path}: {e}", file=sys.stderr)
        return None


def count_bodyindex_frames(bodyindex_dir: str) -> Optional[int]:
    if not os.path.isdir(bodyindex_dir):
        return None
    try:
        return sum(1 for n in os.listdir(bodyindex_dir) if n.lower().endswith(".png"))
    except Exception as e:
        print(f"WARN: failed listing {bodyindex_dir}: {e}", file=sys.stderr)
        return None


def analyze(root: str, participants_arg: str, limit: Optional[int] = None) -> None:
    joint_root = os.path.join(root, "JointCSV(P01-P20)")
    rgb_root = os.path.join(root, "RGB(P01-P20)")
    bi_root = os.path.join(root, "BodyIndex(P01-P20)")

    if not os.path.isdir(joint_root):
        print(f"ERROR: JointCSV root not found: {joint_root}")
        sys.exit(1)

    participants = None
    if participants_arg and participants_arg.lower() != "all":
        participants = [p.strip() for p in participants_arg.split(",") if p.strip()]

    participants_seen = set()
    actions_seen = set()
    sessions_per_p: Dict[str, set] = {}

    seq_count = 0
    missing_rgb = 0
    missing_bodyindex = 0
    skel_frames: List[int] = []
    bi_frames: List[int] = []

    examples_checked = 0
    for key, csv_path in iter_skeleton_csvs(joint_root, participants):
        participants_seen.add(key.participant)
        actions_seen.add(key.action_id)
        sessions_per_p.setdefault(key.participant, set()).add(key.session)

        seq_count += 1

        # skeleton frames
        n_frames = safe_count_lines(csv_path)
        if n_frames is not None:
            skel_frames.append(n_frames)

        # RGB existence
        rgb_path = os.path.join(rgb_root, key.participant, key.session, key.base_name + ".mp4")
        if not os.path.isfile(rgb_path):
            missing_rgb += 1

        # BodyIndex existence + frame count
        bi_dir = os.path.join(bi_root, key.participant, key.session, f"[bodyindex]{key.base_name}")
        if os.path.isdir(bi_dir):
            n_bi = count_bodyindex_frames(bi_dir)
            if n_bi is not None:
                bi_frames.append(n_bi)
        else:
            missing_bodyindex += 1

        if limit is not None:
            examples_checked += 1
            if examples_checked >= limit:
                break

    # Summaries
    print("ETRI Dataset Analysis")
    print(f"root: {os.path.abspath(root)}")
    print("")
    print(f"participants_scanned: {len(participants_seen)} -> {sorted(participants_seen)}")
    print("sessions_per_participant:")
    for p in sorted(sessions_per_p):
        print(f"  {p}: {len(sessions_per_p[p])} sessions -> {sorted(sessions_per_p[p])}")
    print("")
    print(f"unique_actions: {len(actions_seen)} -> sample: {sorted(actions_seen)[:10]}{'...' if len(actions_seen) > 10 else ''}")
    print(f"sequences_count (skeleton CSVs scanned): {seq_count}")
    print(f"missing_rgb_videos: {missing_rgb}")
    print(f"missing_bodyindex_dirs: {missing_bodyindex}")

    def summarize(name: str, arr: List[int]):
        if not arr:
            print(f"{name}: no data")
            return
        s = sorted(arr)
        total = len(s)
        mean = sum(s) / total
        med = s[total // 2] if total % 2 == 1 else (s[total // 2 - 1] + s[total // 2]) / 2
        print(f"{name}: count={total}, min={s[0]}, max={s[-1]}, mean={mean:.1f}, median={med}")

    print("")
    summarize("skeleton_frames_per_sequence", skel_frames)
    summarize("bodyindex_frames_per_sequence", bi_frames)


def main():
    parser = argparse.ArgumentParser(description="Summarize ETRI folder contents")
    parser.add_argument("--root", default="etri", help="Path to ETRI root folder")
    parser.add_argument(
        "--participants",
        default="P01",
        help="Comma-separated list (e.g., P01,P02) or 'all'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of sequences to scan (for speed)",
    )
    args = parser.parse_args()

    analyze(args.root, args.participants, limit=args.limit)


if __name__ == "__main__":
    main()

