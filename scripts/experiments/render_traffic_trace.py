"""Render route-action traces as lightweight top-down traffic figures."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_steps(trace_dir: Path) -> list[dict[str, Any]]:
    path = trace_dir / "steps.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing trace file: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "step":
                records.append(record)
    if not records:
        raise ValueError(f"no step records in {path}")
    return records


def _point(transform: dict[str, Any]) -> tuple[float, float]:
    loc = (transform or {}).get("location", {})
    return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))


def _route_index(routes: list[dict[str, Any]]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    by_pair = {
        (int(route.get("junction_id", -1)), int(route["route_id"])): route
        for route in routes
    }
    by_id: dict[int, dict[str, Any]] = {}
    for route in routes:
        by_id.setdefault(int(route["route_id"]), route)
    return by_pair, by_id


def _preference_color(preference: list[Any]) -> str:
    if len(preference) >= 2:
        efficiency = float(preference[0])
        risk = float(preference[1])
        if efficiency > risk:
            return "#1f77b4"
        if risk > efficiency:
            return "#d62728"
    return "#2ca02c"


def _intent_label(agent: dict[str, Any]) -> str:
    route_intent = agent.get("route_intent") or {}
    intent = route_intent.get("selected_intent")
    if intent:
        return str(intent).replace("keep-current", "keep")
    route_type = agent.get("route_type")
    return str(route_type or "")


def _select_step(
    steps: list[dict[str, Any]],
    *,
    episode: int | None,
    step: int | None,
) -> dict[str, Any]:
    if episode is None and step is None:
        return steps[-1]
    candidates = steps
    if episode is not None:
        candidates = [record for record in candidates if int(record.get("episode", -1)) == episode]
    if not candidates:
        raise ValueError(f"no records for episode={episode}")
    if step is None:
        return candidates[-1]
    return min(candidates, key=lambda record: abs(int(record.get("step", -1)) - step))


def render_frame(
    trace_dir: Path,
    out_path: Path,
    *,
    episode: int | None = None,
    step: int | None = None,
    title: str | None = None,
) -> Path:
    routes = _load_json(trace_dir / "routes.json", [])
    steps = _load_steps(trace_dir)
    frame = _select_step(steps, episode=episode, step=step)
    route_by_pair, route_by_id = _route_index(routes)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    ax.set_facecolor("#eef0ea")

    xs: list[float] = []
    ys: list[float] = []

    for route in routes:
        x0, y0 = _point(route.get("spawn_tf", {}))
        x1, y1 = _point(route.get("dest_tf", {}))
        xs.extend([x0, x1])
        ys.extend([y0, y1])
        ax.plot([x0, x1], [y0, y1], color="#b7b7b7", linewidth=0.6, alpha=0.25, zorder=1)

    selected_routes = set()
    for agent in frame.get("agents", []):
        route_intent = agent.get("route_intent") or {}
        if route_intent.get("selected_route_id") is not None:
            selected_routes.add((
                int(route_intent.get("selected_junction_id", -1)),
                int(route_intent["selected_route_id"]),
            ))

    for junction_id, route_id in selected_routes:
        route = route_by_pair.get((junction_id, route_id)) or route_by_id.get(route_id)
        if not route:
            continue
        x0, y0 = _point(route.get("spawn_tf", {}))
        x1, y1 = _point(route.get("dest_tf", {}))
        ax.plot([x0, x1], [y0, y1], color="#111111", linewidth=1.2, alpha=0.45, zorder=2)

    for agent in frame.get("agents", []):
        x = float(agent.get("x", 0.0))
        y = float(agent.get("y", 0.0))
        xs.append(x)
        ys.append(y)
        preference = list(agent.get("preference") or [])
        color = _preference_color(preference)
        ax.add_patch(Circle((x, y), 1.45, facecolor=color, edgecolor="#0b1f3a", linewidth=1.2, zorder=4))
        ax.text(
            x,
            y,
            _intent_label(agent)[:7],
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            zorder=5,
        )
        ax.text(
            x,
            y - 2.3,
            f"{float(agent.get('target_speed', 0.0)):.0f}",
            ha="center",
            va="top",
            fontsize=5.8,
            color="#1b1b1b",
            zorder=5,
        )

    if xs and ys:
        pad = 12.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = title or (
        f"episode {int(frame.get('episode', 0))}, step {int(frame.get('step', 0))}"
    )
    ax.set_title(title, fontsize=11, loc="left", pad=4)
    ax.text(
        0.99,
        0.02,
        "color: efficiency / risk / balanced   label: route intent   number: target speed",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#333333",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--watch", action="store_true",
                        help="Continuously overwrite --out with the latest frame.")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    while True:
        render_frame(
            args.trace_dir,
            args.out,
            episode=args.episode,
            step=args.step,
            title=args.title,
        )
        print(f"[trace-render] wrote {args.out}", flush=True)
        if not args.watch:
            break
        time.sleep(max(0.2, float(args.interval)))


if __name__ == "__main__":
    main()
