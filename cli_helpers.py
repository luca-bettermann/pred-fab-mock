"""CLI helper functions: inline plot display and physics randomization."""

import base64
import os
import sys


def show_plot(path: str, inline: bool = True) -> None:
    """Display a plot: save path always printed, inline image if requested."""
    if not inline:
        print(f"  Plot: {path}")
        return

    print()
    with open(path, "rb") as f:
        img_data = f.read()
    b64 = base64.b64encode(img_data).decode("ascii")
    sys.stdout.write(f"\033]1337;File=inline=1;size={len(img_data)}:{b64}\a\n")
    sys.stdout.flush()
    print(f"  Plot: {path}")


def show_plot_with_header(path: str, title: str, inline: bool = True) -> None:
    """Show plot with a descriptive header line."""
    if inline:
        print(f"\n  \033[1m{title}\033[0m")
    show_plot(path, inline=inline)


def ensure_plot_dir() -> str:
    """Create and return the plots directory."""
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir
