import matplotlib.pyplot as plt
import numpy as np


def show_proba(proba, r, c, ax):
    cm = plt.cm.Reds
    ix = proba.argmax()
    if proba[ix] > 0.9:
        px, py = c+0.5, r+0.5
        ax.text(px, py, ix.item() + 1, ha="center", va="center", fontsize=24)
    else:
        for d in range(9):
            dx = dy = 1/6
            px = c + dx + (d // 3)*(2*dx)
            py = r + dy + (d % 3)*(2*dy)
            p = proba[d]
            ax.fill(
                [px-dx, px+dx, px+dx, px-dx, px-dx], [py-dy, py-dy, py+dy, py+dy, py-dy],
                color=cm(int(p*256))
            )
            ax.text(px, py, d + 1, ha="center", va="center", fontsize=8)

def draw_sudoku(x, probs=True):
    fig, ax = plt.subplots(1, figsize=(7,7))
    ax.set(
        xlim=(0, 9), ylim=(9, 0),
        xticks=np.arange(10), xticklabels=[],
        yticks=np.arange(10), yticklabels=[]
    )
    ax.grid(True, which="major", linewidth=2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(3))
    ax.yaxis.set_major_locator(plt.MultipleLocator(3))
    ax.tick_params(which="major", length=0)

    ax.grid(True, which="minor")
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(which="minor", length=0)

    if probs:
        for r in range(9):
            for c in range(9):
                i = 9 * r + c
                show_proba(x[i], r, c, ax)
    else:
        for r in range(9):
            for c in range(9):
                i = 9 * r + c
                px, py = c+0.5, r+0.5
                ax.text(px, py, int(x[i]), ha="center", va="center", fontsize=24)
