import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import RecurrentRelationalNetwork
from dataset import get_dataset
from plot_utils import draw_sudoku


hidden_dim = 32  # n_node_features
message_dim = 32  # n_edge_features
num_steps = 16  # n_iters
n_nodes = 81
chunksize = 10_000

df = pd.read_csv("./sudoku9000000.csv", chunksize=chunksize)
iterator = iter(df)
next(iterator)
# next(iterator)
# next(iterator)
# next(iterator)
chunk = next(iterator)
# n = int(0.8 * len(first_chunk))
# first_chunk = first_chunk[n:]
trainset, testset = get_dataset(chunk, hidden_dim, batch_size=32)

model = RecurrentRelationalNetwork(message_dim, hidden_dim, num_steps)
graphs = next(iter(trainset))[0]
model(graphs)
model.load_weights("models/model_weights_end.keras")


for graphs, labels in testset:
    pass

logits = model(graphs, return_all_steps=True)
logits = np.array([iter.numpy() for iter in logits])
logits = logits.reshape(num_steps, len(labels), n_nodes, 9)

i = 0
while i < len(labels):
    _labels = labels[i, :].numpy()
    _logits = logits[:, i, :, :]
    i += 1
    n_given = np.isclose(_logits[0], 1, atol=5e-2).sum()
    solved_correctly = np.all(_labels == _logits.argmax(axis=-1)[-1] + 1)
    if solved_correctly: continue
    print(n_given, "given digits in the puzzle")
    print("Solved correctly" if solved_correctly else "Solved incorrectly")

    # for step in range(num_steps):
    #     draw_sudoku(_logits[step], probs=True)
    draw_sudoku(_logits[0], probs=True)
    draw_sudoku(_logits[1], probs=True)
    draw_sudoku(_logits[-1], probs=True)
    draw_sudoku(_labels, probs=False)
    plt.show()
