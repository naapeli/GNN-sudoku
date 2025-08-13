import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
from functools import partial


def edges():
    source_indices = []
    destination_indices = []
    for source in range(81):
        source_ids = []
        target_ids = []
        row = source // 9
        column = source % 9
        for target in range(9 * row, 9 * row + 9):
            if target == source or target in target_ids: continue
            source_ids.append(source)
            target_ids.append(target)
            
        for target in range(column, 81, 9):
            if target == source or target in target_ids: continue
            source_ids.append(source)
            target_ids.append(target)
        
        block_row = row // 3
        block_column = column // 3
        block_start = 9 * 3 * block_row + 3 * block_column
        for i in range(0, 3):
            for j in range(0, 3):
                target = block_start + 9 * j + i
                if target == source or target in target_ids: continue
                source_ids.append(source)
                target_ids.append(target)
        source_indices.extend(source_ids)
        destination_indices.extend(target_ids)
    return tf.convert_to_tensor(source_indices, dtype=tf.int32), tf.convert_to_tensor(destination_indices, dtype=tf.int32)


def sudoku_to_graph_tensor(puzzle, solution, hidden_dim):
    clues = tf.reshape(puzzle, [-1])
    clues_one_hot = tf.one_hot(clues, depth=10)[:, 1:]  # only 9 features and if all of them are zero, the digit is unknown
    n_nodes = 81
    assert clues_one_hot.shape == (n_nodes, 9), "The length of clues_one_hot should be the amount of nodes (81 cells) in a graph."

    source_indices, destination_indices = edges()

    edge_features = {}
    edge_set = tfgnn.EdgeSet.from_fields(
        sizes=tf.constant([len(source_indices)]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=("nodes", source_indices),
            target=("nodes", destination_indices),
        ),
        features=edge_features
    )

    node_set = tfgnn.NodeSet.from_fields(
        sizes=tf.constant([n_nodes]),
        features={
            "clues_one_hot": clues_one_hot,
            "hidden_state": tf.zeros((81, hidden_dim))
        }
    )

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={"nodes": node_set},
        edge_sets={"edges": edge_set}
    )

    labels = tf.reshape(solution, [-1])

    return graph, labels


def get_dataset(df_chunk, hidden_dim):
    puzzle_array = df_chunk["puzzle"].apply(lambda x: list(map(int, x))).to_list()
    puzzle_array = np.array(puzzle_array, dtype=np.int32)
    puzzle_array = puzzle_array.reshape(len(df_chunk), 9, 9)
    puzzle_array = tf.convert_to_tensor(puzzle_array)
    solution_array = df_chunk["solution"].apply(lambda x: list(map(int, x))).to_list()
    solution_array = np.array(solution_array, dtype=np.int32)
    solution_array = solution_array.reshape(len(df_chunk), 9, 9)
    solution_array = tf.convert_to_tensor(solution_array)

    n = int(0.7 * len(puzzle_array))
    batch_size = 32
    trainset = tf.data.Dataset.from_tensor_slices((puzzle_array[:n], solution_array[:n]))
    trainset = trainset.map(partial(sudoku_to_graph_tensor, hidden_dim=hidden_dim))
    trainset = trainset.batch(batch_size)
    trainset = trainset.map(lambda g, y: (g.merge_batch_to_components(), y))
    testset = tf.data.Dataset.from_tensor_slices((puzzle_array[n:], solution_array[n:]))
    testset = testset.map(partial(sudoku_to_graph_tensor, hidden_dim=hidden_dim))
    testset = testset.batch(batch_size)
    testset = testset.map(lambda g, y: (g.merge_batch_to_components(), y))

    return trainset, testset
