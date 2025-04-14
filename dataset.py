import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd
import numpy as np


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


class DataGenerator:
    def __init__(self, data, batch_size=32, subset="train"):
        self.data = data
        self.batch_size = batch_size
        self.subset = subset
        self.indices = np.arange(len(self.data))
        self.source_indices, self.destination_indices = edges()
    
    def _get_graph(self, samples):
        batch_size = len(samples)
        src_ids = tf.concat([src_ids + 81 * i for i in range(batch_size)], axis=0)
        dst_ids = tf.concat([dst_ids + 81 * i for i in range(batch_size)], axis=0)
        adjacency = tfgnn.Adjacency.from_indices(
            source=src_ids,
            target=dst_ids
        )
        graph_tensor = tfgnn.GraphTensor.from_pieces(
            node_sets={'nodes': tfgnn.NodeSet.from_fields()},
            edge_sets={'edges': tfgnn.EdgeSet.from_fields(adjacency=adjacency)}
        )
        return graph_tensor

    def _process_data(self, idx):
        quiz = self.data['quizzes'].iloc[idx]
        X = (np.array(list(map(int, list(quiz)))).reshape((9, 9, 1)) / 9) - 0.5
        if self.subset == 'train':
            solution = self.data['solutions'].iloc[idx]
            y = np.array(list(map(int, list(solution)))).reshape((81, 1)) - 1
            return X, y
        return X, None

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def _generator(self):
        for i in range(0, len(self.data), self.batch_size):
            batch_indexes = self.indices[i:i + self.batch_size]
            batch_data = [self._process_data(idx) for idx in batch_indexes]
            X = np.array([item[0] for item in batch_data])
            if self.subset == 'train':
                y = np.array([item[1] for item in batch_data])
                yield X, y
            else:
                yield X

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self._generator, output_signature=(
            tf.TensorSpec(shape=(self.batch_size, 9, 9, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(self.batch_size, 81, 1), dtype=tf.float32)
        ))
        
        dataset = dataset.shuffle(buffer_size=len(self.data))
        return dataset.batch(self.batch_size)




if __name__ == "__main__":
    data = pd.read_csv("./sudoku/sudoku.csv")
    n = len(data)
    train = data[:int(0.7 * n)]
    test = data[int(0.7 * n):]
    trainloader = DataGenerator(train, batch_size=32, subset="train").get_dataset()
    testloader = DataGenerator(test, batch_size=32, subset="test").get_dataset()
