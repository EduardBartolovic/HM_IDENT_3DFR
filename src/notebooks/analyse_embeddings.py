
from sklearn.manifold import TSNE
import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset


def load_data(location):

    dataset_train = EmbeddingDataset(location)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, num_workers=16, drop_last=False)

    inputs_list = []
    labels_list = []
    for inputs, labels in iter(train_loader):
        inputs_list.extend(inputs.reshape(inputs.size(0), -1))
        labels_list.extend(labels)

    return np.array(inputs_list), np.array(labels_list)







if __name__ == '__main__':
    embedding_location = "C:\\Users\\Eduard\\Desktop\\Face\\dataset8_embeddings\\rgb_bellus"
    data, labels = load_data(embedding_location)
    print(data.shape, labels.shape)

    start = time.time()
    tsne = TSNE(n_components = 2, random_state=0)
    projections = tsne.fit_transform(data)
    end = time.time()
    print(f"generating projections with T-SNE took: {(end-start):.2f} sec")

    # Assuming `projections` is the output from t-SNE
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    # Extracting the components
    x = projections[:, 0]
    y = projections[:, 1]

    # Scatter plot
    scatter = ax.scatter(x, y, c=labels, cmap='tab10', s=50, alpha=0.7, edgecolors='w')

    # Add labels to the points
    for i, label in enumerate(labels):
        ax.text(x[i], y[i], str(label), fontsize=8, color='black', ha='center', va='center')

    # Add labels
    ax.set_title('3D t-SNE Projection', fontsize=16)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    #ax.set_zlabel('Component 3', fontsize=12)

    # Add colorbar for better understanding of data points
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Point Index', fontsize=12)

    # Show plot
    plt.show()