import torch
import torchvision
import numpy as np
from copy import deepcopy
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

### Hyperparameters
val_split = 0.1
unused_size = 0.99
unlabelled_size = 0.99
lr = 0.0005
batch_size = 64
num_epochs = 100
data_iterations = 100
label_iterations = 150
torch.manual_seed(42)

#Function made by Rasmus Hannibal Tirsgaard
def transfer_unlabelled_to_labeled(unused_dataset, train_dataset, indexes):
    # Convert indexes to boolean mask
    indexes = torch.tensor([i in indexes for i in range(len(unused_dataset.targets))])
    try:
        train_dataset.targets = torch.cat([train_dataset.targets, unused_dataset.targets[indexes]])
    except:
        train_dataset.targets = torch.cat([torch.as_tensor(train_dataset.targets), torch.as_tensor(unused_dataset.targets)[indexes]]).cpu().numpy()
    
    try:
        train_dataset.data = torch.cat([train_dataset.data, unused_dataset.data[indexes]])
    except:
        train_dataset.data = torch.cat([torch.as_tensor(train_dataset.data), torch.as_tensor(unused_dataset.data)[indexes]]).cpu().numpy()
    unused_dataset.targets = unused_dataset.targets[~indexes]
    unused_dataset.data = unused_dataset.data[~indexes]

    return train_dataset, unused_dataset

def transfer_unused_to_labeled(unused_dataset, train_dataset, indexes):
    # Convert indexes to boolean mask
    indexes = torch.tensor([i in indexes for i in range(len(unused_dataset.targets))])
    try:
        train_dataset.targets = torch.cat([train_dataset.targets, unused_dataset.targets[indexes]])
    except:
        train_dataset.targets = torch.cat([torch.as_tensor(train_dataset.targets), torch.as_tensor(unused_dataset.targets)[indexes]]).cpu().numpy()
    
    try:
        train_dataset.data = torch.cat([train_dataset.data, unused_dataset.data[indexes]])
    except:
        train_dataset.data = torch.cat([torch.as_tensor(train_dataset.data), torch.as_tensor(unused_dataset.data)[indexes]]).cpu().numpy()
    unused_dataset.targets = unused_dataset.targets[~indexes]
    unused_dataset.data = unused_dataset.data[~indexes]

    return train_dataset, unused_dataset

#Function made by Rasmus Hannibal Tirsgaard
def validate_model(model, val_loader, device):

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

#Function made by Rasmus Hannibal Tirsgaard
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, val_interval=1):
    accuracies = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            try:
                loss = criterion(outputs, labels)
            except:
                labels = labels.long()
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % val_interval == 0:
            val_accuracy = validate_model(model, val_loader, device)
            accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}, Accuracy: {val_accuracy:.2f}%')
    return accuracies

frac = 0.01

#Function made by Rasmus Hannibal Tirsgaard
def label_iteration_uncertanty_sampling(model, train_dataset, unlabelled_dataset, device, top_frac=frac):
    # Use model to label all images in validation set
    model.eval()
    predictions = []
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for images, _ in tqdm(unlabelled_loader):
            images = images.to(device)
            outputs = model(images).softmax(dim=1)
            predictions.extend(outputs.detach().cpu().numpy())

    predictions = torch.tensor(predictions)
    # Find top % of images with lowest top-confidence
    top_percent = int(top_frac * len(predictions))
    _, top_indices = predictions.max(-1)[0].topk(top_percent, largest=False)
    print(f"Adding {len(top_indices)} images to training set")
    train_dataset, unlabelled_dataset = transfer_unlabelled_to_labeled(unlabelled_dataset, train_dataset, top_indices)
    
    return train_dataset, unlabelled_dataset

def label_iteration_margin_based(model, train_dataset, unlabelled_dataset, device, top_frac=frac):
    # Set the model to evaluation mode
    model.eval()
    predictions = []
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for images, _ in tqdm(unlabelled_loader):
            images = images.to(device)
            # Get the class probabilities using softmax
            outputs = model(images).softmax(dim=1)
            predictions.extend(outputs.cpu().numpy())

    # Convert predictions to a tensor for easier manipulation
    predictions = torch.tensor(predictions)

    # Compute the margin: difference between the top two probabilities for each data point
    sorted_probs, _ = predictions.sort(dim=1, descending=True)
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]  # Top class probability minus second highest

    # Select the indices of the samples with the smallest margins
    top_percent = int(top_frac * len(margins))
    _, top_indices = margins.topk(top_percent, largest=False)  # Smallest margins, hence least confident

    print(f"Adding {len(top_indices)} images to training set")

    # Transfer the selected samples from unlabelled to labelled dataset
    train_dataset, unlabelled_dataset = transfer_unlabelled_to_labeled(unlabelled_dataset, train_dataset, top_indices)

    return train_dataset, unlabelled_dataset


def label_iteration_BADL(model, train_dataset, unlabelled_dataset, device, top_frac=frac):
    # Put model in evaluation mode
    model.eval()
    
    # DataLoader for unlabeled dataset
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    all_entropies = []
    avg_predictions = None
    num_epochs=10

    with torch.no_grad():
        # Perform multiple stochastic forward passes for MC Dropout
        for _ in range(num_epochs):
            predictions = []

            for images, _ in tqdm(unlabelled_loader, desc="MC Dropout passes"):
                images = images.to(device)
                outputs = model(images).softmax(dim=1)
                predictions.append(outputs.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)

            if avg_predictions is None:
                avg_predictions = predictions / num_epochs
            else:
                avg_predictions += predictions / num_epochs

            # Compute entropy for this pass
            pass_entropies = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
            all_entropies.append(pass_entropies)

    # Average entropy across passes
    all_entropies = np.stack(all_entropies, axis=1)
    avg_entropy = np.mean(all_entropies, axis=1)

    # Entropy of the average predictions
    avg_pred_entropy = -np.sum(avg_predictions * np.log(avg_predictions + 1e-10), axis=1)

    # Compute BADL scores
    badl_scores = avg_pred_entropy - avg_entropy

    # Select top samples based on BADL scores
    top_k = int(len(badl_scores) * top_frac)
    top_indices = np.argsort(badl_scores)[-top_k:]

    # Transfer top samples from unlabeled to training dataset
    print(f"Adding {len(top_indices)} images to training set")
    train_dataset, unlabelled_dataset = transfer_unlabelled_to_labeled(unlabelled_dataset, train_dataset, top_indices)

    return train_dataset, unlabelled_dataset


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def label_iteration_cluster_KMEANS(model, train_dataset, unlabelled_dataset, device, n_clusters=12, top_frac=0.01):
    # Step 1: Extract features for the unlabelled dataset using the model
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Use all layers except the final FC
    features = []
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for images, _ in tqdm(unlabelled_loader):
            images = images.to(device)
            feature = feature_extractor(images)  # Extract features
            features.append(feature.view(feature.size(0), -1).cpu().numpy())  # Flatten and move to CPU

    features = np.vstack(features)  # Combine all features into a single array

    # Step 2: Apply K-Means clustering to the extracted features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    # Step 3: Compute distances to centroids
    distances = cdist(features, centroids, metric='euclidean')  # Shape: (N_samples, n_clusters)

    # Step 4: Identify points near cluster boundaries
    margin_distances = distances[np.arange(len(features)), cluster_labels]  # Distance to the assigned cluster centroid
    second_distances = np.partition(distances, 1, axis=1)[:, 1]  # Distance to the second closest centroid
    margins = second_distances - margin_distances  # Margin between closest and second closest centroid

    # Select points with smallest margins (near boundaries)
    top_percent = int(top_frac * len(features))
    boundary_indices = np.argsort(margins)[:top_percent]

    print(f"Adding {len(boundary_indices)} boundary points to the training set")

    # Step 5: Transfer the selected points to the training dataset
    train_dataset, unlabelled_dataset = transfer_unlabelled_to_labeled(unlabelled_dataset, train_dataset, boundary_indices)

    return train_dataset, unlabelled_dataset