import numpy as np
import torch
import random
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F
from domainbed import networks
import os


GLOBAL_DROPOUT_RATE = 0.5

class MNIST_DROPOUT_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_DROPOUT_CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.dr1 = nn.Dropout(GLOBAL_DROPOUT_RATE)
        self.dr2 = nn.Dropout(GLOBAL_DROPOUT_RATE)
        self.dr3 = nn.Dropout(GLOBAL_DROPOUT_RATE)
        self.dr4 = nn.Dropout(GLOBAL_DROPOUT_RATE)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)
        x = self.dr1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dr2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dr3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.dr4(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


def DropoutClassifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.Dropout(GLOBAL_DROPOUT_RATE),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.Dropout(GLOBAL_DROPOUT_RATE),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)




mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

# Build environments

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
    def torch_xor(a, b):
        return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    gt_labels = labels
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
        'images': (images.float() / 255.).to(device),
        'labels': labels[:, None].to(device),
        'gt_labels': gt_labels[:, None].to(device),
        'colors': colors.to(device)
    }

envs = [
make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
make_environment(mnist_val[0], mnist_val[1], 0.9)
]
envs = [(e['images'], e['labels'], e['gt_labels'], e['colors']) for e in envs]

training_envs = envs[:-1]
test_envs = envs[-1:]
batch_size = 2048

def data_generator(dataset, batch_size=512):
    num_samples = len(dataset[0])
    indices = list(range(num_samples))
    while True:
        random.shuffle(indices)
        num_batches = int(num_samples/batch_size)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield [dataset[0][batch_indices], dataset[1][batch_indices],
                   dataset[2][batch_indices], dataset[3][batch_indices]]


def flatten_reshape(x):
    flat_x = torch.flatten(x, start_dim=1)
    reshaped_x = flat_x.reshape(torch.cat([torch.tensor([1]), torch.tensor(flat_x.shape)]).tolist())
    return reshaped_x





def get_color_label_correlation(labels, colors):
    labels = labels.T[0]
    return np.correlate(labels, colors)/len(labels)


def mean_accuracy(logits, y):
    a = torch.argmax(logits, axis=1)
    b = torch.flatten(y)
    acc = torch.sum((a == b).type(torch.int))/len(a)
    return acc


def train_erm():
    num_epochs = 10000
    num_classes = 2
    input_shape = (2, 28, 28)
    hparams = {'data_augmentation': True,
     'resnet18': False,
     'resnet_dropout': 0.0,
     'class_balanced': False,
     'nonlinear_classifier': False,
     'lr': 0.001,
     'weight_decay': 0.0,
     'batch_size': 64}


    # Get the dropout based featurizer and classifier
    featurizer = networks.Featurizer(input_shape, hparams)
    classifier = networks.Classifier(featurizer.n_outputs, num_classes, hparams['nonlinear_classifier'])
    network = torch.nn.Sequential(featurizer, classifier).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=hparams['lr'], weight_decay=1e-3)

    # Define models for each environment
    train_env_models = []
    for i, env in enumerate(training_envs):
        f = networks.Featurizer(input_shape, hparams)
        c = networks.Classifier(featurizer.n_outputs, num_classes, hparams['nonlinear_classifier'])
        n = torch.nn.Sequential(f, c).to(device)
        train_env_models.append({
            'featurizer': f,
            'classifier': c,
            'network': n,
            'optimizer': torch.optim.Adam(n.parameters(), lr=hparams['lr'], weight_decay=1e-3)
        })


    train_generators = [data_generator(x, batch_size=batch_size) for x in training_envs]
    test_generators = [data_generator(x) for x in test_envs]
    num_train_environments = len(train_generators)
    num_all_envs = num_train_environments + len(test_generators)
    filter_ratio = 0.2

    # num_uncertainty_predictions = 10

    # For each epoch
    for i in range(num_epochs):

        # Train each classifier on its own environment
        for e_i, gen in enumerate(train_generators):
            x, y, t, c = next(gen)
            n = train_env_models[e_i]['network']
            o = train_env_models[e_i]['optimizer']
            preds = n(x)
            env_loss = F.cross_entropy(preds, y.flatten().long())
            o.zero_grad()
            env_loss.backward()
            o.step()

        # Get the most confident predictions in the test environments
        for test_i, test_gen in enumerate(test_generators):
            x, y, t, c = next(test_gen)
            env_preds = []
            for train_i, train_model in enumerate(train_env_models):
                env_preds.append(train_model['network'](x).detach())

            env_preds.append(network(x).detach())

            # Get the samples with most confidence
            env_preds = torch.stack(env_preds)

            variation = torch.mean(torch.var(env_preds, axis=0), axis=1)
            sorted_variation = torch.sort(variation)
            sorted_indices = sorted_variation.indices
            important_indices = sorted_indices[:int(filter_ratio*len(x))]

            silver_labels = torch.argmax(torch.mean(env_preds, axis=0), axis=1).long()
            loss = F.cross_entropy(network(x[important_indices]), silver_labels[important_indices].flatten().long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            t_x, t_y, _, _ = next(test_generators[0])
            print(f'Test Accuracy: {mean_accuracy(network(t_x), t_y.flatten().long())}')

        # if i % 10 == 0:
        #     # Percentage of mislabels in actual batch
        #     p_correct = torch.sum(((combined_y == combined_t).float()))/len(combined_y)
        #     print(f'% correct labels: {p_correct}')
        #     # Percentage filtered correct
        #     p_filtered_correct = torch.sum(((filtered_labels == filtered_true).float()))/len(filtered_labels)
        #     print(f'% correct filtered: {p_filtered_correct}')
        #     # Print the loss
        #     print(f'Loss: {loss.item()}')
        #     # Print the accuracy for the training environments
        #     print(f'Accuracy: {mean_accuracy(network(combined_x), combined_y.flatten().long())}')
        #     # Print the accuracy for the test environments
        #     t_x, t_y, _, _ = next(test_generators[0])
        #     print(f'Test Accuracy: {mean_accuracy(network(t_x), t_y.flatten().long())}')



if __name__ == "__main__":
    # env_detector, env_predictor = train_environment_predictor()
    # print('====')
    train_erm()


