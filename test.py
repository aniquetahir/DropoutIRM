import numpy as np
import torch
import random
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F


mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

# Build environments

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
        'images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda(),
        'gt_labels': gt_labels[:, None].cuda(),
        'colors': colors.cuda()
    }

envs = [
make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
make_environment(mnist_val[0], mnist_val[1], 0.9)
]
envs = [(e['images'], e['labels'], e['gt_labels'], e['colors']) for e in envs]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


training_envs = envs[:-1]
test_envs = envs[-1:]

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


def train_environment_predictor():
    num_epochs = 3000

    input_shape = training_envs[0][0][0].shape
    flattened_input_shape = np.prod(list(input_shape))
    num_hidden_features = 64
    num_domains = len(training_envs)

    # Define the sequence classification model
    environment_detector = torch.nn.GRU(input_size = flattened_input_shape,
                                        hidden_size = num_hidden_features,
                                        num_layers = 1,
                                        batch_first = True,
                                        bidirectional = True).to(device)

    environment_predictor = torch.nn.Linear(2 * num_hidden_features, num_domains).to(device)

    all_params = list(environment_detector.parameters()) + list(environment_predictor.parameters())

    optimizer = torch.optim.Adam(
        all_params,
        lr=0.005
    )
    train_generators = [data_generator(x) for x in training_envs]

    for j in range(num_epochs):
        # For each training environment
        pred_envs = []
        gt_envs = []

        for i, env in enumerate(training_envs):
            # Get the sequence of the samples
            env_batch_x, _, _, _ = next(train_generators[i])
            num_env_samples = len(env_batch_x)

            gt = torch.ones(num_env_samples).to(device) * i

            flat_x = torch.flatten(env_batch_x, start_dim=1)
            reshaped_x = flat_x.reshape(torch.cat([torch.tensor([1]), torch.tensor(flat_x.shape)]).tolist())
            m_env_feats = environment_detector(reshaped_x)[0][0]

            preds = environment_predictor(m_env_feats)
            pred_envs.append(preds)
            gt_envs.append(gt)

        loss_env_prediction = F.cross_entropy(torch.vstack(pred_envs), torch.cat(gt_envs).type(torch.long))

        # Get the total loss across the environments
        optimizer.zero_grad()
        loss_env_prediction.backward()
        optimizer.step()

        # optimize
        if j % 100 == 0:
            print(f'Epoch: {j}, Loss: {loss_env_prediction.item()}')

    return environment_detector, environment_predictor


if __name__ == "__main__":
    env_detector, env_predictor = train_environment_predictor()
    # print('====')


