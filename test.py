import numpy as np
import torch
import random
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F
from domainbed import networks
import os


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
batch_size = 512

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


def train_environment_predictor():
    # TODO incorporte information about the labels in the environment classifier
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

    # if os.path.exists('environment_predictor.pth') and os.path.exists('environment_detector.pth'):
    #     environment_predictor = torch.load('environment_predictor.pth')
    #     environment_detector = torch.load('environment_detector.pth')
    #     # Since model is already trained, we ensure it is doing the right thing
    #     num_epochs = 100


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
            reshaped_x = flatten_reshape(env_batch_x)
            # flat_x = torch.flatten(env_batch_x, start_dim=1)
            # reshaped_x = flat_x.reshape(torch.cat([torch.tensor([1]), torch.tensor(flat_x.shape)]).tolist())
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

    torch.save(environment_detector, 'environment_detector.pth')
    torch.save(environment_predictor, 'environment_predictor.pth')
    print(loss_env_prediction.item())
    return environment_detector, environment_predictor


def get_color_label_correlation(labels, colors):
    labels = labels.T[0]
    return np.correlate(labels, colors)/len(labels)


def mean_accuracy(logits, y):
    a = torch.argmax(logits, axis=1)
    b = torch.flatten(y)
    acc = torch.sum((a == b).type(torch.int))/len(a)
    return acc


def train_erm(env_detector, env_predictor):
    num_epochs = 10000
    num_classes = 2
    input_shape = (2, 28, 28)
    hparams = {'data_augmentation': True,
     'resnet18': False,
     'resnet_dropout': 0.0,
     'class_balanced': False,
     'nonlinear_classifier': False,
     'lr': 0.005,
     'weight_decay': 0.0,
     'batch_size': 512}
    featurizer = networks.Featurizer(input_shape, hparams)
    classifier = networks.Classifier(
        featurizer.n_outputs,
        num_classes,
        hparams['nonlinear_classifier']
    )
    network = nn.Sequential(featurizer, classifier).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr = hparams['lr']
    )
    train_generators = [data_generator(x, batch_size=batch_size) for x in training_envs]
    test_generators = [data_generator(x) for x in test_envs]
    num_train_environments = len(train_generators)

    for j in range(num_epochs):
        augmentation_environment = random.choice(range(num_train_environments))
        augmented_batch_x = []
        augmented_batch_y = []
        augmented_batch_colors = []
        augmented_true_y = []

        for i, gen in enumerate(train_generators):
            x, y, t, c = next(gen)
            batch_len = len(x)

            sample_selection_cap = int(batch_len * 0.3)
            num_env_samples = int(sample_selection_cap / num_train_environments)

            # Get the environment
            reshaped_x = flatten_reshape(x)
            m_env_feats = env_detector(reshaped_x)[0][0]
            env_preds = env_predictor(m_env_feats)
            obj_probs = env_preds[:, augmentation_environment].detach()
            obj_probs -= torch.min(obj_probs)
            # obj_probs /= torch.max(obj_probs)
            # obj_probs = torch.abs(0.5 - obj_probs)
            # obj_probs -= torch.min(obj_probs)
            obj_probs /= torch.sum(obj_probs)
            obj_indices = np.arange(batch_len)
            # TODO fix the size to be less than the batch size
            augmented_dataset_indices = np.random.choice(obj_indices, size=num_env_samples, replace=False, p=obj_probs.cpu().numpy())
            augmented_batch_x.append(x[augmented_dataset_indices])
            augmented_batch_y.append(y[augmented_dataset_indices])
            augmented_true_y.append(t[augmented_dataset_indices])
            augmented_batch_colors.append(c[augmented_dataset_indices])
            # augmented_batch_x.append(x)
            # augmented_batch_y.append(y)
            # augmented_true_y.append(t)
            # augmented_batch_colors.append(c)

            if j % 100 == 0:
                print(f'Augmentation Environment: {augmentation_environment}')
                print(f'E{i} Augmented Label/True percentage: {torch.sum((t[augmented_dataset_indices] == y[augmented_dataset_indices]).type(torch.int8))/len(augmented_dataset_indices)}')
                print(f'E{i} Label/True percentage: {torch.sum((t == y).type(torch.int8))/len(y)}')

        all_x = torch.vstack(augmented_batch_x)
        all_y = torch.cat(augmented_batch_y)
        all_colors = torch.cat(augmented_batch_colors)
        all_true_y = torch.cat(augmented_true_y)

        loss_label_predictor = F.cross_entropy(network(all_x), all_y.T[0].type(torch.long))
        optimizer.zero_grad()
        loss_label_predictor.backward()
        optimizer.step()
        color_label_corr = get_color_label_correlation(all_y.detach().to('cpu'), all_colors.detach().to('cpu'))
        if j % 100 == 0:
            print(f'Epoch: {j}, Loss: {loss_label_predictor.item()}, Color_Correlation: {color_label_corr}')
            print(f'All Label/True percentage: {torch.sum((all_y == all_true_y).type(torch.int8))/len(all_y)}')
            # print the accuracy for a sample of the test environment
            e1_x, e1_y, e1_true_y, e1_colors = next(train_generators[0])
            e2_x, e2_y, e2_true_y, e2_colors = next(train_generators[1])
            # Get a batch of the test data
            test_x, test_y, test_true_y, test_colors = next(test_generators[0])
            # get the predictions 
            test_preds = network(test_x)
            e1_preds = network(e1_x)
            e2_preds = network(e2_x)
            # compare the number of predictions that match the ground truth
            test_acc = mean_accuracy(test_preds, test_y)
            test_true_acc = mean_accuracy(test_preds, test_true_y)

            e1_acc = mean_accuracy(e1_preds, e1_y)
            e1_true_acc = mean_accuracy(e1_preds, e1_true_y)


            e2_acc = mean_accuracy(e2_preds, e2_y)
            e2_true_acc = mean_accuracy(e2_preds, e2_true_y)

            print(f'Test Accuracy: {test_acc}, True: {test_true_acc}')
            print(f'E1 Accuracy: {e1_acc}, True: {e1_true_acc}')
            print(f'E2 Accuracy: {e2_acc}, True: {e2_true_acc}')
    pass

if __name__ == "__main__":
    env_detector, env_predictor = train_environment_predictor()
    # print('====')
    train_erm(env_detector, env_predictor)


