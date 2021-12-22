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


def train_environment_predictor():
    # TODO incorporte information about the labels in the environment classifier
    num_epochs = 3000

    input_shape = training_envs[0][0][0].shape
    flattened_input_shape = np.prod(list(input_shape))
    num_hidden_features = 64
    num_domains = len(training_envs) + len(test_envs)

    one_hot_domains = torch.nn.functional.one_hot(torch.arange(num_domains), num_domains).to(device)

    # Define the sequence classification model
    environment_detector = torch.nn.GRU(input_size = flattened_input_shape,
                                        hidden_size = num_hidden_features,
                                        num_layers = 1,
                                        batch_first = True,
                                        bidirectional = True).to(device)

    environment_predictor = torch.nn.Sequential(
        torch.nn.Linear(2 * num_hidden_features + num_domains, 512),
        torch.nn.Linear(512, 2)
    ).to(device)

    # if os.path.exists('environment_predictor.pth') and os.path.exists('environment_detector.pth'):
    #     environment_predictor = torch.load('environment_predictor.pth')
    #     environment_detector = torch.load('environment_detector.pth')
    #     # Since model is already trained, we ensure it is doing the right thing
    #     num_epochs = 100

    all_params = list(environment_detector.parameters()) + list(environment_predictor.parameters())

    optimizer = torch.optim.Adam(
        all_params,
        lr=0.003
    )
    train_generators = [data_generator(x) for x in training_envs]
    test_generators = [data_generator(x) for x in test_envs]
    all_generators = train_generators + test_generators
    all_envs = training_envs + test_envs

    for j in range(num_epochs):
        # For each training environment
        mix_ratio = torch.tensor(np.random.uniform(size=num_domains)).to(device)
        mix_ratio = mix_ratio/torch.sum(mix_ratio)

        pred_envs = []
        gt_envs = []
        x_envs = []

        for i, env in enumerate(all_envs):
            # Get the sequence of the samples
            env_batch_x, _, _, _ = next(all_generators[i])
            batch2_x, _, _, _ = next(all_generators[i])

            num_env_samples = len(env_batch_x)
            num_filtered_samples = int(num_env_samples * mix_ratio[i])


            filtered_samples_indices = np.random.choice(np.arange(num_env_samples), size=num_filtered_samples, replace=False)
            filtered_batch = env_batch_x[filtered_samples_indices]
            x_envs.append(filtered_batch)
            gt = torch.ones(num_filtered_samples).to(device) * i
            #reshaped_x = flatten_reshape(env_batch_x)
            # flat_x = torch.flatten(env_batch_x, start_dim=1)
            # reshaped_x = flat_x.reshape(torch.cat([torch.tensor([1]), torch.tensor(flat_x.shape)]).tolist())
            #m_env_feats = environment_detector(reshaped_x)[0][0]
            #preds = environment_predictor(m_env_feats)
            #pred_envs.append(preds)
            gt_envs.append(gt)

        x_envs = torch.vstack(x_envs)
        gt_envs = torch.cat(gt_envs)

        batch_indices = np.arange(len(gt_envs))
        # change the number of classification samples to make sure that the batch classification works on
        # different lengths of the sequences
        np.random.shuffle(batch_indices)
        num_selections = np.random.randint(len(batch_indices)) + 1
        batch_indices = batch_indices[:num_selections]
        x_envs = x_envs[batch_indices]
        gt_envs = gt_envs[batch_indices]

        reshaped_x = flatten_reshape(x_envs)
        m_feats = environment_detector(reshaped_x)[0][0]

        classification_domain = np.random.choice(np.arange(num_domains))
        gt_prediction = (gt_envs == classification_domain).long()
        m_feats_input = torch.ones(len(m_feats), 1).to(device) * one_hot_domains[classification_domain]
        m_feats_input = torch.hstack([m_feats_input, m_feats])

        preds = environment_predictor(m_feats_input)
        loss_env_prediction = F.cross_entropy(preds, gt_prediction)

        # Get the total loss across the environments
        optimizer.zero_grad()
        loss_env_prediction.backward()
        optimizer.step()
        # optimize
        if j % 100 == 0:
            print(f'Epoch: {j}, Loss: {loss_env_prediction.item()}')
            print(f'Domain: {classification_domain}, Accuracy: {mean_accuracy(preds, gt_prediction)}')

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
    num_all_envs = num_train_environments + len(test_generators)

    one_hot_envs = torch.nn.functional.one_hot(torch.arange(num_all_envs), num_all_envs).to(device)

    for j in range(num_epochs):
        augmentation_environment = random.choice(range(num_train_environments, num_all_envs))
        augmented_batch_x = []
        augmented_batch_y = []
        augmented_batch_colors = []
        augmented_true_y = []

        sample_ratio = 0.1
        for i, gen in enumerate(train_generators):
            x, y, t, c = next(gen)
            augmented_batch_x.append(x)
            augmented_batch_y.append(y)
            augmented_batch_colors.append(c)
            augmented_true_y.append(t)


        all_x = torch.vstack(augmented_batch_x)
        all_y = torch.cat(augmented_batch_y)
        all_colors = torch.cat(augmented_batch_colors)
        all_true_y = torch.cat(augmented_true_y)

        flattened_x = flatten_reshape(all_x)
        m_feats = env_detector(flattened_x)[0][0]

        # Choose the environment to train from randomly from the training ones
        num_samples = len(m_feats)
        m_feats_input = torch.ones(num_samples, 1).to(device) * one_hot_envs[augmentation_environment]
        m_feats_input = torch.hstack([m_feats_input, m_feats])

        selection_logits = env_predictor(m_feats_input)
        selection_probs = selection_logits[:, 1].detach().cpu().numpy()
        selection_probs -= np.min(selection_probs)
        selection_probs /= np.sum(selection_probs)

        num_selections = int(len(selection_logits) * sample_ratio)

        choice_indices = np.random.choice(np.arange(num_samples), size=num_selections, replace=False, p=selection_probs)
        # get the x and y for the choices and train on them
        filtered_x = all_x[choice_indices]
        filtered_y = all_y[choice_indices]
        filtered_colors = all_colors[choice_indices]
        filtered_true_y = all_true_y[choice_indices]


        loss_label_predictor = F.cross_entropy(network(filtered_x), filtered_y.T[0].type(torch.long))
        optimizer.zero_grad()
        loss_label_predictor.backward()
        optimizer.step()
        color_label_corr = get_color_label_correlation(filtered_y.detach().to('cpu'), filtered_colors.detach().to('cpu'))
        if j % 100 == 0:
            print(f'Epoch: {j}, Loss: {loss_label_predictor.item()}, Color_Correlation: {color_label_corr}')
            print(f'All Label/True percentage: {torch.sum((filtered_y == filtered_true_y).type(torch.int8))/len(filtered_y)}')
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


