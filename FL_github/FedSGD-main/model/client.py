from pathlib import Path

import torch

from torch import nn
from torch.autograd import Variable

from model.lenet import lenet5


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class client(object):
    def __init__(self, rank, data_loader):
        # seed
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # rank
        self.rank = rank
        self.device = get_device()
        self.cache_dir = Path(__file__).resolve().parent.parent / 'cache'

        # data loader
        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]

    def __load_global_model(self):
        global_model_state = torch.load(self.cache_dir / 'global_model_state.pkl', map_location=self.device)
        model = lenet5().to(self.device)
        model.load_state_dict(global_model_state)
        return model

    def __train(self, model):
        train_loss = 0
        train_correct = 0
        model.train()
        for data, target in self.train_loader:
            data = Variable(data).to(self.device)
            target = Variable(target).to(self.device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            train_loss += loss
            loss.backward()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        grads = {'n_samples': len(self.train_loader.dataset), 'named_grads': {}}
        for name, param in model.named_parameters():
            grads['named_grads'][name] = param.grad

        print('[Rank {:>2}]  Loss: {:>4.6f},  Accuracy: {:>.4f}'.format(
            self.rank,
            train_loss,
            train_correct / len(self.train_loader.dataset)
        ))
        return grads

    def run(self):
        model = self.__load_global_model()
        grads = self.__train(model=model)
        torch.save(grads, self.cache_dir / 'grads_{}.pkl'.format(self.rank))
