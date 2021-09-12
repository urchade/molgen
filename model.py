from layers import Generator, RecurrentDiscriminator, TransformerDiscriminator
from tokenizer import Tokenizer
import torch
from torch import nn
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rdkit import Chem
import numpy as np

class MolGen(nn.Module):

    def __init__(self, data, hidden_dim=128, lr=1e-3, device='cpu'):
        super().__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        self.tokenizer = Tokenizer(data)

        self.generator = Generator(
            latent_dim=hidden_dim,
            vocab_size=self.tokenizer.vocab_size - 1,
            start_token=self.tokenizer.start_token - 1, # no need token
            end_token=self.tokenizer.end_token - 1,
        ).to(device)

        self.discriminator = TransformerDiscriminator(
            hidden_size=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token
            # bidirectional=True
        ).to(device)

        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr)

        self.b = 0.

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def train_step(self, x):

        batch_size, len_real = x.size()

        # create real and fake labels
        real_labels = torch.ones(batch_size, len_real).to(self.device)

        # laten var
        z = self.sample_latent(batch_size)

        real_x = x.to(self.device)  # real
        generated = self.generator.forward(z, max_len=35)

        generated_x, log_probabilities, entropies = generated.values()

        _, len_gen = generated_x.size()

        fake_labels = torch.zeros(batch_size, len_gen).to(self.device)

        # Train Discriminator

        self.disc_opt.zero_grad()

        # disc fake loss
        disc_pred_fake, fake_mask = self.discriminator(generated_x).values()

        fake_loss = F.binary_cross_entropy(
            disc_pred_fake, fake_labels, reduction='none') * fake_mask

        fake_loss = fake_loss.sum() / fake_mask.sum()

        # disc real loss
        disc_pred_real, real_mask = self.discriminator(real_x).values()

        real_loss = F.binary_cross_entropy(
            disc_pred_real, real_labels, reduction='none') * real_mask

        real_loss = real_loss.sum() / real_mask.sum()

        # combined loss
        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()

        # clip grad
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # update params
        self.disc_opt.step()

        # --------------------------
        # Train Generator
        # --------------------------

        self.gen_opt.zero_grad()

        # prediction for generated x
        gen_pred, gen_mask = self.discriminator(generated_x).values()

        # Reward
        R = (2 * gen_pred - 1)

        # reward len for each sequence
        len_rewards = gen_mask.sum(1).long()

        # list of rew of each sequences
        rewards = [rw[:lr] for rw, lr in zip(R, len_rewards)]

        # compute - (r - b) log x
        loss_gen = []
        for r, p in zip(rewards, log_probabilities):
            loss_gen.append(- ((r - self.b) * p).sum())

        # mean loss + entropy reg
        loss_gen = torch.stack(loss_gen).mean() - sum(entropies) * 0.01 / batch_size

        with torch.no_grad():

            mean_reward = (R * gen_mask).sum() / gen_mask.sum()

            self.b = 0.8 * self.b + (1 - 0.8) * mean_reward

        loss_gen.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.gen_opt.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward}

    def create_dataloader(self, data, batch_size=128, shuffle=True, num_workers=5):

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.tokenizer.batch_tokenize,
            num_workers=num_workers
        )

    def train_n_steps(self, train_loader, max_step=100):

        iter_loader = iter(train_loader)

        for step in range(max_step):

            try:
                batch = next(iter_loader)
            except:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            losses = self.train_step(batch)

            if step % 50 == 0:

                s = self.decode_n(100)

                print(f'valid = {s: .2f}')

    def get_mapped(self, seq):
        return ''.join([self.tokenizer.inv_mapping[i] for i in seq])

    @torch.no_grad()
    def decode_n(self, n):
        z = torch.randn((n, 128)).cuda()

        seq = self.generator(z)['x'].cpu()

        lenghts = (seq > 0).sum(1)

        pack = [self.get_mapped(x[:l-1].numpy()) for x, l in zip(seq, lenghts)]

        s = np.array([Chem.MolFromSmiles(k) is not None for k in pack])

        return s.mean()


if __name__ == '__main__':

    data = []

    with open('qm9.csv', "r") as f:
        for line in f.readlines()[1:]:
            data.append(line.split(",")[1])

    model = MolGen(data, device="cuda")

    loader = model.create_dataloader(data)

    model.train_n_steps(loader, 10000)