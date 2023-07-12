# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms

import copy
import numpy as np
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from .lib.misc import CudaTimer

from codes.models.utils import infinite_iterator


try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from . import networks
from .lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj, Nonparametric
)



ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'EQRM',
    'InformationalHeat',
    'CT4Recognition',
    'ISR_Mean',
    'ISR_Cov',
    'LaCIM',
    'TCM'
]

DA_ONLY_ALGORITHMS = [
    'InformationalHeat',
    'TCM',
    'CDTrans',
    'TVT'
]

HEAVY_PREDICTIONS = [
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def begin_epoch(self):
        pass
    def end_epoch(self):
        pass
    def setup(self, start_epoch, n_epoch):
        """
            setting up schedulers, etc.
        """
        pass

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.step = 0
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))
        
        self.writer: SummaryWriter = hparams['writer']
        hparams['writer'] = None

    def update(self, minibatches, unlabeled=None):
        if self.conditional:    unlabeled = None
        
        device = minibatches[0][0].device
        self.update_count += 1
        self.step += 1
        x = torch.cat([x for x, y in minibatches] + (unlabeled if unlabeled is not None else []))
        all_y = torch.cat([y for x, y in minibatches])
        z = self.featurizer(x)
        all_z = z[:all_y.shape[0]]

        if self.conditional and unlabeled is not None:
            disc_input = all_z + self.class_embeddings(all_y)
            raise NotImplemented()
        else:
            disc_input = z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ] + ([
            torch.full((x.shape[0], ), i + len(minibatches), dtype=torch.int64, device=device)
            for i, x in enumerate(unlabeled)
        ] if unlabeled is not None else []))

        if self.class_balance:
            raise NotImplemented()
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        # self.writer.add_scalar('disc', disc_loss, global_step=self.step)

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            # self.writer.add_scalar('classifier', classifier_loss, global_step=self.step)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = logits[0][0].device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_meta_test = hparams['n_meta_test']

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = x.device
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters(), allow_unused=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            if grads is None or grads[0] is None:
                continue
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(),
                                        create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(),
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}


class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2

        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True, allow_unused=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss}

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            if grads is None or grads[0] is None:
                continue
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)

class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(num_domains+1)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(num_domains+1)]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches, unlabeled=None):

        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams['iters']:
            # TRM
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)

            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}

class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = logits[0][0].device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IRM_penalty': irm_penalty.item(),
                'IB_penalty': ib_penalty.item()}


class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional):
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"clf_loss": clf_loss.item(), "bn_loss": bn_loss.item(), "total_loss": total_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)


class Transfer(Algorithm):
    '''Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)'''
    ''' tries to ensure transferability among source domains, and thus transferabiilty between source and target'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # Architecture
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier.load_state_dict(self.classifier.state_dict())

        # Optimizers
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr'])
        else:
            self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d'])

    def loss_gap(self, minibatches, device):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device
        # outer loop
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
        return {'loss': loss.item(), 'gap': -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}


    def predict(self, x):
        return self.classifier(self.featurizer(x))


class AbstractCausIRL(ERM):
    '''Abstract class for Causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractCausIRL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class CausIRL_MMD(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the MMD distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_MMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=True)


class CausIRL_CORAL(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_CORAL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=False)


class EQRM(ERM):
    """
    Empirical Quantile Risk Minimization (EQRM).
    Algorithm 1 from [https://arxiv.org/pdf/2207.09944.pdf].
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, dist=None):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.register_buffer('alpha', torch.tensor(self.hparams["eqrm_quantile"], dtype=torch.float64))
        if dist is None:
            self.dist = Nonparametric()
        else:
            self.dist = dist

    def risk(self, x, y):
        return F.cross_entropy(self.network(x), y).reshape(1)

    def update(self, minibatches, unlabeled=None):
        env_risks = torch.cat([self.risk(x, y) for x, y in minibatches])

        if self.update_count < self.hparams["eqrm_burnin_iters"]:
            # Burn-in/annealing period uses ERM like penalty methods (which set penalty_weight=0, e.g. IRM, VREx.)
            loss = torch.mean(env_risks)
        else:
            # Loss is the alpha-quantile value
            self.dist.estimate_parameters(env_risks)
            loss = self.dist.icdf(self.alpha)

        if self.update_count == self.hparams['eqrm_burnin_iters']:
            # Reset Adam (like IRM, VREx, etc.), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["eqrm_lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return {'loss': loss.item()}

from codes.models import (
    model, 
    abstraction, 
    attention,
    attention_memory, 
    erm, 
    discrete_rate_distortion, 
    inference,
)
from codes.algorithms import training

class InformationalHeat(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        
        heads=12
        dim=768
        mlp_dim=3072
        depth=5
        patch_size=16
        n_key=768


        n_query = 256
        attention_args= {
            'k': 1,
            'n_query': n_query,
            'uncaptured_dot_loss_weight': hparams['weight_single'],
        }
        assert input_shape[1] == input_shape[2], input_shape
        self.model = model.Model(
            attention.ViT(
                image_size=input_shape[1],
                patch_size=patch_size,
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                attention_type=attention.ShannonTop1Attention,
                max_token=n_query,
                dropout=0.0,
                dim_head=mlp_dim // heads,
                pool='none',
                attention_args=attention_args,
                skip_connection=True,
                n_cls=hparams['n_cls']
            ),
            inference.InferencePlugin(n_inductive_bias=hparams['n_cls'], pool='max_bias', skip_connection=1),
            erm.ERM(),
            attention_memory.AttentionMemoryPlugin(has_residual=False),
            # LocalSimplicityPlugin(
                # max_n_token=max(n_query, 16, (image_size // patch_size)**2),
                # n_inductive_bias=args.n_inductive_bias,
                # n_class=n_class,
                # exchange_block_size=exchange_block_size,
                # total_layer=depth + 1,
                # dim=dim,
                # heads=heads_simplicity,
                # mlp_dim=mlp_dim_simplicity,
                # dim_head=mlp_dim_simplicity//heads_simplicity,
                # weight=args.weight_local_simplicity
            # ),
            discrete_rate_distortion.DiscreteRateDistortionPlugin(
                D=hparams['D'], 
                dim_input=(heads * 1 + 2) * dim, dim_output=dim, n_key=n_key, n_domain=12, depth_mlp=3, dim_mlp=512, beta=0.5, 
                lr=hparams['lr_d'] * 10,
                weight_distortion=hparams['weight_distortion'],
                weight_heat=hparams['weight_heat']
            ),
            erm.Confidence(weight=hparams['weight_confidence']),
            abstraction.AdversarialAbstraction(
                n_domain=2,
                dim=dim,
                n_heads=heads,
                k=1,
                weight=1e1,
                abstraction_per_update=0,
                alpha=hparams['lr_d'] / hparams['lr'],
                lr=hparams['lr_d'],
                loss_fn=abstraction.WassersteinLoss(c=hparams['wasserstein_clip']),
            ),
            writer=hparams['writer']
        )

        self.trainer = training.Training(
            0, 
            self.model, 
            [None] * 15, 
            hparams['lr'],
            writer=hparams['writer'], 
            test_every_epoch=None,
            test_batch=None,
            save_every_epoch=None,
            weight_decay=hparams['weight_decay'],
            optimizer='Adam',
            scheduler=True
        )

        # print(self.model)
        # print(self.trainer)

        self.num_domains = num_domains
        self.trainer._init_training()
        self.model.iteration = 0
        self.model.epoch = 0

    def update(self, minibatches, unlabeled=None):
        assert unlabeled is not None
        if len(minibatches) > 1:
            print("warning: multiple source domains, where only the 1st on is used")
        x, y = minibatches[0]
        unlabeled = unlabeled[0]
        unlabeled_Y = torch.zeros([len(unlabeled)], device=unlabeled.device, dtype=torch.long)
        X, Y = torch.cat([x, unlabeled]), torch.cat([y, unlabeled_Y])
        D = torch.cat([torch.full([len(x)], fill_value=0, device=x.device, dtype=torch.long), torch.full([len(unlabeled)], fill_value=1, device=unlabeled.device, dtype=torch.long)])
        labeled = torch.cat([torch.full([len(x)], fill_value=True, device=x.device, dtype=torch.bool), torch.full([len(unlabeled_Y)], fill_value=False, device=x.device, dtype=torch.bool)])
        Y, D, labeled = Y.flatten(), D.flatten(), labeled.flatten()

        losses = self.trainer._train_batch(X, Y, D, labeled)
        self.model.iteration = self.model.iteration + 1

        losses = {key: float(value) for key, value in losses.items()}
        return {
            'n_inductive_bias_difference': losses['n_inductive_bias/difference'],
            'domain_discrimination': abs(losses['domain discrimination loss/attention']) + abs(losses['domain discrimination loss/initial token']),
            **{key: value for key, value in losses.items() if 'distortion' in key},
            'Q_F': losses['heat/Q_F'],
            'Q_0': losses['heat/Q_0']
        }


    def predict(self, x):
        self.model.eval()
        return self.model(x, None, None, None, test_mode=True)
    def end_epoch(self):
        self.model.epoch += 1
        self.eval()
        self.train()

from torch.utils.data import Dataset, DataLoader
import sys
class DatasetRequiringAlgorithm(Algorithm):
    def set_datasets(self, datasets: 'list[Dataset]'):
        self.datasets = datasets

from baselines.ISR.real_datasets import isr
class AbstractISR(DatasetRequiringAlgorithm):
    """Invariant Subspace Recovery

        only ERM- or GroupDRO-trained backbones are supported

        @misc{wang_provable_2022,
            title = {Provable Domain Generalization via Invariant-Feature Subspace Recovery},
            url = {http://arxiv.org/abs/2201.12919},
            number = {{arXiv}:2201.12919},
            publisher = {{arXiv}},
            author = {Wang, Haoxiang and Si, Haozhe and Li, Bo and Zhao, Han},
            urldate = {2023-06-13},
            date = {2022-07-07},
            langid = {english},
            eprinttype = {arxiv},
            eprint = {2201.12919 [cs, stat]},
            keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
        }

    """

    CHECKPOINT_FREQ = 300

    class ISRClassifier(isr.ISRClassifier):
        def fit(self, features, labels, envs, chosen_class: int = None, d_spu: int = None, given_clf=None,
                spu_scale: float = None):

            # estimate the stats (mean & cov) and fit a PCA if requested
            self.fit_data(features, labels, envs)

            if chosen_class is None:
                assert self.chosen_class is not None, "chosen_class must be specified if not given in the constructor"
                chosen_class = self.chosen_class

            if self.version == 'mean':
                self.fit_isr_mean(chosen_class=chosen_class, d_spu=d_spu)
            elif self.version == 'cov':
                self.fit_isr_cov(chosen_class=chosen_class, d_spu=d_spu)
            else:
                raise ValueError(f"Unknown ISR version: {self.version}")

            self.fit_clf(features, labels, given_clf=given_clf) # spu_sclae is passed unexpectedly in original isr.ISRClassifier
            return self

    def __init__(self, version, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        if hparams['backbone'] == 'GroupDRO':
            self.backbone = GroupDRO(input_shape, num_classes, num_domains, hparams)
        elif hparams['backbone'] == 'ERM':
            self.backbone = ERM(input_shape, num_classes, num_domains, hparams)
        else:
            raise NotImplemented(hparams['backbone'])
        self.isr_classifier = self.ISRClassifier(
            version, 
            d_spu= int(hparams['d_spu_ratio'] * self.backbone.featurizer.n_outputs) if hparams['d_spu_ratio'] >= 0 else -1
        )

        self.features = None
        self.labels = None
        self.envs = None
        self.is_classifier_latest = False

        self.batch_size = hparams['batch_size']

    
    def update(self, minibatches, unlabeled=None):
        self.backbone.featurizer.train()
        self.is_classifier_latest = False
        self.features = None
        self.labels = None,
        self.envs = None
        return self.backbone.update(minibatches, unlabeled)
    
    def parse_feature(self, device='cuda'):
        self.features = []
        self.labels = []
        self.envs = []
        self.backbone.featurizer.eval()
        with torch.no_grad():
            for env, dataset in enumerate(self.datasets):
                dl = DataLoader(dataset, 64, False, num_workers=8, pin_memory=True)
                for X, Y in dl:
                    feature = self.backbone.featurizer(X.to(device))
                    self.features.append(feature)
                    self.labels.append(Y.to(device))
                    self.envs.append(torch.full([len(Y)], env, device=self.features[-1].device))
            self.features = torch.cat(self.features, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
            self.envs = torch.cat(self.envs, dim=0)
            assert len(self.features.shape) == 2, self.features.shape
            assert len(self.labels.shape) == 1, self.labels.shape
            assert len(self.envs.shape) == 1, self.envs.shape


    def fit(self):
        self.isr_classifier.fit(self.features.cpu().numpy(), self.labels.cpu().numpy(), self.envs.cpu().numpy(), chosen_class=0)
        self.is_classifier_latest = True
        self.features = None
        self.labels = None,
        self.envs = None

    def predict(self, x):
        if not self.is_classifier_latest:
            self.parse_feature(x.device)
            self.fit()
        features = self.backbone.featurizer(x)
        res = torch.tensor(self.isr_classifier.predict(features.cpu().numpy()))
        return res.to(x.device)
    def train(self):
        self.backbone.featurizer.train()
    def eval(self):
        self.backbone.featurizer.eval()


class ISR_Mean(AbstractISR):
     def __init__(self, input_shape, num_classes, num_domains, hparams):
         super().__init__("mean", input_shape, num_classes, num_domains, hparams)

class ISR_Cov(AbstractISR):
     def __init__(self, input_shape, num_classes, num_domains, hparams):
         super().__init__("cov", input_shape, num_classes, num_domains, hparams)


import torch.nn.functional as nnf
class ShufflePatches(object):
    """
        https://stackoverflow.com/a/66963266
    """
    def __init__(self, patch_size):
        self.ps = patch_size

    def __call__(self, x):
        # divide the batch of images into non-overlapping patches
        u = nnf.unfold(x, kernel_size=self.ps, stride=self.ps, padding=0)
        # permute the patches of each image in the batch
        pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
        # fold the permuted patches back together
        f = nnf.fold(pu, x.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
        return f

from torch.utils.data import ConcatDataset, RandomSampler, Subset
class CT4Recognition(DatasetRequiringAlgorithm):
    """Causal Transportability for Recognition

    Only SimCLR pretraining is supported. SimCLR codes are adapted from https://github.com/sthalles/SimCLR.

    Color-related augmentations in SimCLR are disabled if spurious features are colors.

    @misc{mao_causal_2022,
        title = {Causal Transportability for Visual Recognition},
        url = {http://arxiv.org/abs/2204.12363},
        number = {{arXiv}:2204.12363},
        publisher = {{arXiv}},
        author = {Mao, Chengzhi and Xia, Kevin and Wang, James and Wang, Hao and Yang, Junfeng and Bareinboim, Elias and Vondrick, Carl},
        urldate = {2023-05-24},
        date = {2022-04-26},
        eprinttype = {arxiv},
        eprint = {2204.12363 [cs]},
        keywords = {Computer Science - Computer Vision and Pattern Recognition},
    }
    """
    CHECKPOINT_FREQ = 300
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        assert input_shape[1] == input_shape[2]
        from baselines.SimCLR import SimCLR
        self.simCLR = SimCLR(input_shape[1], self.featurizer, hparams)

        x_prime_shape = 1
        for l in input_shape:
            x_prime_shape *= l
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs + x_prime_shape,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.n_j = hparams['n_j']
        self.shuffle = ShufflePatches(16)
        self.num_classes = num_classes

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    
    def setup(self, start_epoch, n_epoch):
        self.simCLR.setup(start_epoch, n_epoch)
    def begin_epoch(self):
        self.simCLR.begin_epoch()
    def end_epoch(self):
        self.simCLR.end_epoch()
    class DualDataset(Dataset):
        def __init__(self, dataset: Dataset, num_classes) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.dataset = dataset
            tmp_dataloader = DataLoader(dataset, 128, False, num_workers=8) # to fasten label gathering with multiple workers

            self.label_specific_dataloader = []
            labels = torch.cat([label for _, label in tqdm(tmp_dataloader, desc="gathering labels")]).flatten()
            assert max(labels) < self.num_classes, max(labels)

            for y in tqdm(range(self.num_classes), desc="conditional dataloader"):
                indices = [i for i, label in enumerate(labels) if label == y]
                filtered = Subset(dataset, indices)
                label_specific_sampler = RandomSampler(filtered, replacement=True)
                self.label_specific_dataloader.append(DataLoader(filtered, 1, sampler=label_specific_sampler, drop_last=True))
            self.conditional_iterators = [infinite_iterator(l) for l in self.label_specific_dataloader]
        def __len__(self):  
            return len(self.dataset)
        def __getitem__(self, index):
            x, y = self.dataset[index]
            x_p, _ = next(self.conditional_iterators[y])
            return x, x_p[0], y




    def build_marginal_dataloaders(self, b):
        dataset = ConcatDataset(self.datasets)
        sampler = RandomSampler(dataset, replacement=True)
        self.marginal_dataloader = DataLoader(dataset, batch_size=b, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)
        self.marginal_iterator = infinite_iterator(self.marginal_dataloader)

    def build_joint_dataloaders(self, b):
        dataset = ConcatDataset(self.datasets)
        dual_dataset = self.DualDataset(dataset, self.num_classes)
        sampler = RandomSampler(dual_dataset, replacement=True)
        self.joint_dataloader = DataLoader(dual_dataset, b, sampler=sampler, num_workers=8, pin_memory=True)
        self.joint_iterator = infinite_iterator(self.joint_dataloader)
        

    def sample(self, b):
        if not hasattr(self, 'marginal_dataloader') or self.marginal_dataloader is None or self.marginal_dataloader.batch_size < b:
            self.build_marginal_dataloaders(b)

        return next(self.marginal_iterator)[0][:b]

    def corrupted(self, b, device):
        samples = self.sample(b).to(device)
        assert len(samples.shape) == 4, samples.shape # b c w h
        assert samples.shape[1] == 3, samples.shape
        assert len(samples) == b, (samples.shape, b)
        return self.shuffle(samples)
            
    def joint(self, r):
        x_prime = self.corrupted(b=len(r), device=r.device)
        joint = torch.cat([r, x_prime.flatten(1)], dim=1)
        assert len(joint.shape) == 2 and joint.shape[0] == r.shape[0]
        return joint
        
    def predict(self, x):
        r = self.featurizer(x)
        p_y = []
        for _ in range(self.n_j):
            logit = self.classifier(self.joint(r))
            assert len(logit.shape) == 2 and logit.shape[0] == x.shape[0], logit.shape
            p_y.append(torch.softmax(logit, dim=-1))
        p_y = torch.stack(p_y, dim=1)
        return p_y.mean(dim=1)
    
    def update(self, minibatches, unlabeled=None):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        device = x.device


        losses_SimCLR = self.simCLR.update(x)

        batch_size = len(x)
        if not hasattr(self, 'joint_dataloader') or self.joint_dataloader is None or self.joint_dataloader.batch_size < batch_size:
            self.build_joint_dataloaders(batch_size)

        losses = []
        self.featurizer.eval()
        for j in range(1):
            x, x_p, y = [a.to(device) for a in next(self.joint_iterator)]

            with torch.no_grad():
                r = self.featurizer(x)
            x_p = self.shuffle(x_p)

            joint = torch.cat([r, x_p.flatten(1)], dim=1)

            losses.append(F.cross_entropy(self.classifier(joint), y).reshape(1))
        
        self.featurizer.train()
        
        loss = torch.cat(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            **losses_SimCLR,
            'CE_loss': float(loss)
        }
    def train(self):
        self.simCLR.train()
        self.classifier.train()
    def eval(self):
        self.simCLR.eval()
        self.classifier.eval()


sys.path.insert(-1, 'baselines/LaCIM')
sys.path.insert(-1, 'baselines/LaCIM/real_world')
from baselines.LaCIM.real_world.LaCIM_rho import LaCIM_rho
class LaCIM(Algorithm):
    """LaCIM

        using LaCIM_rho because this version gives results in the paper according to LaCIM's `README.md`

        @article{sun_recovering_nodate,
            title = {Recovering Latent Causal Factor for Generalization to Distributional Shifts},
            author = {Sun, Xinwei and Wu, Botong and Zheng, Xiangyu and Liu, Chang and Chen, Wei and Qin, Tao and Liu, Tie-Yan},
            langid = {english},
        }
    """
    CHECKPOINT_FREQ = 300
    class ExpandedLaCIM_rho(LaCIM_rho):
        def __init__(self, in_channel=1, zs_dim=256, num_classes=1, decoder_type=0, total_env=2, args=None, is_cuda=1):
            self.image_size = args.image_size
            super().__init__(in_channel, zs_dim, num_classes, decoder_type, total_env, args, is_cuda)
        def get_Enc_x_256(self):
            return nn.Sequential(
                self.Conv_bn_ReLU(self.in_channel, 32),
                nn.MaxPool2d(2),
                self.Conv_bn_ReLU(32, 64),
                nn.MaxPool2d(2),
                self.Conv_bn_ReLU(64, 128),
                nn.MaxPool2d(2),
                self.Conv_bn_ReLU(128, 256),
                nn.MaxPool2d(2),
                self.Conv_bn_ReLU(256, 256),
                nn.MaxPool2d(2),
                self.Conv_bn_ReLU(256, 256),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        def get_Enc_x_28(self):
            # HACK: avoid modification to paper's codes
            return self.get_Enc_x_256()
        
        def get_Dec_x_256(self):
            from baselines.LaCIM.real_world.models import UnFlatten
            return nn.Sequential(
                UnFlatten(type='2d'),
                nn.Upsample(2),
                self.TConv_bn_ReLU(in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
                self.TConv_bn_ReLU(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
                self.TConv_bn_ReLU(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0),
                self.TConv_bn_ReLU(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0),
                self.TConv_bn_ReLU(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=0),
                self.TConv_bn_ReLU(in_channels=8, out_channels=4, kernel_size=2, stride=2, padding=0),
                self.TConv_bn_ReLU(in_channels=4, out_channels=3, kernel_size=2, stride=2, padding=0),
                # nn.Conv2d(in_channels=4, out_channels=self.in_channel, kernel_size=256 - self.image_size + 1),
                    # WHY: this conv2d make back propagation 100x slow
                nn.Sigmoid()
            )
        
        def get_Dec_x_28(self):
            # HACK: avoid modification to paper's codes
            return self.get_Dec_x_256()
        def get_x_y(self, z, s):
            zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(zs[:, self.z_dim:])
            return rec_x[:, :, -self.image_size:, -self.image_size:].contiguous(), pred_y
        def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor, sample_num=None):
            if sample_num is None:
                return super().reparametrize(mu, logvar)
            std = logvar.mul(0.5).exp_()
            shape = torch.Size([sample_num] + list(std.shape))
            if self.is_cuda:
                eps = torch.cuda.FloatTensor(shape).normal_()
            else:
                eps = torch.FloatTensor(shape).normal_()
            return eps.mul(std.unsqueeze(dim=0)).add_(mu.unsqueeze(dim=0))
        def broadcast(self, indices: torch.Tensor):
            assert len(indices.shape) == 1
            if not hasattr(self, '_broadcast') or len(self._broadcast) != len(indices):
                self._broadcast = torch.arange(0, len(indices), device=indices.device, dtype=torch.long)
            return self._broadcast
        def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
            raw_x = x
            x = self.Enc_x(x)
            if is_train == 0:
                # test only
                # init
                with torch.no_grad():
                    z_init, s_init = None, None
                    for env_idx in range(self.args.env_num):
                        mu, logvar = self.encode(x, env_idx)
                        zs = self.reparametrize(mu, logvar, self.args.sample_num)
                        z = self.phi_z[env_idx](zs[..., :self.z_dim])
                        s = self.phi_s[env_idx](zs[..., self.z_dim:])
                        zs = torch.cat([z, s], dim=-1)
                        recon_x = self.Dec_x(zs.flatten(start_dim=0, end_dim=1)).unflatten(dim=0, sizes=z.shape[:2])
                        recon_x = recon_x[..., -self.image_size:, -self.image_size:].contiguous() # sampe_num b c h w
                        rec_losses = F.binary_cross_entropy(
                            recon_x.flatten(-3),
                            (raw_x * 0.5 + 0.5).flatten(-3).expand(len(recon_x), -1, -1),
                            reduction='none'
                        ).mean(-1)
                        # assert len(rec_losses.shape) == 2 and tuple(rec_losses.shape) == (self.args.sample_num, len(x)), (rec_losses.shape, x.shape)
                        mins, indices = rec_losses.min(dim=0)
                        brd = self.broadcast(indices)
                        new_z, new_s = z[indices, brd], s[indices, brd]

                        if z_init is None:
                            z_init, s_init = new_z, new_s
                            min_rec_loss = mins
                        else:
                            replace = (mins < min_rec_loss)
                            # assert len(replace.shape) == 1 and replace.shape[0] == z_init.shape[0], (replace.shape, z_init.shape)
                            # assert z_init.shape == new_z.shape, (z_init.shape, new_z.shape)
                            # assert len(z_init.shape) == 2, z_init.shape
                            replace = replace.unsqueeze(dim=-1)
                            z_init = (~replace) * z_init + replace * new_z
                            s_init = (~replace) * s_init + replace * new_s
                            min_rec_loss = min_rec_loss.minimum(mins)
                
                with torch.enable_grad():
                    z, s = z_init, s_init
                    if is_debug:
                        pred_y_init = self.get_y(s)
                    z.requires_grad = True
                    s.requires_grad = True
            
                    optimizer = torch.optim.Adam(params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)
            
                    for i in range(self.args.test_ep):
                        optimizer.zero_grad()
                        zs = torch.cat([z, s], dim=1)
                        rec_x = self.Dec_x(zs)
                        loss = rec_x.mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        recon_x, _ = self.get_x_y(z, s)
                        assert recon_x.requires_grad
                        BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                    (raw_x * 0.5 + 0.5).view(-1, 3 * self.args.image_size ** 2),
                                                    reduction='none')
                        loss = BCE.mean(1) 
                        loss = loss.mean()
                        loss.backward()
                        optimizer.step()
                        # print(i, s)
                pred_y = self.get_y(s)
                if is_debug:
                    return pred_y_init, pred_y
                else:
                    return pred_y
            elif is_train == 2:
                mu, logvar = self.encode(x, 0)
                zs = self.reparametrize(mu, logvar)
                s = self.phi_s[0](zs[:, self.z_dim:])
                return self.Dec_y(s)
            else:
                mu, logvar = self.encode(x, env)
                zs = self.reparametrize(mu, logvar)
                z = self.phi_z[env](zs[:, :self.z_dim])
                s = self.phi_s[env](zs[:, self.z_dim:])
                zs = torch.cat([z, s], dim=1)
                rec_x = self.Dec_x(zs)
                pred_y = self.Dec_y(zs[:, self.z_dim:])
                if feature == 1:
                    return pred_y, rec_x[:, :, -self.image_size:, -self.image_size:].contiguous(), mu, logvar, z, s, zs
                else:
                    return pred_y, rec_x[:, :, -self.image_size:, -self.image_size:].contiguous(), mu, logvar, z, s
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        from argparse import Namespace
        self.model = LaCIM.ExpandedLaCIM_rho(
            in_channel=3,
            zs_dim=hparams['zs_dim'],
            num_classes=num_classes,
            total_env=num_domains,
            decoder_type=1,
            args=Namespace(**{
                'z_ratio': 0.5, 
                'image_size': input_shape[1],
                'env_num': num_domains,
                'sample_num': hparams['sample_num'],
                'lr2': hparams['lr2'],
                'reg2': hparams['reg2'],
                'test_ep': hparams['test_ep']
            })
        )
        # TODO: ViT as backbone

        print(input_shape)
        if not isinstance(input_shape, int) and len(input_shape) > 1:
            assert len(input_shape) == 3, input_shape
            assert input_shape[1] == input_shape[2]
            input_shape = input_shape[1]
        self.image_size = input_shape

        if hparams['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams['lr'], momentum=hparams['momentum'], weight_decay=hparams['weight_decay'])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])

        self.hparams = hparams

        from torchvision import transforms
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.lr_controller = hparams['lr_controller']
        self.lr_decay = hparams['lr_decay']
        self.lr = hparams['lr']

        self.epoch = 0
        
    
    def predict(self, x):
        x = self.test_transform(x)
        pred_y_init, pred_y = self.model(x, is_train=0, is_debug=1)
        return pred_y
    def VAE_loss(self, recon_x, x, mu, logvar):
        """
        pred_y: predicted y
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        q_y_s: prior
        beta: tradeoff params
        """
        x = x * 0.5 + 0.5
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 3 * self.image_size ** 2), 
            x.view(-1, 3 * self.image_size ** 2),
            reduction='mean'
        )
    
        KLD = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
    
        return BCE, KLD
        
    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            new_lr = self.lr * self.lr_decay ** (self.epoch // self.lr_controler)
            param_group['lr'] = self.lr * self.lr_decay ** (self.epoch // self.lr_controler)

    def update(self, minibatches, unlabeled=None):
        """
            adapted from `train()` of `LaCIM/real_world/LaCIM_rho.py`
        """
        x = torch.cat([x for x, y in minibatches])
        x = self.train_transform(x)
        target = torch.cat([y for x, y in minibatches])
        env = torch.cat([torch.full_like(y, i) for i, (x, y) in enumerate(minibatches)])
        device = x.device
        self.model.to(device)

        loss = torch.FloatTensor([0.0]).to(device)

        recon_loss = torch.FloatTensor([0.0]).to(device)
        kld_loss = torch.FloatTensor([0.0]).to(device)
        cls_loss = torch.FloatTensor([0.0]).to(device)
        for ss in range(self.model.total_env):
            if torch.sum(env == ss) <= 1:
                continue
            _, recon_x, mu, logvar, z, s, zs = self.model(x[env == ss,:,:,:], ss, feature=1, is_train = 1)
            pred_y = self.model.get_y_by_zs(mu, logvar, ss)
            recon_loss_t, kld_loss_t = self.VAE_loss(recon_x, x[env == ss,:,:,:], mu, logvar)
            cls_loss_t = F.nll_loss(torch.log(pred_y), target[env == ss])
            
            recon_loss = torch.add(recon_loss, torch.sum(env == ss) * recon_loss_t)
            kld_loss = torch.add(kld_loss, torch.sum(env == ss) * kld_loss_t)
            cls_loss = torch.add(cls_loss, torch.sum(env == ss) * cls_loss_t)
        recon_loss = recon_loss / x.size(0)
        kld_loss = kld_loss / x.size(0)
        cls_loss = cls_loss / x.size(0)

        loss = torch.add(loss, self.hparams['weight_recon'] * recon_loss + self.hparams['weight_kld'] * kld_loss + cls_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "cls_loss":     cls_loss.item(),
            "kld_loss":     kld_loss.item(),
            "recon_loss":   recon_loss.item(),
            "loss":         loss.item(),
        }
    def end_epoch(self):
        self.epoch += 1
    def begin_epoch(self):
        self.adjust_learning_rate()



sys.path.insert(-1, 'baselines/tcm')
from baselines.tcm.cyclegan.base_dataset import get_transform
from baselines.tcm.cyclegan.cycle_gan_model import CycleGANModel
from baselines.tcm.dda_model.dda_model import DDAModel
class TCM(Algorithm):
    """Transporting Cusal Mechanisms
        
        @misc{yue_transporting_2021,
            title = {Transporting Causal Mechanisms for Unsupervised Domain Adaptation},
            url = {http://arxiv.org/abs/2107.11055},
            doi = {10.48550/arXiv.2107.11055},
            number = {{arXiv}:2107.11055},
            publisher = {{arXiv}},
            author = {Yue, Zhongqi and Sun, Qianru and Hua, Xian-Sheng and Zhang, Hanwang},
            urldate = {2023-06-18},
            date = {2021-07-28},
            eprinttype = {arxiv},
            eprint = {2107.11055 [cs]},
            keywords = {Computer Science - Computer Vision and Pattern Recognition},
        }
    """
    CHECKPOINT_FREQ = 100
    def translate_opt(self, num_classes, num_domains, hparams):
        import argparse

        main_args = {
            'debug': False,
            'input_nc': 3,
            'output_nc': 3,
            'ngf': 64, 'ndf': 64,
            'netD': 'basic', 'netG': 'resnet_9blocks',
            'n_layers_D': 3,
            'norm': 'instance',
            'init_type': 'normal',
            'init_gain': 0.02,
            'no_dropout': False,
            'direction': 'AtoB',
            'load_size': 256, 'crop_size': 224,
            'max_dataset_size': float("inf"),
            'preprocess': 'resize_and_crop',
            'no_flip': False,

            'phase': 'train',

            'n_experts': 1,
            'expert_criteria': 'dc',
            'c_criteria_iterations': 5000,
            'i_criteria_iterations': 5000,
            'expert_warmup_mode': 'random',
            'expert_warmup_iterations': 999999,
            'lr_trick': 0,


            'lr': hparams['lr'],
            'beta1': hparams['beta1'],
            'gan_mode': 'lsgan',
            'pool_size': 50,
            'lr_policy': 'linear',
            'lr_decay_iters': 50,
            'aspect_ratio': 1.0,

            'gpu_ids': [0],
            'isTrain': True,
            'continue_train': False,
            'verbose': False,
            'checkpoints_dir': 'domainbed_test',
            'exp_name': 'none',
        }

        cyclegan_args = {
            'lambda_A': hparams['weight_cycleloss_ABA'],
            'lambda_B': hparams['weight_cycleloss_BAB'],
            'lambda_identity': hparams['weight_cycleloss_identity'],
            'lambda_diversity': hparams['weight_cycleloss_diversity'],
            'epoch_count': 1,
        }

        dda_args = {
            'backbone_train_mode': hparams['train_backbone'],
            'num_classes': num_classes,
            'resnet_name': 'ResNet50',
            'use_maxpool': False,
            'freeze_layer1': False,
            'z_dim': hparams['z_dim'],
            'backbone_lr': hparams['lr_backbone'],
            'vae_lr': hparams['lr_vae'],
            'linear_lr': hparams['lr_linear'],
            'linear_weight_decay': hparams['linear_weight_decay'],
            'linear_momentum': hparams['linear_momentum'],
            'dda_init_type': 'kaiming',
            'dda_init_gain': 0.02,
            'align_feature': False,
            'align_logits': True,
            'align_t2s': True,
            'discriminator_hidden_dim': 1024,
            'discriminator_lr': hparams['lr_d'],
            'gvbd_weight': 0.0,
            'gvbg_weight': 0.0,
            'alignment_weight': hparams['weight_align'],
            'backward_linear_loss': True,
            'accurate_mu': False,
            'no_entropy_weight': False,
            'label_smoothing': False,
            'beta_vae': 1.0,
            'bundle_transform': False,
            'bundle_resized_crop': False,
            'use_target_estimate': False,
            'use_linear_logits': True,
            'use_dda2': False,
            'use_dropout': False,
            'all_experts': False,
            'dda_checkpoints_dir': 'none',
            'dda_exp_name': 'none',
            'baseline': False,
            'no_mapping': 0,
            'pretrain_iteration': 0,

            'dda_continue_train': False
        }

        disciminator_args = {
            'cg_resnet_name': 'ResNet50',
            'cg_num_classes': num_classes,
            'cg_align_feature': False,
            'cg_align_logits': False,
            'cg_gvbd_weight': 0.0,
            'cg_gvbg_weight': 0.0
        }

        args = {**main_args, **cyclegan_args, **dda_args, **disciminator_args}
        return argparse.Namespace(**args)
        
    class SimplifiedCycleGANModel(CycleGANModel):
        def set_input(self, input):
            self.A_Y = input['A_Y']
            self.B_Y = input['B_Y']
            super().set_input(input)
        
        def update_input_classname(self):
            self.input_classnames = []
            for i in range(self.current_batch_size):
                self.input_classnames.append(str(int(self.A_Y[i])))
            for i in range(self.current_batch_size):
                self.input_classnames.append(str(int(self.B_Y[i])))
        def train(self):
            """Make models eval mode during test time"""
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    net.train()
        def backward_G(self, backward_loss=True):
            """Calculate the loss for generators G_A and G_B"""
            lambda_idt = self.opt.lambda_identity
            lambda_A = self.opt.lambda_A
            lambda_B = self.opt.lambda_B
            current_batch_size = self.real_A.shape[0]
            # Identity loss
            losses = torch.zeros(self.n_experts, current_batch_size * 2, device=self.device)   # n * batch_size
            self.loss_G_A = []
            self.loss_G_B = []
            self.loss_cycle_A = []
            self.loss_cycle_B = []
            self.loss_idt_A = []
            self.loss_idt_B = []
            for i in range(0, self.n_experts):
                if lambda_idt > 0:
                    # G_A should be identity if real_B is fed: ||G_A(B) - B||
                    self.idt_A = self.netG_A.get_expert(i)(self.real_B)
                    self.loss_idt_A.append(self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt)
                    # G_B should be identity if real_A is fed: ||G_B(A) - A||
                    self.idt_B = self.netG_B.get_expert(i)(self.real_A)
                    self.loss_idt_B.append(self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt)
                else:
                    self.loss_idt_A.append(torch.zeros(current_batch_size, 1, 1, 1, device=self.device))
                    self.loss_idt_B.append(torch.zeros(current_batch_size, 1, 1, 1, device=self.device))

                # GAN loss D_A(G_A(A))
                self.loss_G_A.append(self.criterionGAN(self.netD_A(self.fake_B_all[i]), True))
                # GAN loss D_B(G_B(B))
                self.loss_G_B.append(self.criterionGAN(self.netD_B(self.fake_A_all[i]), True))
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A.append(self.criterionCycle(self.rec_A_all[i], self.real_A) * lambda_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B.append(self.criterionCycle(self.rec_B_all[i], self.real_B) * lambda_B)
                losses[i] = self.get_expert_loss(i)
            expert_idx = self.get_expert_results(losses)    # 2batch_size
            self.loss_G = 0
            for i in range(0, current_batch_size):
                if self.opt.lr_trick:
                    a_expert_loss_scale = float(self.panel_tracker.sum() / float(self.panel_tracker[expert_idx[i]]) / float(self.n_experts))
                    self.loss_G += (self.loss_G_A[expert_idx[i]][i].mean() + self.loss_cycle_A[expert_idx[i]][i].mean() + self.loss_idt_A[expert_idx[i]][i].mean()) / a_expert_loss_scale
                    b_expert_loss_scale = float(self.panel_tracker.sum() / float(self.panel_tracker[expert_idx[i+current_batch_size]]) / float(self.n_experts))
                    self.loss_G += (self.loss_G_B[expert_idx[i+current_batch_size]][i].mean() + self.loss_cycle_B[expert_idx[i+current_batch_size]][i].mean() + self.loss_idt_B[expert_idx[i+current_batch_size]][i].mean()) / b_expert_loss_scale
                else:
                    self.loss_G += (self.loss_G_A[expert_idx[i]][i].mean() + self.loss_G_B[expert_idx[i+current_batch_size]][i].mean() + self.loss_cycle_A[expert_idx[i]][i].mean() + self.loss_cycle_B[expert_idx[i+current_batch_size]][i].mean() + self.loss_idt_A[expert_idx[i]][i].mean() + self.loss_idt_B[expert_idx[i+current_batch_size]][i].mean())
            self.loss_G /= float(current_batch_size)
        
            # select fake and reconstruct images
            fake_B = torch.zeros(self.fake_B_all[0].shape, device=self.device)
            rec_A = torch.zeros(self.rec_A_all[0].shape, device=self.device)
            fake_A = torch.zeros(self.fake_A_all[0].shape, device=self.device)
            rec_B = torch.zeros(self.rec_B_all[0].shape, device=self.device)
            for i in range(0, self.real_A.shape[0]):
                fake_B[i] = self.fake_B_all[expert_idx[i]][i]
                rec_A[i] = self.rec_A_all[expert_idx[i]][i]
                fake_A[i] = self.fake_A_all[expert_idx[i+current_batch_size]][i]
                rec_B[i] = self.rec_B_all[expert_idx[i+current_batch_size]][i]

            # convert loss to mean
            loss_diversity = 0
            for i in range(0, self.n_experts):
                self.panel_tracker[i] += (expert_idx == i).sum()
                self.epoch_panel_tracker[i] += (expert_idx == i).sum()
                self.loss_G_A[i] = self.loss_G_A[i].mean()
                self.loss_G_B[i] = self.loss_G_B[i].mean()
                self.loss_cycle_A[i] = self.loss_cycle_A[i].mean()
                self.loss_cycle_B[i] = self.loss_cycle_B[i].mean()
                self.loss_idt_A[i] = self.loss_idt_A[i].mean()
                self.loss_idt_B[i] = self.loss_idt_B[i].mean()
                loss_diversity += self.criterionDiversity(fake_B, self.fake_B_all[i]) * lambda_A\
                                + self.criterionDiversity(fake_A, self.fake_A_all[i]) * lambda_B
            loss_diversity /= -float(self.n_experts)
            self.loss_diversity = loss_diversity
            self.loss_G += loss_diversity * self.opt.lambda_diversity
            if backward_loss:
                self.loss_G.backward(retain_graph=self.discriminate_feature)

            self.fake_B = fake_B
            self.rec_A = rec_A
            self.fake_A = fake_A
            self.rec_B = rec_B
            self.expert_idx = expert_idx
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        opt = self.translate_opt(num_classes, num_domains, hparams)
        self.opt = opt
        print(opt)
        self.cyclegan = TCM.SimplifiedCycleGANModel(opt)
        # self.cyclegan.setup(opt)
        self.breaks = False
        self.early_stop_active_expert = False
        self.n_experts = 1

        self.transform_A = get_transform(self.opt, grayscale=False)
        self.transform_B = get_transform(self.opt, grayscale=False)

        from torchvision import transforms
        def remove_to_tensor(transform: transforms.Compose):
            assert isinstance(transform, transforms.Compose)
            transform.transforms = [t for t in transform.transforms if not isinstance(t, transforms.ToTensor)]

        remove_to_tensor(self.transform_A)
        remove_to_tensor(self.transform_B)

        self.step = 0
        self.n_cyclegan_step = hparams['n_cyclegan_step']

        self.dda = DDAModel(self.opt)
        self.dda.train_mode()
        # self.dda.setup()
    
    def update_cyclegan(self, minibatches, unlabeled):
        if self.breaks:
            return None
        assert self.cyclegan.isTrain
        self.cyclegan.train()
        
        def make_data(minibatches, unlabeled):
            assert unlabeled is not None    # DA with 1 target domain as B
            assert isinstance(unlabeled, list)
            A = torch.cat([x for x, y in minibatches], dim=0)
            Y_A = torch.cat([y for x, y in minibatches], dim=0)
            B = torch.cat([x for x in unlabeled], dim=0)
            A = self.transform_A(A)
            B = self.transform_B(B)

            return {
                'A': A,
                'B': B,
                'A_paths': None,
                'B_paths': None,
                'A_Y': Y_A,
                'B_Y': torch.full_like(Y_A, fill_value=-1)
            }
    
        self.cyclegan.set_input(make_data(minibatches, unlabeled))         # unpack data from dataset and apply preprocessing
        self.cyclegan.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        return {**dict(self.cyclegan.get_current_losses()), 'status': "CycleGAN", 'train_accuracy': 'none', 'losses': 'none'}
    
    def setup(self, start_epoch, n_epoch):
        from argparse import Namespace
        args = {
            **vars(self.opt),
            'n_epochs': n_epoch // 2,
            'n_epochs_decay': n_epoch // 2,
            'epoch_count': start_epoch,
        }
        self.cyclegan.setup(Namespace(**args))
        self.dda.setup()
    
    def begin_epoch(self):
        self.cyclegan.update_learning_rate()
    def end_epoch(self):
        if self.early_stop_active_expert:
            n_active_expert = (model.epoch_panel_tracker > 0).sum()
            if n_active_expert < self.n_experts - 1:
                self.breaks = True
        
        self.cyclegan.end_epoch()

    def generate(self, *, A=None, B=None):
        with torch.no_grad():
            self.cyclegan.eval()
            inputs = {'A': A, 'B': B, 'A_paths': None, 'B_paths': None, 'A_Y': torch.full([len(A)], -1, device=A.device), 'B_Y': torch.full([len(B)], -1, device=B.device)}
            self.cyclegan.set_input(inputs)
            self.cyclegan.forward()
            self.cyclegan.backward_G(backward_loss=False)
            assert len(self.cyclegan.fake_B_all) == 1, len(self.cyclegan.fake_B_all)
            assert len(self.cyclegan.fake_A_all) == 1, len(self.cyclegan.fake_A_all)
            return self.cyclegan.fake_B_all[0], self.cyclegan.fake_A_all[0]


    def update_dda(self, minibatches, unlabeled):
        def make_inputs(minibatches, unlabeled):
            assert len(minibatches) == 1    # DA with only 1 source domain as A
            assert unlabeled is not None    # DA with 1 target domain as B
            assert isinstance(unlabeled, list) and len(unlabeled) == 1
            A, Y = minibatches[0]
            B = unlabeled[0]

            s2t, t2s = self.generate(A=A, B=B)

            return A, Y, s2t, B, t2s

        self.dda.set_input(*make_inputs(minibatches, unlabeled))
        train_acc = self.dda.optimize()

        return {
            'status': 'DDA',
            'dda_train_accuracy': train_acc.item(),
            'losses': self.dda.get_current_losses()
        }

    def update(self, minibatches, unlabeled):
        if self.step < self.n_cyclegan_step:
            losses_cyclegan = self.update_cyclegan(minibatches, unlabeled)
        else:
            losses_cyclegan = {}
        losses_dda = self.update_dda(minibatches, unlabeled)
        self.step += 1
        return {
            **losses_cyclegan,
            **losses_dda
        }
    
    def predict(self, x):
        if self.opt.accurate_mu:
            self.dda.target_mu = self.dda.get_t2s_mu(test_loader)
        _, t2s= self.generate(A=x, B=x)
        self.dda.set_input(None, None, None, x, t2s)
        logits = self.dda.predict()
        return logits
    def train(self):
        self.dda.train_mode()
        self.cyclegan.train()

    def eval(self):
        self.dda.test_mode()
        self.cyclegan.eval()
        
        
