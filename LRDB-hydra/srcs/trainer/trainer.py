import logging
import torch
from torchvision.utils import make_grid
from .base import BaseTrainer
from srcs.utils import inf_loop
from srcs.logger import BatchMetrics


logger = logging.getLogger('trainer')

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        (self.data_loader, self.valid_data_loader) = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler

        self.train_metrics = BatchMetrics('loss', postfix='/train', writer=self.writer)
        self.valid_metrics = BatchMetrics('loss', postfix='/valid', writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            (dst, ref) = data
            ref, dst = ref.to(self.device), dst.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model((dst, ref))
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k+'/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        q = []
        sq = []
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                (dst, ref) = data
                ref, dst = ref.to(self.device), dst.to(self.device)
                target = target.to(self.device)

                output = self.model((dst, ref))
                loss = self.criterion(output, target)
                q.append(output.item())
                sq.append(target.item())
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)
                self.valid_metrics.update('loss', loss.item())
        val_log = self.valid_metrics.result()
        for met in self.metric_ftns:
            val_log.update({met.__name__+'/valid': met(q, sq)})

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            # epoch-based training
            total = len(self.data_loader.dataset)
            current = batch_idx * self.data_loader.batch_size
        except AttributeError:
            # iteration-based training
            total = self.len_epoch
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)
