import torch
from metrics.metrics import MetricTracker
from utils.utils import read_json, write_json
from tqdm import tqdm
import os


class Trainer:
    def __init__(
        self,
        config,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        metrics=[],
        scheduler=None,
        start_epoch=0,
        train_history={},
    ):

        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.resume = config.TRAINER.resume_path
        self.metrics = metrics
        self.n_epochs = config.TRAINER.n_epochs
        self.log_interval = config.TRAINER.log_interval
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.train_history = train_history
        self.train_history_fn = os.path.join(
            self.config.TRAINER.save_dir, "train_history.json"
        )

        # resume from checkpoint (if self.resume is not None)
        if self.resume is not None:
            self._resume_checkpoint(self.resume)

    def fit(self):
        """
        Full training logic
        """
        log = "Start training for {} epochs".format(self.n_epochs)
        print(log)
        best_loss = float("inf")
        for epoch in range(self.start_epoch, self.n_epochs):

            # train epoch
            train_loss = self._train_epoch(epoch)
            log = "Epoch: {}/{} summary:".format(epoch, self.n_epochs)
            log += "\n>>>> {}: {:.6f} ".format(train_loss.name, train_loss.avg)

            # valid epoch
            valid_loss = self._valid_epoch()
            log += "\n>>>> {}: {:.6f} ".format(valid_loss.name, valid_loss.avg)
            print(log)

            # update train history
            self.update_history(metric_list=[train_loss, valid_loss])

            # check if valid loss improved
            is_best = False
            if valid_loss.avg < best_loss:
                is_best = True
                best_loss = valid_loss.avg

            # save model checkpoint (and best model weights if is_best==True)
            self.save_checkpoint(epoch, is_best)

    def _train_epoch(self, epoch):

        # initialize MetricTracker objects
        train_loss = MetricTracker(name="Train loss")

        self.model.train()
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            if (self.config.TRAINER.max_iter_per_epoch is not None) and (
                batch_idx > self.config.TRAINER.max_iter_per_epoch
            ):
                break
            # unpack batch
            anchor = {k: v.to(self.device) for k, v in batch["anchor"].items()}
            pos = {k: v.to(self.device) for k, v in batch["pos"].items()}
            neg = {k: v.to(self.device) for k, v in batch["neg"].items()}
            batch_size = anchor["image"].size(0)

            # Clean grads
            self.optimizer.zero_grad()

            # Forward pass
            anchor_emb = self.model(anchor)
            pos_emb = self.model(pos)
            neg_emb = self.model(neg)

            # Loss computation
            loss = self.loss_fn(anchor_emb, pos_emb, neg_emb)

            # Backward pass
            loss.backward()

            # Parameter optimization
            self.optimizer.step()

            # Update Metric tracker objects
            train_loss.update(loss.item(), count=batch_size)

            if batch_idx % self.log_interval == 0 and batch_idx > 0:
                log = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * batch_size,
                    len(self.train_loader.dataset),
                    100.0 * batch_idx / len(self.train_loader),
                    train_loss.avg,
                )

                print(log)

        return train_loss

    def _valid_epoch(self):
        # initialize MetricTracker objects
        valid_loss = MetricTracker(name="Valid loss")

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc="Validating",
                total=len(self.val_loader),
            ):
                # unpack batch
                anchor = {k: v.to(self.device) for k, v in batch["anchor"].items()}
                pos = {k: v.to(self.device) for k, v in batch["pos"].items()}
                neg = {k: v.to(self.device) for k, v in batch["neg"].items()}
                batch_size = anchor["image"].size(0)

                # Clean grads
                self.optimizer.zero_grad()

                # Forward pass
                anchor_emb = self.model(anchor)
                pos_emb = self.model(pos)
                neg_emb = self.model(neg)

                # Loss computation
                loss = self.loss_fn(anchor_emb, pos_emb, neg_emb)

                # Update Metric tracker objects
                valid_loss.update(loss.item(), count=batch_size)

            return valid_loss

    def save_checkpoint(self, epoch, is_best=False):
        """
        Saves the current model.

        Args:
            epoch (int): current epoch number
            is_best (bool): if True, rename the saved checkpoint to 'best_model_weights.pth'
        """

        state = {
            "epoch": epoch,
            "model": self.model,
            "optimizer": self.optimizer,
            "config": self.config,
            "train_history": self.train_history,
        }

        # save current model checkpoint
        # checkpoint_fn = os.path.join(self.config.TRAINER.save_dir, 'model_checkpoint_{}.pth'.format(epoch))
        checkpoint_fn = os.path.join(
            self.config.TRAINER.save_dir, "model_checkpoint.pth"
        )
        torch.save(state, checkpoint_fn)
        print("Saving checkpoint: {}".format(checkpoint_fn))

        # save current best model
        if is_best:
            best_path = os.path.join(
                self.config.TRAINER.save_dir, "best_model_weights.pth"
            )
            torch.save(self.model.state_dict(), best_path)
            print("Saving current best model: best_model_weights.pth")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        Args:
            resume_path (str): Checkpoint path for model to resume
        """

        print("Loading checkpoint: {} ...".format(resume_path))

        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.model = checkpoint["model"]
        self.optimizer = checkpoint["optimizer"]

        # load train history
        self.train_history = read_json(self.train_history_fn)

        print(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

    def update_history(self, metric_list):
        """Update and save train history"""

        # update train history
        for m in metric_list:
            if m.name not in self.train_history:
                self.train_history[m.name] = []

            self.train_history[m.name].append(m.avg)

        # save train history
        write_json(self.train_history, self.train_history_fn)

    def get_history(self):
        return self.train_history
