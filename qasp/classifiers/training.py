"""Small PyTorch training loop for QASP classifiers."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics


class ClassifierTrainer:
    """Train and evaluate :class:`~qasp.classifiers.PQCClassifier` models."""

    def __init__(
        self,
        model,
        *,
        objective: str | None = None,
        optimizer=None,
        learning_rate: float = 0.01,
    ) -> None:
        """Attach a classifier, objective and optimizer without starting training.

        ``objective=None`` uses model.default_objective. An explicit optimizer must
        own the model parameters; otherwise create Adam with learning_rate. Training
        and validation metrics accumulate in the mutable history dictionary."""
        self.model = model
        self.objective = model.default_objective if objective is None else objective
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=learning_rate
        )
        self.history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }

    @staticmethod
    def _tensors(inputs, targets):
        """Convert features to the default Torch floating dtype and targets to long integers; require equal leading batch sizes."""
        inputs = torch.as_tensor(inputs, dtype=torch.get_default_dtype())
        targets = torch.as_tensor(targets, dtype=torch.long)
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must have the same first dimension.")
        return inputs, targets

    def fit(
        self,
        inputs,
        targets,
        *,
        epochs: int = 20,
        batch_size: int | None = None,
        shuffle: bool = True,
        validation_data=None,
        verbose: int = 0,
    ) -> dict[str, list[float]]:
        """Optimize on (inputs, integer targets) and return the cumulative history dictionary.

        inputs has shape (N,features), targets (N,). epochs must be positive;
        batch_size=None uses full batches and shuffle controls the DataLoader.
        validation_data is an optional (inputs,targets) pair evaluated each epoch.
        verbose=0 is silent; positive values print at that epoch interval and the
        last epoch. History records sample-weighted train_loss and, when supplied,
        validation_loss/validation_accuracy. Repeated calls append to history.
        Training leaves the model in train mode unless validation switches it to
        eval mode; the next epoch explicitly restores train mode."""
        inputs, targets = self._tensors(inputs, targets)
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        if batch_size is None:
            batch_size = len(targets)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=batch_size,
            shuffle=shuffle,
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            seen = 0
            for batch_inputs, batch_targets in loader:
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(
                    batch_inputs, batch_targets, objective=self.objective
                )
                loss.backward()
                self.optimizer.step()
                batch_count = batch_targets.numel()
                total_loss += float(loss.detach()) * batch_count
                seen += batch_count
            epoch_loss = total_loss / seen
            self.history["train_loss"].append(epoch_loss)

            message = f"Epoch {epoch + 1}/{epochs}: loss={epoch_loss:.6f}"
            if validation_data is not None:
                evaluation = self.evaluate(*validation_data)
                self.history["validation_loss"].append(evaluation["loss"])
                self.history["validation_accuracy"].append(evaluation["accuracy"])
                message += (
                    f", val_loss={evaluation['loss']:.6f}, "
                    f"val_accuracy={evaluation['accuracy']:.4f}"
                )
            if verbose and ((epoch + 1) % verbose == 0 or epoch == epochs - 1):
                print(message, flush=True)
        return self.history

    @torch.no_grad()
    def evaluate(self, inputs, targets) -> dict[str, object]:
        """Evaluate a labeled dataset under no_grad and leave the model in eval mode.

        Returns loss, scalar/per-class metrics, confusion matrix, probabilities
        and integer predictions. Scalars become Python floats; non-scalar tensors
        are detached on CPU. This method does not split the dataset: callers must
        supply held-out rows if they want a generalization estimate."""
        inputs, targets = self._tensors(inputs, targets)
        self.model.eval()
        loss = self.model.compute_loss(inputs, targets, objective=self.objective)
        probabilities = self.model.predict_proba(inputs)
        predictions = probabilities.argmax(dim=-1)
        metrics = classification_metrics(
            targets,
            predictions,
            n_classes=self.model.n_classes,
            probabilities=probabilities,
        )
        result = {
            key: (float(value) if value.ndim == 0 else value.detach().cpu())
            for key, value in metrics.items()
        }
        result["loss"] = float(loss)
        result["probabilities"] = probabilities.detach().cpu()
        result["predictions"] = predictions.detach().cpu()
        return result


__all__ = ["ClassifierTrainer"]
