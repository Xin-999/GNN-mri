"""
Ensemble Models for Brain Connectivity Prediction
=================================================
Combines predictions from multiple GNN architectures for maximum accuracy.

Strategies:
1. Simple averaging
2. Weighted averaging (learned from validation)
3. Stacking with meta-learner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble that learns optimal combination weights.
    Weights are optimized on validation set.
    """

    def __init__(self, n_models: int = 3):
        super().__init__()
        self.n_models = n_models
        # Learnable weights (will be softmaxed)
        self.weights = nn.Parameter(torch.ones(n_models) / n_models)

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, n_models) - predictions from each model
        Returns:
            ensemble_pred: (batch,) - weighted combination
        """
        weights = F.softmax(self.weights, dim=0)
        return (predictions * weights.unsqueeze(0)).sum(dim=-1)

    def get_weights(self) -> np.ndarray:
        """Get current model weights as numpy array."""
        return F.softmax(self.weights, dim=0).detach().cpu().numpy()


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with a meta-learner MLP.
    Takes predictions from base models and learns optimal combination.
    """

    def __init__(self, n_models: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.n_models = n_models

        # Meta-learner network
        self.meta_learner = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, n_models)
        Returns:
            ensemble_pred: (batch,)
        """
        return self.meta_learner(predictions).squeeze(-1)


class EnsembleModel:
    """
    Complete ensemble system for combining multiple trained models.

    Supports:
    - Simple averaging
    - Weighted averaging (optimized on validation)
    - Stacking with meta-learner
    - Model selection based on uncertainty
    """

    def __init__(
        self,
        models: List[nn.Module],
        model_names: List[str],
        ensemble_type: str = 'weighted',  # 'mean', 'weighted', 'stacking'
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            models: List of trained PyTorch models
            model_names: Names of models (for logging)
            ensemble_type: Type of ensemble strategy
            device: Device to run on
        """
        self.models = models
        self.model_names = model_names
        self.ensemble_type = ensemble_type
        self.device = device

        # Move models to device and set to eval
        for model in self.models:
            model.to(device)
            model.eval()

        # Initialize ensemble combiner
        if ensemble_type == 'weighted':
            self.combiner = WeightedEnsemble(n_models=len(models)).to(device)
        elif ensemble_type == 'stacking':
            self.combiner = StackingEnsemble(n_models=len(models)).to(device)
        else:
            self.combiner = None

    @torch.no_grad()
    def predict(self, data_loader) -> np.ndarray:
        """
        Make ensemble predictions on a dataset.

        Args:
            data_loader: PyTorch DataLoader
        Returns:
            predictions: (n_samples,) - ensemble predictions
        """
        all_predictions = []
        all_targets = []

        for batch in data_loader:
            batch = batch.to(self.device)

            # Get predictions from all models
            model_preds = []
            for model in self.models:
                pred = model(batch)
                model_preds.append(pred)

            # Stack predictions: (batch, n_models)
            model_preds = torch.stack(model_preds, dim=-1)

            # Combine predictions
            if self.ensemble_type == 'mean':
                ensemble_pred = model_preds.mean(dim=-1)
            else:
                ensemble_pred = self.combiner(model_preds)

            all_predictions.append(ensemble_pred.cpu())
            all_targets.append(batch.y.cpu())

        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()

        return predictions, targets

    def optimize_weights(
        self,
        val_loader,
        n_epochs: int = 100,
        lr: float = 0.01
    ):
        """
        Optimize ensemble weights on validation set.
        Only applicable for 'weighted' and 'stacking' ensemble types.
        """
        if self.combiner is None:
            if verbose:
                print("Using simple mean ensemble - no weights to optimize.")
            return

        optimizer = torch.optim.Adam(self.combiner.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in val_loader:
                batch = batch.to(self.device)

                # Get base model predictions (no gradients)
                with torch.no_grad():
                    model_preds = []
                    for model in self.models:
                        pred = model(batch)
                        model_preds.append(pred)
                    model_preds = torch.stack(model_preds, dim=-1)

                # Forward through combiner (with gradients)
                ensemble_pred = self.combiner(model_preds)
                loss = criterion(ensemble_pred, batch.y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.4f}")

        if verbose:
            if self.ensemble_type == 'weighted':
                weights = self.combiner.get_weights()
                print("\nOptimized weights:")
                for name, weight in zip(self.model_names, weights):
                    print(f"  {name}: {weight:.4f}")

    @torch.no_grad()
    def predict_with_uncertainty(self, data_loader) -> tuple:
        """
        Make predictions with uncertainty estimates.
        Uncertainty is measured as disagreement between models.

        Returns:
            predictions: (n_samples,) - ensemble predictions
            targets: (n_samples,) - ground truth
            uncertainties: (n_samples,) - prediction uncertainties (std across models)
        """
        all_predictions = []
        all_targets = []
        all_uncertainties = []

        for batch in data_loader:
            batch = batch.to(self.device)

            # Get predictions from all models
            model_preds = []
            for model in self.models:
                pred = model(batch)
                model_preds.append(pred)

            # Stack: (batch, n_models)
            model_preds = torch.stack(model_preds, dim=-1)

            # Compute uncertainty as standard deviation
            uncertainty = model_preds.std(dim=-1)

            # Combine predictions
            if self.ensemble_type == 'mean':
                ensemble_pred = model_preds.mean(dim=-1)
            else:
                ensemble_pred = self.combiner(model_preds)

            all_predictions.append(ensemble_pred.cpu())
            all_targets.append(batch.y.cpu())
            all_uncertainties.append(uncertainty.cpu())

        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()
        uncertainties = torch.cat(all_uncertainties).numpy()

        return predictions, targets, uncertainties

    def save(self, save_path: Path):
        """Save ensemble combiner weights."""
        if self.combiner is not None:
            torch.save({
                'combiner_state': self.combiner.state_dict(),
                'ensemble_type': self.ensemble_type,
                'model_names': self.model_names,
            }, save_path)

    def load(self, load_path: Path):
        """Load ensemble combiner weights."""
        if self.combiner is not None:
            checkpoint = torch.load(load_path, map_location=self.device)
            self.combiner.load_state_dict(checkpoint['combiner_state'])


def load_ensemble_from_checkpoints(
    checkpoint_paths: Dict[str, Path],
    model_classes: Dict[str, type],
    model_configs: Dict[str, dict],
    ensemble_type: str = 'weighted',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> EnsembleModel:
    """
    Utility function to load ensemble from saved checkpoints.

    Args:
        checkpoint_paths: Dict mapping model_name -> checkpoint path
        model_classes: Dict mapping model_name -> model class
        model_configs: Dict mapping model_name -> model config dict
        ensemble_type: Type of ensemble
        device: Device to load on

    Returns:
        EnsembleModel ready for inference

    Example:
        >>> from models import BrainGNN, BrainGT, FBNetGenFromGraph
        >>> checkpoint_paths = {
        ...     'braingnn': Path('models/braingnn_best.pt'),
        ...     'braingt': Path('models/braingt_best.pt'),
        ...     'fbnetgen': Path('models/fbnetgen_best.pt'),
        ... }
        >>> model_classes = {
        ...     'braingnn': BrainGNN,
        ...     'braingt': BrainGT,
        ...     'fbnetgen': FBNetGenFromGraph,
        ... }
        >>> model_configs = {
        ...     'braingnn': {'in_dim': 268, 'hidden_dim': 128},
        ...     'braingt': {'in_dim': 268, 'hidden_dim': 128},
        ...     'fbnetgen': {'in_dim': 268, 'hidden_dim': 128},
        ... }
        >>> ensemble = load_ensemble_from_checkpoints(
        ...     checkpoint_paths, model_classes, model_configs
        ... )
    """
    models = []
    model_names = []

    for name, ckpt_path in checkpoint_paths.items():
        # Initialize model
        model_class = model_classes[name]
        model_config = model_configs[name]
        model = model_class(**model_config)

        # Load checkpoint
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded {name} from {ckpt_path}")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, using random init")

        models.append(model)
        model_names.append(name)

    return EnsembleModel(
        models=models,
        model_names=model_names,
        ensemble_type=ensemble_type,
        device=device,
    )


def evaluate_ensemble(
    ensemble: EnsembleModel,
    test_loader,
    save_results: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate ensemble model and compute metrics.

    Args:
        ensemble: EnsembleModel instance
        test_loader: Test data loader
        save_results: Optional path to save detailed results

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    predictions, targets, uncertainties = ensemble.predict_with_uncertainty(test_loader)

    # Compute metrics
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r, p_value = pearsonr(targets, predictions)

    metrics = {
        'mse': mse,
        'mae': mae,
        'r': r,
        'p_value': p_value,
        'mean_uncertainty': uncertainties.mean(),
    }

    print("\n=== Ensemble Performance ===")
    print(f"MSE:         {mse:.4f}")
    print(f"MAE:         {mae:.4f}")
    print(f"Correlation: {r:.4f} (p={p_value:.4e})")
    print(f"Mean Uncertainty: {uncertainties.mean():.4f}")

    # Save detailed results
    if save_results:
        import pandas as pd
        results_df = pd.DataFrame({
            'prediction': predictions,
            'target': targets,
            'uncertainty': uncertainties,
        })
        results_df.to_csv(save_results, index=False)
        print(f"\nDetailed results saved to {save_results}")

    return metrics
