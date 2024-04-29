import lightning as L

__all__ = ["PanguLightningModule"]


class PanguLightningModule(L.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
