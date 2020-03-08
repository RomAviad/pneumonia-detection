from torch.nn import Module, BCELoss, MSELoss


class DetectionLoss(Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = BCELoss()
        self.mse_loss = MSELoss()

    def forward(self, predictions, targets):
        classification = self.bce_loss(predictions[:, 0], targets[:, 0])
        loss_x = self.mse_loss(predictions[:, 1], targets[:, 1])
        loss_y = self.mse_loss(predictions[:, 2], targets[:, 2])
        loss_w = self.mse_loss(predictions[:, 3], targets[:, 3])
        loss_h = self.mse_loss(predictions[:, 4], targets[:, 4])
        total_loss = classification + loss_x + loss_y + loss_w + loss_h
        return total_loss
