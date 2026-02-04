import torch
import lightning as L
import segmentation_models_pytorch as smp
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from collections import defaultdict

from losses import DiceLoss, FusionLoss, FocalLoss
from learningRate import CyclicLR, WarmUpLR, CosineLR  
import numpy as np
    
class UNetPPModule(L.LightningModule):
    """
    Lightning version of TrainerTMJ:
    - smp.UnetPlusPlus(resnet50, in_channels=1, classes=4)
    - FusionLoss([DiceLoss, FocalLoss])
    - custom LR schedule (Cosine + Cyclic + WarmUp) like TrainerTMJ.getLr()
    - optional prob image logging like TrainerTMJ.probImg()

    Expected batch (from TMJDataset.py):
      batch = {"image": (B,1,H,W), "mask": (B,H,W), "filename": ...}
    """

    def __init__(
        self,
        num_classes: int = 4,
        encoder_name: str = "resnet50",
        in_channels: int = 1,
        learning_rate: float = 1e-4,
        total_epochs: int = 1000,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.test_outputs = []
        self.num_classes = num_classes
        self.base_lr = learning_rate
        self.total_epochs = total_epochs
        self.class_id_to_name = {
            1: "Disc",
            2: "Condyle",
            3: "Eminence",
        }
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        loss_fn0 = DiceLoss(weights=[0.1, 5, 1, 1], use_softmax=True)
        loss_fn1 = FocalLoss(gamma=3)
        self.criterion = FusionLoss([loss_fn0, loss_fn1], weights=[0.5, 0.5], device="cpu") 

        self.iou_metrics = {
            "train": MeanIoU(num_classes=num_classes, per_class=False),
            "val": MeanIoU(num_classes=num_classes, per_class=False),
        }
        for stage, metric in self.iou_metrics.items():
            self.add_module(f"{stage}_mean_iou", metric)

        self.dice_metrics = {
            "train": GeneralizedDiceScore(num_classes=num_classes),
            "val": GeneralizedDiceScore(num_classes=num_classes),
        }
        for stage, metric in self.dice_metrics.items():
            self.add_module(f"{stage}_dice_score", metric)

        base_instance = CosineLR()
        cycle_instance = CyclicLR(cycle_at=[50, 200], lr_snippet=base_instance, decay=[0.5, 0.1])
        self.lr_instance = WarmUpLR(lr_instance=cycle_instance, warmup_frac=0.05)

        self.prob_msk = None

    def setup(self, stage: str | None = None):
        if hasattr(self.criterion, "device"):
            self.criterion.device = self.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _common_step(self, batch: dict, stage: str) -> torch.Tensor:
        images = batch["image"]          # (B,1,H,W)
        masks = batch["mask"]            # (B,H,W)

        logits = self(images)            # (B,C,H,W)
        loss = self.criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)  # (B,H,W)

        self.iou_metrics[stage].update(preds, masks)
        self.dice_metrics[stage].update(preds, masks)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int):
        images = batch["image"]
        masks = batch["mask"]
        patient_ids = batch["patient_id"]
        slice_idxs = batch["slice_idx"]

        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        out = {
            "preds": preds.detach().cpu(),
            "masks": masks.detach().cpu(),
            "patient_id": patient_ids,
            "slice_idx": slice_idxs,
        }

        self.test_outputs.append(out)
        return out


    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        logits = self(images)
        return torch.argmax(logits, dim=1)

    # ---------- epoch end metrics ----------
    def _common_epoch_end(self, stage: str):
        iou = self.iou_metrics[stage].compute()
        dice = self.dice_metrics[stage].compute()
        self.log(f"{stage}/iou", iou, prog_bar=True)
        self.log(f"{stage}/dice", dice, prog_bar=False)
        self.iou_metrics[stage].reset()
        self.dice_metrics[stage].reset()

    def on_train_epoch_end(self):
        self._common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_epoch_end("val")

    def on_test_epoch_end(self):
        patients = defaultdict(lambda: {"preds": {}, "gts": {}})

        for out in self.test_outputs:
            preds = out["preds"]
            gts = out["masks"]
            pids = out["patient_id"]
            sids = out["slice_idx"]

            if torch.is_tensor(sids):
                sids = sids.tolist()
            if torch.is_tensor(pids):
                pids = pids.tolist()

            for i in range(len(preds)):
                pid = str(pids[i])
                z = int(sids[i])
                patients[pid]["preds"][z] = preds[i]
                patients[pid]["gts"][z] = gts[i]

        dice_per_class = defaultdict(list)
        iou_per_class = defaultdict(list)

        for pid, data in patients.items():
            zs = sorted(data["preds"].keys())
            pred_vol = torch.stack([data["preds"][z] for z in zs])
            gt_vol = torch.stack([data["gts"][z] for z in zs])

            for cls in range(1, self.num_classes):
                pred_bin = (pred_vol == cls)
                gt_bin = (gt_vol == cls)

                if gt_bin.sum() == 0:
                    continue

                inter = (pred_bin & gt_bin).sum().float()
                union = pred_bin.sum().float() + gt_bin.sum().float() - inter

                dice = (2 * inter + 1e-8) / (pred_bin.sum() + gt_bin.sum() + 1e-8)
                iou  = (inter + 1e-8) / (union + 1e-8)

                dice_per_class[cls].append(dice.item())
                iou_per_class[cls].append(iou.item())

        dice_means, iou_means = [], []

        for cls in range(1, self.num_classes):
            if cls not in dice_per_class:
                continue

            class_name = self.class_id_to_name.get(cls, f"class_{cls}")

            d = float(np.mean(dice_per_class[cls]))
            i = float(np.mean(iou_per_class[cls]))

            self.log(f"test/dice_{class_name}", d)
            self.log(f"test/iou_{class_name}", i)

            dice_means.append(d)
            iou_means.append(i)

        self.log("test/dice_volume_per_class", float(np.mean(dice_means)))
        self.log("test/iou_volume_per_class",  float(np.mean(iou_means)))

        self.test_outputs.clear()

    # ---------- optimizer + custom LR schedule ----------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        def lr_lambda(epoch: int):
            lr_abs = float(self.lr_instance(epoch, self.total_epochs, self.base_lr))
            return lr_abs / float(self.base_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
