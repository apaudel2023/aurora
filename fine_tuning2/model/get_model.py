# fine_tuning2/model/get_model.py

import logging
import torch
from aurora import Aurora, AuroraSmall, AuroraHighRes, rollout as aurora_rollout


class AuroraModel:
    """
    Wrapper for loading one of the three pretrained Aurora variants,
    ensuring both model and batch live on the same device, and exposing
    a clean `predict` interface.
    """
    def __init__(
        self,
        model_type: str,
        use_lora: bool = False,
        device: str = "cuda",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device(device)

        self.logger.info("Initializing AuroraModel (type=%s, LoRA=%s) on %s",
                         model_type, use_lora, self.device)

        # Choose model variant
        if model_type == "small":
            self.model = AuroraSmall(use_lora=use_lora)
            ckpt = "aurora-0.25-small-pretrained.ckpt"
        elif model_type == "highres":
            self.model = AuroraHighRes(use_lora=use_lora)
            ckpt = "aurora-0.1-finetuned.ckpt"
        else:
            self.model = Aurora(use_lora=use_lora)
            ckpt = "aurora-0.25-pretrained.ckpt"

        repo = "microsoft/aurora"
        self.logger.info("Loading checkpoint %s/%s (strict=False)", repo, ckpt)
        # always strict=False now
        self.model.load_checkpoint(repo, ckpt, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model is in eval mode on %s", self.device)

    @torch.inference_mode()
    def predict(self, batch, steps: int):
        """
        Runs the official aurora_rollout, making sure both model and batch
        are on the same device.
        """
        # Move batch to the model's device
        batch = batch.to(self.device)
        self.logger.info("Running rollout: %d steps on %s", steps, self.device)

        preds = aurora_rollout(self.model, batch, steps=steps)
        preds_cpu = [p.to("cpu") for p in preds]
        self.logger.info("Rollout complete, returning %d preds to CPU", len(preds_cpu))
        return preds_cpu
