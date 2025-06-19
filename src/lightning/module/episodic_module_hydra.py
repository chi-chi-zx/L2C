import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

import src.utils.logging as logging
import torch.optim as optim

from src.solver.losses import *

logger = logging.get_logger("smart_canada_goose")


def prepare_train_inputs(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    domain_ids: torch.Tensor,
    metadata: torch.Tensor,
    support_ratio: float = 0.5,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Spilit the batch into the support set for adaptation and the query set for inference.
    During training
    - Each batch has three parts of data examples: support_positive, support_negative and query
    - Each support set have a subset of examples from a positive domian called support_positive and the rest of examples
        from multiple negative domains called support_negative

    Arguments:
        inputs (torch.Tensor): The images in a mini-batch. BxCxHxW
        labels (torch.Tensor): The labels in a mini-batch.
        domain_ids (torch.Tensor): The domain ids of images in a mini-batch.
        support_ratio (float): The ratio of the number of images as the support_positive.
    Returns:
    """
    # assert len(torch.unique(domain_ids)) == 1, ("The batch are not from the same domain"
    #                                             "")
    batch_size, _, _, _ = inputs.shape
    n_support = round(batch_size * support_ratio)

    support_inputs = inputs[:n_support]
    support_targets = labels[:n_support]
    support_domain_ids = domain_ids[:n_support]

    query_inputs = inputs[n_support:]
    query_targets = labels[n_support:]
    query_metadata = metadata[n_support:]
    query_domain_ids = domain_ids[n_support:]

    return (
        support_inputs,
        support_targets,
        support_domain_ids,
        query_inputs,
        query_targets,
        query_metadata,
        query_domain_ids,
    )


def prepare_test_inputs(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    metadata: torch.Tensor,
    domain_ids: torch.Tensor,
    support_size: int = 1,
    is_adapt: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_inputs = len(inputs)
    if num_inputs <= support_size:
        return inputs, labels, domain_ids, inputs, labels, metadata

    if not is_adapt:
        return inputs, labels, domain_ids, inputs, labels, metadata
    else:
        support_inputs = inputs[:support_size]
        support_targets = labels[:support_size]
        support_domain_ids = domain_ids[:support_size]
        return support_inputs, support_targets, support_domain_ids, inputs, labels, metadata


def get_domain_ids(batch, datamodule):
    metadata = batch[-1]
    domain_ids = datamodule.grouper.metadata_to_group(metadata.cpu())
    if not isinstance(domain_ids, torch.Tensor):
        raise ValueError(f"The current batch has not domain_ids: {batch, metadata}")
    return domain_ids


class DomainSpecificEpisodicLearningLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_func: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_support_ratio: float = 0.5,
        test_support_size: float = 1,
        text_loss_coeff: float = 1.0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = model

        self.loss_func = loss_func
        self.text_loss_coeff = text_loss_coeff
        self.train_support_ratio = train_support_ratio
        self.test_support_size = test_support_size

        self.validation_step_outputs = []
        self.text_embed_flag = False

        # for poverty dataset
        self.start_idx = 0
        self.end_idx = 3
        print(':: checking starting idx: ', self.start_idx, self.end_idx)
        
    def adapt_and_inference(self, query_imgs, support_imgs, support_domain_ids):
        _ = self.model.encode_prompt(
            support_imgs, support_domain_ids
        )
        im_feature, text_feature, log_scale, logits = self.model(query_imgs)
        return im_feature, text_feature, log_scale, logits


    def inference(self, query_imgs):
        im_feature, text_feature, log_scale, logits = self.model(query_imgs)
        return im_feature, text_feature, log_scale, logits

    def training_step(self, batch, batch_idx):
        inputs, labels, metadata = batch

        if not self.text_embed_flag:

            self.model.text_embedding(self.trainer.datamodule.label_names, self.trainer.datamodule.p_template, inputs.device, single_temp=False)
            self.text_embed_flag = True

        if self.trainer.datamodule.dataset.name == 'poverty':
            inputs_raw = inputs[:, self.start_idx:self.end_idx]
            inputs = self.trainer.datamodule.post_transform(inputs_raw, is_train=True).float()
            labels = labels.float()

        domain_ids = get_domain_ids(batch, self.trainer.datamodule)
        (
            support_inputs,
            _,
            support_domain_ids,
            query_inputs,
            query_targets,
            query_metadata,
            _,
        ) = prepare_train_inputs(inputs, labels, domain_ids, metadata, self.train_support_ratio)

        im_feature, text_feature, log_scale, logits = self.adapt_and_inference(
            query_inputs, support_inputs, support_domain_ids
        )


        if self.trainer.datamodule.dataset.name != 'poverty':
            ''' Final prediction '''
            preds = logits.argmax(dim=1, keepdim=True).view_as(query_targets)
            metrics, _ = self.trainer.datamodule.train_dataset.eval(
                preds.cpu(), query_targets.cpu(), metadata=query_metadata.cpu()
            )
            if isinstance(self.loss_func, ClipLoss):
                our_loss = self.loss_func(im_feature, text_feature[query_targets], log_scale)
            else:
                our_loss = self.loss_func(logits, query_targets)


            centroid = aux_tex.mean(dim=0)[None, :]
            dist = torch.norm(aux_tex - centroid, dim=1)
            text_loss = - dist.mean()

            text_loss_final = torch.pdist(text_feature.to(torch.float), p=2).pow(2.0).mul(
                -2.0).exp().mean()

            if type(our_loss) == list:
                loss_main = our_loss[0]
                dis_loss = our_loss[1]
            else:
                loss_main = our_loss
                dis_loss = 0.0

            loss = loss_main + self.text_loss_coeff * text_loss

            self.log("train_loss", loss, on_step=True, on_epoch=False)

        else:
            loss = self.loss_func(logits, query_targets)
            preds = logits.view_as(query_targets)
            metrics, _ = self.trainer.datamodule.train_dataset.eval(
                preds.cpu(), query_targets.cpu(), metadata=query_metadata.cpu()
            )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Each dataloader has multiple mini-batches of images from the same domain.

        For example, 'dataloader_ids' starts from 0, and all data examples from 0th domain would
        be produced in multiple mini-batches. As a result, `batch_idx` would be from 0 to N before
        the dataloader move to the next domain where its dataloader_ids ++
        """

        imgs, targets, metadata = batch

        if not self.text_embed_flag:
            self.model.text_embedding(self.trainer.datamodule.label_names, self.trainer.datamodule.p_template, imgs.device, single_temp=False)

            self.text_embed_flag = True

        if self.trainer.datamodule.dataset.name == 'poverty':
            imgs = imgs[:, self.start_idx:self.end_idx]
            imgs = self.trainer.datamodule.post_transform(imgs, is_train=False).float()

        domain_ids = get_domain_ids(batch, self.trainer.datamodule)
        if len(torch.unique(domain_ids)) != 1:
            raise ValueError(
                "The imgs in a mini-batch must come from a same domain. But the domain ids in the current batch is {}".format(
                    domain_ids
                )
            )
        domain_id = domain_ids.tolist()[0]

        if (
            batch_idx == 0
        ):  # Adapt on the support set first and then make inference on all images
            (
                support_imgs,
                support_targets,
                support_domain_ids,
                query_imgs,
                query_targets,
                query_metadata
            ) = prepare_test_inputs(
                imgs, targets, metadata, domain_ids, self.test_support_size, is_adapt=True
            )
            im_feature, text_feature, log_scale, logits = self.adapt_and_inference(
                query_imgs, support_imgs, support_domain_ids
            )

        else:  # After adaptation, the model need to make inference on all imgs
            (
                support_imgs,
                support_targets,
                support_domain_ids,
                query_imgs,
                query_targets,
                query_metadata
            ) = prepare_test_inputs(
                imgs, targets, metadata, domain_ids, self.test_support_size, is_adapt=False
            )
            im_feature, text_feature, log_scale, logits = self.inference(query_imgs)

        if self.trainer.datamodule.dataset.name == 'poverty':
            preds = logits.view_as(query_targets)
            preds_CLIP = preds.view_as(query_targets)
        else:
            preds = logits.argmax(dim=1, keepdim=True).view_as(query_targets)

        results = dict()
        results["logits"] = logits
        results["predictions"] = preds
        results["targets"] = query_targets
        results["domain_id"] = domain_id
        results["metadata"] = metadata

        self.validation_step_outputs.append(results)

    def on_validation_epoch_start(self) -> None:
        logger.info("Start validation ...")
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        # 1. Resume the experiment and skip validation
        if len(self.validation_step_outputs) <15:
            logger.info(
                f"There is no predictions in Epoch: {self.trainer.current_epoch}. Skip the validation."
            )

            self.log("r_all", 0)
            self.log("test_ood_acc_avg", 0)
            self.log('val_ood_f1_score', 0)
            return

        # 2. Save all predictions for post-offline-analysis
        # skip

        # 3. Calculate metrics in ood_test
        preds = []
        gts = []
        metadata = []
        preds_CLIP = []
        preds_aux = []
        for r in self.validation_step_outputs:
            if r["domain_id"] in self.trainer.datamodule.ood_test_domain_ids:
                preds.append(r["predictions"])
                gts.append((r["targets"]))
                metadata.append(r["metadata"])

        logger.info("OOD data number: {}".format(len(metadata)))

        ood_metrics, _ = self.trainer.datamodule.ood_test_dataset.eval(
            torch.cat(preds).cpu(), torch.cat(gts).cpu(), metadata=torch.cat(metadata).cpu()
        )

        
        if self.trainer.datamodule.dataset.name == "fmow": 
            metric_names = ["acc_avg", "acc_worst_region"]
        elif self.trainer.datamodule.dataset.name == "iwildcam":
            metric_names = ["acc_avg", "F1-macro_all"]
        elif self.trainer.datamodule.dataset.name == "poverty":
            metric_names = ["r_all", "r_wg"]
        else: # ToDo: Need to consider PovertyMap (regression) in the future
            metric_names = ["acc_avg"] 
        
        def log_eval_metrics(metrics, split="id", metric_names=["acc_avg", "F1-macro_all", "acc_worst_region"]):
            for metric_name in metric_names:
                self.log(f"test_{split}_{metric_name}", metrics[metric_name])
                logger.info(f"Epoch {self.trainer.current_epoch}: test_{split}_{metric_name} = {metrics[metric_name]}")

        log_eval_metrics(ood_metrics, split="ood", metric_names=metric_names)
        
        if self.trainer.datamodule.dataset.name == "poverty":
            self.log('r_all', ood_metrics['r_all'])
        # 4. Calculate metrics in id_test (if required)
        if self.trainer.datamodule.dataset.id_test_split:
            preds = []
            gts = []
            metadata = []
            for r in self.validation_step_outputs:
                if r["domain_id"] in self.trainer.datamodule.id_test_domain_ids:
                    preds.append(r["predictions"])
                    gts.append((r["targets"]))
                    metadata.append(r["metadata"])

            logger.info("ID data number: {}".format(len(metadata)))
            id_metrics, _ = self.trainer.datamodule.id_test_dataset.eval(
                torch.cat(preds).cpu(), torch.cat(gts).cpu(), metadata=torch.cat(metadata).cpu()
            )
            log_eval_metrics(id_metrics, split="id", metric_names=metric_names)

        self.validation_step_outputs.clear()

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
