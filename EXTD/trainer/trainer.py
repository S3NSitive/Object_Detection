import sys
import cv2
import torch
import numpy as np

sys.path.append("..")

from torchvision.utils import make_grid
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

from base import BaseTrainer
from utils import inf_loop
from data_loader import data_loaders


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # conv_size = [5, 10, 20, 40, 80, 160]
        conv_size = [160, 80, 40, 20, 10, 5]
        anchors = []
        # stride = 128
        stride = 4

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        col_idx = 0
        image = np.zeros((640, 640, 3), np.uint8)
        img_list = ImageList(image, [(640, 640)])

        for fm_size in conv_size:
            # anchor_gene = AnchorGenerator(sizes=((stride * 4),), aspect_ratios=((1.0),))
            anchor = Generate_Anchors(stride, fm_size)
            # anchor = anchor_gene(img_list, torch.randn(1, fm_size, fm_size))[0]
            # anchors.append(anchor_gene(img_list, torch.randn(1, fm_size, fm_size))[0])
            stride *= 2
            for box in anchor:
                # x1 = box[1]
                # y1 = box[0]
                # x2 = box[3]
                # y2 = box[2]
                x1 = box[1]
                y1 = box[0]
                x2 = box[3]
                y2 = box[2]
                print(x1, y1, x2, y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), colors[col_idx], 2)
            col_idx += 1

        anchors = torch.cat(anchors, 0)
        anchors[0]
        gt_box = torch.tensor([71., 42., 90., 62.], dtype=torch.float32)
        jaccard(gt_box, anchors)

        img = cv2.resize(image, (1000, 1000))
        cv2.imshow("anchors", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def Generate_Anchors(stride, fm_size):
    anchor_scale = torch.tensor([4], dtype=torch.float32)
    anchor_ratio = torch.tensor([1], dtype=torch.float32)

    center_y = torch.arange(stride / 2, fm_size * stride, stride)
    center_x = torch.arange(stride / 2, fm_size * stride, stride)

    x_vec, y_vec = torch.meshgrid([center_x, center_y])
    x_vec = x_vec.flatten()
    y_vec = y_vec.flatten()

    h_vec = stride * torch.mm(torch.sqrt(anchor_ratio).view(max(anchor_ratio.shape), 1),
                              anchor_scale.view(1, max(anchor_scale.shape))).flatten()
    w_vec = stride * torch.mm(torch.sqrt(1. / anchor_ratio).view(max(anchor_ratio.shape), 1),
                              anchor_scale.view(1, max(anchor_scale.shape))).flatten()

    anc_0 = (y_vec.view(y_vec.size(0), 1) - h_vec / 2.).flatten()
    anc_1 = (x_vec.view(x_vec.size(0), 1) - w_vec / 2.).flatten()
    anc_2 = (y_vec.view(y_vec.size(0), 1) + h_vec / 2.).flatten()
    anc_3 = (x_vec.view(x_vec.size(0), 1) + w_vec / 2.).flatten()

    anchors = torch.stack([anc_0, anc_1, anc_2, anc_3], dim=1)

    return anchors


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    return inter / union  # [A,B]


def main():
    conv_size = [160, 80, 40, 20, 10, 5]
    anchors = []
    # stride = 128
    stride = 4

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    col_idx = 0
    image = np.zeros((640, 640, 3), np.uint8)
    # img_list = ImageList(image, [(640, 640)])

    for fm_size in conv_size:
        # anchor_gene = AnchorGenerator(sizes=((stride * 4),), aspect_ratios=((1.0),))
        anchor = Generate_Anchors(stride, fm_size)
        # anchor = anchor_gene(img_list, torch.randn(1, fm_size, fm_size))[0]
        anchors.append(anchor)
        # anchors.append(anchor_gene(img_list, torch.randn(1, fm_size, fm_size))[0])
        stride *= 2
        for box in anchor:
            # x1 = box[1]
            # y1 = box[0]
            # x2 = box[3]
            # y2 = box[2]
            x1 = box[1]
            y1 = box[0]
            x2 = box[3]
            y2 = box[2]
            # print(x1, y1, x2, y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[col_idx], 2)
        col_idx += 1

    anchors.shape

    anchors = torch.cat(anchors, 0)
    gt_box = torch.tensor([[71., 42., 90., 62.]], dtype=torch.float32)
    po = jaccard(gt_box, anchors)
    for i in po:
        for j in i:
            if j != 0:
                print(j)

    img = cv2.resize(image, (1000, 1000))
    cv2.imshow("anchors", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # dataloader = data_loaders.WIDERDataLoader("./data/WIDER_FACE", "wider_face_train.mat", 1)
    # for idx, train_iter in enumerate(dataloader):
    #     data_loaders.show_landmarks(**train_iter)

if __name__ == '__main__':
    main()
