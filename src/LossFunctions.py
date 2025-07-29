import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Union
from torch.cuda.amp import autocast
# import pytorch_ssim  # You might need to install this: pip install pytorch-ssim
import pytorch_msssim  # You might need to install this: pip install pytorch-ssim
import numpy as np
from scipy.ndimage import distance_transform_edt

class BaseLoss(nn.Module, ABC):
    def __init__(self, weights: torch.Tensor = None, 
                 temperature: float = 1.0, device: str = None):
        super().__init__()
        self.device = torch.device(device)
        self.temperature = temperature
        self.weights = weights
        if weights is not None:
            self.weights = self.weights.to(self.device)
            
    @abstractmethod
    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]],
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        pass 

class IoULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, smooth: float = 1e-6, 
                 device: str = None, temperature: float = 1.0, *args, **kwargs):
        
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.smooth = smooth

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        pred_seg = pred['seg']
        B, C, H, W = pred_seg.shape

        pred_seg = torch.softmax(pred_seg/self.temperature, dim=1)

        target = target.long()
        target = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        pred_seg = pred_seg.view(B, C, -1)
        target = target.view(B, C, -1)

        intersection = (pred_seg * target).sum(dim=2)
        union = pred_seg.sum(dim=2) + target.sum(dim=2) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)

        if self.weights is not None:
            iou_loss = (1 - iou) * self.weights.view(1, C)  # Expand weights to match (B, C)
            return iou_loss.sum() / (self.weights.sum() + self.smooth)  # Avoid division by zero
        
        return (1 - iou).mean()

class DiceLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, 
                 smooth: float = 1e-6, device: str = None, 
                 temperature: float = 1.0, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.smooth = smooth

    def forward(self,pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        pred_seg = pred['seg']
        
        B, C, H, W = pred_seg.shape

        # Apply softmax for multi-class probabilities
        pred_seg = torch.softmax(pred_seg/self.temperature, dim=1)

        # Ensure target is long for one-hot encoding
        target = target.long()
        target = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        # Flatten to (B, C, H*W)
        pred_seg = pred_seg.view(B, C, -1)
        target = target.view(B, C, -1)

        # Compute Dice Score
        intersection = (pred_seg * target).sum(dim=2)
        dice = (2 * intersection + self.smooth) / (pred_seg.sum(dim=2) + target.sum(dim=2) + self.smooth)

        # Apply class weights if provided
        if self.weights is not None:
            dice_loss = (1 - dice) * self.weights.view(1, C)
            return dice_loss.sum() / (self.weights.sum() + self.smooth)  # Normalize by valid weights
        
        return (1 - dice).mean()  # Average Dice loss over classes
    
class FocalLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, alpha:float=1, gamma:float=3, 
                 device: str = None, temperature: float = 1.0, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.alpha = alpha  # Class balancing (e.g., [1.0, 2.0, 5.0, 10.0])
        self.gamma = gamma  # Focusing parameter
        self.ce_loss = nn.CrossEntropyLoss(weight=weights, reduction="none")
        
    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        # ce_loss = F.cross_entropy(inputs, targets, self.weights, reduction="none")
        pred_seg = pred['seg']
        ce_loss = self.ce_loss(pred_seg/self.temperature, target)  # Compute CrossEntropy loss
        pt = torch.exp(-ce_loss)  # p_t = probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class CrossEntrophyLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 temperature: float = 1.0, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        pred_seg = pred['seg']

        return self.ce_loss(pred_seg / self.temperature, target)


class LDMCE_Loss(nn.Module):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 temperature: float = 1.0, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)

    def compute_inverse_distance_map(self, mask_np):
        H, W = mask_np.shape
        distance_map = np.zeros((H, W), dtype=np.float32)

        for cls in range(self.num_classes):
            binary_mask = (mask_np == cls).astype(np.uint8)
            dist_outside = distance_transform_edt(1 - binary_mask)
            dist_inside = distance_transform_edt(binary_mask)
            boundary_dist = np.minimum(dist_outside, dist_inside)
            distance_map = np.maximum(distance_map, boundary_dist)

        return 1.0 / (distance_map + 1.0)

    def forward(self, input, target):
        N, C, H, W = input.shape

        # Compute inverse distance maps
        distance_maps = []
        for i in range(N):
            mask_np = target[i].cpu().numpy()
            dist_map = self.compute_inverse_distance_map(mask_np)
            distance_maps.append(torch.tensor(dist_map, device=input.device))

        distance_map = torch.stack(distance_maps)  # (N, H, W)

        input_flat = input.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.view(-1)
        distance_flat = distance_map.view(-1)

        # Log-softmax
        log_probs_flat = F.log_softmax(input_flat, dim=1)
        log_probs_true = log_probs_flat[torch.arange(log_probs_flat.shape[0]), target_flat]

        # Distance weighting
        weight = 1 + distance_flat

        # Apply class weights if given
        if self.class_weights is not None:
            class_weights_tensor = self.class_weights.to(input.device)
            class_weights_flat = class_weights_tensor[target_flat]  # (N*H*W,)
            weight *= class_weights_flat

        # Weighted loss
        weighted_log_probs = weight * log_probs_true
        loss = -weighted_log_probs.mean()

        return loss


class N3LLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.nll_loss = nn.NLLLoss(weight=weights)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        """
        inputs: (B, C, H, W) - Log probabilities (output of log_softmax)
        targets: (B, H, W) - Class indices
        """
        # Apply log_softmax if it hasn't been applied in the model
        pred_seg = pred['seg']
        log_probs = F.log_softmax(pred_seg/self.temperature, dim=1)
        
        # Compute loss
        return self.nll_loss(log_probs, target)

class WeightedLovaszSoftmaxLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 temperature: float = 1.0, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weights)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.weight_lvz = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.weight_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    @staticmethod
    def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = gt_sorted.size(0)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax_flat(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        probs: [C, H*W] probabilities for each class
        labels: [H*W] ground truth labels
        """
        C = probs.size(0)
        losses = []

        for c in range(C):
            fg = (labels == c).float()  # foreground for class c
            if fg.sum() == 0:
                continue

            class_pred = probs[c]
            errors = torch.abs(fg - class_pred)
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self.lovasz_grad(fg_sorted)

            class_loss = torch.dot(errors_sorted, grad)
            weight = self.weights[c] if self.weights is not None else 1.0
            losses.append(weight * class_loss)

        if not losses:
            return torch.tensor(0.0, device=probs.device)
        return torch.mean(torch.stack(losses))

    def forward(self, pred: dict[str, torch.Tensor], target: torch.Tensor = None, depth=None):
        """
        pred: dict containing 'seg' -> [B, C, H, W] logits
        target: [B, H, W] ground truth labels
        """
        pred_seg = pred['seg']
        B, C, H, W = pred_seg.shape

        ce = self.ce_loss(pred_seg, target)

        probs = F.softmax(pred_seg / self.temperature, dim=1)
        probs_flat = probs.view(B, C, -1)
        target_flat = target.view(B, -1)

        lovasz_losses = []
        for i in range(B):
            loss = self.lovasz_softmax_flat(probs_flat[i], target_flat[i])
            lovasz_losses.append(loss)

        lovasz = torch.mean(torch.stack(lovasz_losses)) if lovasz_losses else 0.0
  
        loss = (torch.exp(-self.weight_ce) * ce + self.weight_ce +
                torch.exp(-self.weight_lvz) * lovasz + self.weight_lvz
            ) 
        return loss
    
class LovaszSoftmaxLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 temperature: float = 1.0, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weights)

    @staticmethod
    def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = gt_sorted.size(0)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax_flat(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        probs: [C, H*W] probabilities for each class
        labels: [H*W] ground truth labels
        """
        C = probs.size(0)
        losses = []

        for c in range(C):
            fg = (labels == c).float()  # foreground for class c
            if fg.sum() == 0:
                continue

            class_pred = probs[c]
            errors = torch.abs(fg - class_pred)
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self.lovasz_grad(fg_sorted)

            class_loss = torch.dot(errors_sorted, grad)
            weight = self.weights[c] if self.weights is not None else 1.0
            losses.append(weight * class_loss)

        if not losses:
            return torch.tensor(0.0, device=probs.device)
        return torch.mean(torch.stack(losses))

    def forward(self, pred: dict[str, torch.Tensor], target: torch.Tensor = None, depth=None):
        """
        pred: dict containing 'seg' -> [B, C, H, W] logits
        target: [B, H, W] ground truth labels
        """
        pred_seg = pred['seg']
        B, C, H, W = pred_seg.shape

        ce = self.ce_loss(pred_seg, target)

        probs = F.softmax(pred_seg / self.temperature, dim=1)
        probs_flat = probs.view(B, C, -1)
        target_flat = target.view(B, -1)

        lovasz_losses = []
        for i in range(B):
            loss = self.lovasz_softmax_flat(probs_flat[i], target_flat[i])
            lovasz_losses.append(loss)

        lovasz = torch.mean(torch.stack(lovasz_losses)) if lovasz_losses else 0.0
  
        return lovasz + ce

class StructuralConsistencyLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 temperature: float = 1.0, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor=None, depth: torch.Tensor = None) -> torch.Tensor:
        # def forward(self, enhanced_depth: torch.Tensor, rgb_image: torch.Tensor) -> torch.Tensor:
        org_rgb = pred['rgb']
        dpt_txt = pred['dpt_txt']
        dpt_txt_exp = dpt_txt.expand_as(org_rgb)

        ssim_loss = 1 - pytorch_msssim.ssim(dpt_txt_exp, org_rgb, data_range=1.0)
        # 1 - pytorch_ssim.ssim(depth_3ch, pred_seg)  # SSIM returns a per-image SSIM
        return ssim_loss



class EllipseLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 temperature: float = 1.0, eps: float = 1e-6, clses: list[int] = None, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        self.temperature = temperature

        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses

    def forward(self, pred: Union[torch.Tensor, dict], 
                target: torch.Tensor = None, depth:torch.Tensor=None) -> torch.Tensor:
        """
        pred: Either a tensor (B,C,H,W) logits or dict with key 'seg' for logits.
        target: (unused) included for API compatibility.
        Returns scalar ellipse loss applied on classes in self.clses.
        """
        pred_seg = pred['seg']
        
        B, C, H, W = pred_seg.shape
        device = pred_seg.device

        # Softmax with temperature scaling
        pred_soft = torch.softmax(pred_seg / self.temperature, dim=1)  # (B,C,H,W)

        # Create coordinate grid normalized to [0,1]
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (H*W, 2)

        total_loss = 0.0
        count = 0

        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c].flatten()  # (H*W)
                mass = mask.sum() + self.eps
                if mass < self.eps * 10:
                    continue
                
                # Weighted mean (ellipse center)
                mu = (mask.unsqueeze(1) * coords).sum(dim=0) / mass
                
                diffs = coords - mu.unsqueeze(0)  # (H*W, 2)
                weighted_diffs = diffs * mask.unsqueeze(1)
                cov = (weighted_diffs.t() @ diffs) / mass
                cov += self.eps * torch.eye(2, device=device)
                
                cov_inv = torch.linalg.inv(cov)
                dists = torch.sqrt(torch.sum((diffs @ cov_inv) * diffs, dim=1) + self.eps)
                
                penalty = mask * torch.relu(dists - 1) ** 2
                total_loss += penalty.sum() / mass
                count += 1

        if count == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        
        return total_loss / count



class EllipseLossWithGT(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 temperature: float = 1.0, eps: float = 1e-6, clses: list[int] = None,
                 lambda_cov: float = 1.0, lambda_center: float = 1.0, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        self.temperature = temperature
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses
        self.lambda_cov = lambda_cov
        self.lambda_center = lambda_center

    def _compute_ellipse_params(self, mask: torch.Tensor, coords: torch.Tensor):
        mass = mask.sum() + self.eps
        mu = (mask.unsqueeze(1) * coords).sum(dim=0) / mass
        diffs = coords - mu.unsqueeze(0)
        weighted_diffs = diffs * mask.unsqueeze(1)
        cov = (weighted_diffs.t() @ diffs) / mass
        cov += self.eps * torch.eye(2, device=mask.device)
        return mu, cov

    def forward(self, pred: Union[torch.Tensor, dict], 
                target: torch.Tensor,  # (B, H, W) with int class labels
                depth: torch.Tensor = None) -> torch.Tensor:
        pred_seg = pred['seg']  # (B, C, H, W)
        B, C, H, W = pred_seg.shape
        device = pred_seg.device

        pred_soft = torch.softmax(pred_seg / self.temperature, dim=1)

        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (H*W, 2)

        total_loss = 0.0
        count = 0

        for b in range(B):
            for c in self.clses:
                # Create binary GT mask for class c
                gt_mask = (target[b] == c).to(dtype=torch.float32).flatten()  # (H*W)
                pred_mask = pred_soft[b, c].flatten()

                if gt_mask.sum() < self.eps * 10:
                    continue

                mu_gt, cov_gt = self._compute_ellipse_params(gt_mask, coords)
                mu_pred, cov_pred = self._compute_ellipse_params(pred_mask, coords)

                center_loss = torch.sum((mu_pred - mu_gt) ** 2)
                cov_loss = torch.sum((cov_pred - cov_gt) ** 2)

                cov_gt_inv = torch.linalg.inv(cov_gt)
                diffs_pred = coords - mu_gt.unsqueeze(0)
                dists_gt = torch.sqrt(torch.sum((diffs_pred @ cov_gt_inv) * diffs_pred, dim=1) + self.eps)
                penalty = pred_mask * torch.relu(dists_gt - 1) ** 2
                pixel_penalty = penalty.sum() / (pred_mask.sum() + self.eps)

                loss = (self.lambda_center * center_loss + 
                        self.lambda_cov * cov_loss +
                        pixel_penalty)

                total_loss += loss
                count += 1

        if count == 0:
            return torch.tensor(0., device=device, requires_grad=True)
        return total_loss / count

class ConvexityEnvelopeLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list=None, kernel_size=21, eps=1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.kernel_size = kernel_size
        self.eps = eps
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None):
        pred_seg = pred['seg']
        B, C, H, W = pred_seg.shape
        pred_soft = torch.softmax(pred_seg/self.temperature, dim=1)
        total_loss = 0.0
        count = 0

        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c:c+1]  # (1, H, W)

                # Soft convex envelope via max pooling (convex-like hull)
                envelope = F.max_pool2d(mask, kernel_size=self.kernel_size, 
                                        stride=1, padding=self.kernel_size // 2)

                # Penalize predictions that spill outside the convex envelope
                diff = F.relu(mask - envelope)
                loss = diff.mean()
                # loss = loss / (H * W)  # Normalize by area
                
                total_loss += loss 
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_seg.device, requires_grad=True)

class ConvexityLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        
        self.clses = clses

        # Precompute Sobel filters
        self.sobel_x = torch.tensor([[1, 0, -1], 
                                     [2, 0, -2], 
                                     [1, 0, -1]], 
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[1, 2, 1], 
                                     [0, 0, 0], 
                                     [-1, -2, -1]], 
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor = None, depth: torch.Tensor = None) -> torch.Tensor:
        pred_seg = pred['seg']
        B, C, H, W = pred_seg.shape
        
        pred_soft = torch.softmax(pred_seg/self.temperature, dim=1)  # (B, C, H, W)

        total_loss = 0.0
        count = 0

        sobel_x = self.sobel_x.to(pred_seg.device)
        sobel_y = self.sobel_y.to(pred_seg.device)
        
        for b in range(B):
            for c in self.clses:
                prob_map = pred_soft[b, c]  # (H, W)

                area = torch.clamp(prob_map.sum(), min=1e-12)

                # Compute Sobel gradients
                prob_map_ = prob_map.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                dx = F.conv2d(prob_map_, sobel_x, padding=1)[0, 0]
                dy = F.conv2d(prob_map_, sobel_y, padding=1)[0, 0]

                grad_mag = torch.sqrt(torch.clamp(dx ** 2 + dy ** 2, min=1e-12))
                perimeter = grad_mag.sum()

                convexity = torch.log(1 + perimeter / (torch.sqrt(area) + self.eps))
                # convexity = convexity / (H * W)

                total_loss += convexity
                count += 1

        if count == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=pred_seg.device, requires_grad=True)

        return total_loss / count

class ConvexitySoftMorphologicalGTClosingLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list[str]=None, kernel_size:int=21, eps=1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.kernel_size = kernel_size
        self.eps = eps
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses

         # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.gt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.msk = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def soft_closing(self, mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Applies soft morphological closing (dilation followed by erosion) to a mask.
        """
        # Dilation: max pooling
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        # Erosion: min pooling (inverted max pooling)
        eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        return eroded  # the "closed" version of the original mask

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
        """
        Forward pass for the Soft Morphological Closing Loss.
        """
        pred_seg = pred['seg']
        B, C, H, W = pred_seg.shape
        pred_soft = torch.softmax(pred_seg/self.temperature, dim=1)
        
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c:c+1]  # (1, H, W)

                # Apply soft morphological closing to the mask
                # closing = self.soft_closing(mask, kernel_size=self.kernel_size)
                
                # Penalize underfilled regions (holes)
                # diff = F.relu(closing - mask)  # penalizes when closing > mask (missing pixels inside)
                # loss = diff.mean()

                closing_pred = self.soft_closing(mask, kernel_size=self.kernel_size)
                loss1 = F.relu(closing_pred - mask).mean()

                if target is not None:
                    target_seg = target['seg']
                    gt_mask = (target_seg[b:b+1] == c).float()  # (1, H, W)
                    closing_gt = self.soft_closing(gt_mask, kernel_size=self.kernel_size)
                    loss2 = F.relu(closing_pred - closing_gt).mean()
                else:
                    loss2 = 0.0
                
                total_loss += (torch.exp(-self.gt)  * loss1 + self.gt +
                        torch.exp(-self.msk) * loss2 + self.msk)
                count += 1
        
        # Return the average loss
        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_seg.device, requires_grad=True)

class ConvexitySoftMorphologicalClosingLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list[str]=None, kernel_size:int=21, eps=1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.kernel_size = kernel_size
        self.eps = eps
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses

    def soft_closing(self, mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Applies soft morphological closing (dilation followed by erosion) to a mask.
        """
        # Dilation: max pooling
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        # Erosion: min pooling (inverted max pooling)
        eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        return eroded  # the "closed" version of the original mask

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
        """
        Forward pass for the Soft Morphological Closing Loss.
        """
        pred_seg = pred['seg']
        B, C, H, W = pred_seg.shape
        pred_soft = torch.softmax(pred_seg/self.temperature, dim=1)
        
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c:c+1]  # (1, H, W)

                # Apply soft morphological closing to the mask
                closing = self.soft_closing(mask, kernel_size=self.kernel_size)
                
                # Penalize underfilled regions (holes)
                diff = F.relu(closing - mask)  # penalizes when closing > mask (missing pixels inside)
                loss = diff.mean()

                total_loss += loss
                count += 1
        
        # Return the average loss
        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_seg.device, requires_grad=True)

class MultiKernelConvexitySoftMorphologicalBalancedClosingLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list[int] = None, kernel_sizes: list[int] = [3, 7, 15, 21, 41, 51, 61, 71], eps: float = 1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.kernel_sizes = kernel_sizes
        self.eps = eps
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses

    def soft_closing(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Applies soft morphological closing (dilation followed by erosion) to a mask.
        """
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return eroded

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
        """
        Forward pass for the Soft Morphological Closing Loss (multi-scale).
        """
        pred_seg = pred['seg']  # (B, C, H, W)
        B, C, H, W = pred_seg.shape
        pred_soft = torch.softmax(pred_seg / self.temperature, dim=1)

        total_loss = 0.0
        count = 0

        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

                # Apply multiple kernel sizes and average the closings
                closing_sum = 0.0
                for k in self.kernel_sizes:
                    closing_sum += self.soft_closing(mask, kernel_size=k)
                closing = closing_sum / len(self.kernel_sizes)

                # Compare the closing with the original mask
                mask = mask.squeeze(0).squeeze(0)       # (H, W)
                closing = closing.squeeze(0).squeeze(0) # (H, W)

                diff_pos = F.relu(closing - mask)
                diff_neg = F.relu(mask - closing)
                loss = (diff_pos + diff_neg).mean()

                total_loss += loss
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_seg.device, requires_grad=True)

class TrainableMultiKernelConvexitySoftMorphologicalBalancedClosingLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list[int] = None, kernel_sizes: list[int] = [3, 7, 15, 21, 41, 51, 61, 71], eps: float = 1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.kernel_sizes = kernel_sizes
        self.eps = eps
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses

        # Learnable weights for each kernel size
        self.kernel_weights = nn.Parameter(torch.ones(len(kernel_sizes)), requires_grad=True)

    def soft_closing(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return eroded

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
        pred_seg = pred['seg']  # (B, C, H, W)
        B, C, H, W = pred_seg.shape
        pred_soft = torch.softmax(pred_seg / self.temperature, dim=1)

        total_loss = 0.0
        count = 0

        # Normalize kernel weights
        weights = torch.softmax(self.kernel_weights, dim=0)

        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

                per_kernel_losses = []
                for i, k in enumerate(self.kernel_sizes):
                    closing = self.soft_closing(mask, kernel_size=k)
                    diff_pos = F.relu(closing - mask)
                    diff_neg = F.relu(mask - closing)
                    loss = (diff_pos + diff_neg).mean()
                    per_kernel_losses.append(loss)

                per_kernel_losses = torch.stack(per_kernel_losses)  # (K,)
                weighted_loss = (weights * per_kernel_losses).sum()
                total_loss += weighted_loss
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_seg.device, requires_grad=True)

class AreaAwareTrainableMultiKernelConvexitySoftGradientBasedMorphologicalBalancedClosingLoss(nn.Module):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list[int] = None, kernel_sizes: list[int] = [3, 7, 15, 21, 41, 51, 61, 71], eps: float = 1e-6, *args, **kwargs):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.eps = eps
        self.temperature = temperature

        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses

        self.kernel_weights = nn.Parameter(torch.ones(len(kernel_sizes)), requires_grad=True)
        self._beta = nn.Parameter(torch.log(torch.tensor(10.0)))

    @property
    def beta(self):
        return F.softplus(self._beta)

    def soft_dilate(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        weight = torch.ones((1, 1, kernel_size, kernel_size), device=x.device)
        conv = F.conv2d(x, weight, padding=padding)
        flat = conv.view(x.shape[0], -1)
        softmax_weights = F.softmax(self.beta * flat, dim=1).view_as(conv)
        return softmax_weights * conv  # approximate soft dilation

    def soft_erode(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return -self.soft_dilate(-x, kernel_size)

    def soft_closing(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        dilated = self.soft_dilate(mask, kernel_size)
        eroded = self.soft_erode(dilated, kernel_size)
        return eroded

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
        pred_seg = pred['seg']  # (B, C, H, W)
        B, C, H, W = pred_seg.shape
        device = pred_seg.device

        with autocast(enabled=pred_seg.is_cuda):
            pred_soft = torch.softmax(pred_seg / self.temperature, dim=1)
            total_loss = 0.0
            total_weight = 0.0

            kernel_weights = torch.softmax(self.kernel_weights, dim=0)

            for b in range(B):
                for c in self.clses:
                    mask = pred_soft[b, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    area = mask.sum().detach() + self.eps
                    object_weight = 1.0 / area

                    kernel_loss = 0.0
                    for i, k in enumerate(self.kernel_sizes):
                        closing = self.soft_closing(mask, kernel_size=k)
                        diff = F.relu(closing - mask) + F.relu(mask - closing)
                        loss = diff.mean()
                        kernel_loss += kernel_weights[i] * loss

                    total_loss += object_weight * kernel_loss
                    total_weight += object_weight

            return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=device, requires_grad=True)

# class AreaAwareTrainableMultiKernelConvexitySoftGradientBasedMorphologicalBalancedClosingLoss(BaseLoss):
#     def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
#                  clses: list[int] = None, kernel_sizes: list[int] = [3, 7, 15, 21, 41, 51, 61, 71], eps: float = 1e-6, *args, **kwargs):
#         super().__init__(weights=weights, temperature=temperature, device=device)
#         self.kernel_sizes = kernel_sizes
#         self.eps = eps
#         if clses is None or len(clses) == 0:
#             raise ValueError("Class list 'clses' must be provided and cannot be empty.")
#         self.clses = clses

#         # Learnable weights for each kernel size
#         self.kernel_weights = nn.Parameter(torch.ones(len(kernel_sizes)), requires_grad=True)

#         self._beta = nn.Parameter(torch.log(torch.tensor(10.0)))
    
#     @property
#     def beta(self):
#         # Ensure beta is always positive
#         return F.softplus(self._beta)

#     def soft_dilate(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
#         # x: (1, 1, H, W)
#         unfold = F.unfold(x, kernel_size=kernel_size, padding=kernel_size // 2)  # (1, k*k, H*W)
#         softmax_weights = F.softmax(self.beta * unfold, dim=1)
#         dilated = (unfold * softmax_weights).sum(dim=1).view_as(x)  # (1, H, W)
#         return dilated

#     def soft_erode(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
#         return -self.soft_dilate(-x, kernel_size)

#     def soft_closing(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
#         dilated = self.soft_dilate(mask, kernel_size=kernel_size)
#         eroded = self.soft_erode(dilated, kernel_size=kernel_size)
#         return eroded

#     def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
#         pred_seg = pred['seg']  # (B, C, H, W)
#         B, C, H, W = pred_seg.shape
#         pred_soft = torch.softmax(pred_seg / self.temperature, dim=1)

#         total_loss = 0.0
#         total_weight = 0.0  # New

#         # Normalize kernel weights
#         weights = torch.softmax(self.kernel_weights, dim=0)

#         for b in range(B):
#             for c in self.clses:
#                 mask = pred_soft[b, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
#                 area = mask.sum() + self.eps  # To avoid division by zero
#                 object_weight = 1.0 / area

#                 per_kernel_losses = []
#                 for i, k in enumerate(self.kernel_sizes):
#                     closing = self.soft_closing(mask, kernel_size=k)
#                     diff_pos = F.relu(closing - mask)
#                     diff_neg = F.relu(mask - closing)
#                     loss = (diff_pos + diff_neg).mean()
#                     per_kernel_losses.append(loss)

#                 per_kernel_losses = torch.stack(per_kernel_losses)  # (K,)
#                 weighted_loss = object_weight * (weights * per_kernel_losses).sum()
#                 total_loss += weighted_loss
#                 total_weight += object_weight  # Accumulate the weight

#         return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=pred_seg.device, requires_grad=True)

class AreaAwareTrainableMultiKernelConvexitySoftMorphologicalBalancedClosingLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list[int] = None, kernel_sizes: list[int] = [3, 7, 15, 21, 41, 51, 61, 71], eps: float = 1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.kernel_sizes = kernel_sizes
        self.eps = eps
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses

        # Learnable weights for each kernel size
        self.kernel_weights = nn.Parameter(torch.ones(len(kernel_sizes)), requires_grad=True)

    def soft_closing(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return eroded

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
        pred_seg = pred['seg']  # (B, C, H, W)
        B, C, H, W = pred_seg.shape
        pred_soft = torch.softmax(pred_seg / self.temperature, dim=1)

        total_loss = 0.0
        total_weight = 0.0  # New

        # Normalize kernel weights
        weights = torch.softmax(self.kernel_weights, dim=0)

        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                area = mask.sum() + self.eps  # To avoid division by zero
                object_weight = 1.0 / area

                per_kernel_losses = []
                for i, k in enumerate(self.kernel_sizes):
                    closing = self.soft_closing(mask, kernel_size=k)
                    diff_pos = F.relu(closing - mask)
                    diff_neg = F.relu(mask - closing)
                    loss = (diff_pos + diff_neg).mean()
                    per_kernel_losses.append(loss)

                per_kernel_losses = torch.stack(per_kernel_losses)  # (K,)
                weighted_loss = object_weight * (weights * per_kernel_losses).sum()
                total_loss += weighted_loss
                total_weight += object_weight  # Accumulate the weight

        return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=pred_seg.device, requires_grad=True)


class PixelwiseMultiScaleAttentionAreaAwareTrainableMultiKernelConvexitySoftMorphologicalBalancedClosingLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list[int] = None, kernel_sizes: list[int] = [3, 7, 15, 21, 41, 51, 61, 71], eps: float = 1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.kernel_sizes = kernel_sizes
        self.temperature = temperature
        self.eps = eps

        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses
        self.num_kernels = len(kernel_sizes)

        # Simple attention head: maps mask → attention weights over K scales
        self.attn_head = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_kernels, kernel_size=3, padding=1)
        )

        self._beta = nn.Parameter(torch.log(torch.tensor(10.0)))

    @property
    def beta(self):
        return F.softplus(self._beta)

    def soft_dilate(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        weight = torch.ones((1, 1, kernel_size, kernel_size), device=x.device)
        conv = F.conv2d(x, weight, padding=padding)
        flat = conv.view(x.shape[0], -1)
        softmax_weights = F.softmax(self.beta * flat, dim=1).view_as(conv)
        return softmax_weights * conv  # approximate soft dilation

    def soft_erode(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return -self.soft_dilate(-x, kernel_size)

    def soft_closing(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        dilated = self.soft_dilate(mask, kernel_size)
        eroded = self.soft_erode(dilated, kernel_size)
        return eroded

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
        pred_seg = pred['seg']  # (B, C, H, W)
        B, C, H, W = pred_seg.shape
        device = pred_seg.device

        with autocast(enabled=pred_seg.is_cuda):
            pred_soft = torch.softmax(pred_seg / self.temperature, dim=1)
            total_loss = 0.0
            total_weight = 0.0

            for b in range(B):
                for c in self.clses:
                    mask = pred_soft[b, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    area = mask.sum().detach() + self.eps
                    object_weight = 1.0 / area

                    # Get per-pixel attention over kernels: (1, K, H, W)
                    attn_logits = self.attn_head(mask)
                    attn_weights = torch.softmax(attn_logits, dim=1)

                    # Apply soft closing with each kernel
                    closings = [self.soft_closing(mask, k) for k in self.kernel_sizes]
                    closing_stack = torch.cat(closings, dim=0).unsqueeze(0)  # (1, K, H, W)

                    # Combine using per-pixel attention
                    closing = (attn_weights * closing_stack).sum(dim=1, keepdim=False)  # (1, H, W)

                    # Compute difference
                    mask = mask.squeeze(0).squeeze(0)
                    closing = closing.squeeze(0)

                    diff_pos = F.relu(closing - mask)
                    diff_neg = F.relu(mask - closing)
                    loss = (diff_pos + diff_neg).mean()

                    total_loss += object_weight * loss
                    total_weight += object_weight

            return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=device, requires_grad=True)


class GraphConvexityLoss(nn.Module):
    def __init__(self, margin: float = 0.5, clses: list = None, *args, **kwargs):
        super().__init__()
        self.margin = margin
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.cached_edges = {}  # Cache per image shape or ID
        self.clses = clses

    def _cache_key(self, mask:torch.Tensor) -> tuple:
        # Use tensor shape or a hash key based on content if needed
        return mask.shape

    def _build_graph(self, mask: torch.Tensor) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
        """Build pixel edge list for all selected classes."""
        H, W = mask.shape
        edges = []

        dy, dx = [-1, 1, 0, 0], [0, 0, -1, 1]

        for class_idx in self.clses:
            selected_pixels = torch.nonzero(mask == class_idx, as_tuple=False)

            for pixel in selected_pixels:
                y, x = pixel[0].item(), pixel[1].item()

                for d in range(4):
                    ny, nx = y + dy[d], x + dx[d]
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] == class_idx:
                        edges.append(((y, x), (ny, nx), class_idx))

        return edges

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        convex_total = 0.0

        for b in range(B):
            mask = targets[b]
            prob = probs[b]

            key = self._cache_key(mask)
            if key not in self.cached_edges:
                self.cached_edges[key] = self._build_graph(mask)

            edges = self.cached_edges[key]
            convex_loss = 0.0
            for (y1, x1), (y2, x2), cls in edges:
                diff = torch.abs(prob[cls, y1, x1] - prob[cls, y2, x2])
                convex_loss += F.relu(self.margin - diff)

            if edges:
                convex_loss /= len(edges)
            convex_total += convex_loss

        return convex_total / B

### MAKES SENSE TO WORK MORE ON THIS ###
class DepthAwareCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=5.0, reduction='mean'):
        """
        Args:
            alpha (float): Strength of the depth-based weighting.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target, depth):
        """
        Args:
            pred (Tensor): Logits from the model. Shape: [B, C, H, W]
            target (Tensor): Ground truth labels. Shape: [B, H, W]
            depth (Tensor): Depth map. Shape: [B, 1, H, W]
        Returns:
            Tensor: Weighted cross entropy loss.
        """
        weight_map = self.compute_weight_map(depth)  # [B, 1, H, W]

        ce_loss = F.cross_entropy(pred, target, reduction='none')  # [B, H, W]
        weighted_loss = ce_loss * weight_map.squeeze(1)  # Remove channel dim

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss

    def compute_weight_map(self, depth):
        """
        Compute a weight map based on depth gradients.
        Args:
            depth (Tensor): [B, 1, H, W]
        Returns:
            Tensor: Weight map. [B, 1, H, W]
        """
        dx = F.pad(depth[:, :, :, 1:] - depth[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(depth[:, :, 1:, :] - depth[:, :, :-1, :], (0, 0, 0, 1))
        grad = torch.sqrt(dx ** 2 + dy ** 2)

        grad = grad / (grad.max() + 1e-8)  # Normalize
        weight_map = 1.0 + self.alpha * grad  # [B, 1, H, W]
        return weight_map

class ConvexitySoftMorphologicalBalancedClosingLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 clses: list[str]=None, kernel_size:int=21, eps:float=1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.kernel_size = kernel_size
        self.eps = eps
        if clses is None or len(clses) == 0:
            raise ValueError("Class list 'clses' must be provided and cannot be empty.")
        self.clses = clses

    def soft_closing(self, mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Applies soft morphological closing (dilation followed by erosion) to a mask.
        """
        # Dilation: max pooling
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        # Erosion: min pooling (inverted max pooling)
        eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        return eroded  # the "closed" version of the original mask

    def forward(self, pred: dict[str, torch.Tensor], target=None, depth=None) -> torch.Tensor:
        """
        Forward pass for the Soft Morphological Closing Loss.
        """
        pred_seg = pred['seg']
        B, C, H, W = pred_seg.shape
        pred_soft = torch.softmax(pred_seg/self.temperature, dim=1)
        
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c]  # (H, W)

                mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch dimension

                # Apply soft morphological closing to the mask
                closing = self.soft_closing(mask, kernel_size=self.kernel_size)
                
                # Penalize underfilled regions (holes)
                # diff = F.relu(closing - mask)  # penalizes when closing > mask (missing pixels inside)

                # Remove batch and channel dimensions to compare
                mask = mask.squeeze(0).squeeze(0)       # (H, W)
                closing = closing.squeeze(0).squeeze(0) # (H, W)

                diff_pos = F.relu(closing - mask)  # penalize underfilled
                diff_neg = F.relu(mask - closing)  # penalize overshrinking
                loss = (diff_pos + diff_neg).mean()

                total_loss += loss
                count += 1
        
        # Return the average loss
        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_seg.device, requires_grad=True)

class EllipticBlobnessLoss(BaseLoss):
    def __init__(self, clses: list[int], eps: float = 1e-6):
        super().__init__()
        self.clses = clses
        self.eps = eps

    def forward(self, pred_seg: torch.Tensor) -> torch.Tensor:
        pred_soft = torch.softmax(pred_seg, dim=1)
        B, C, H, W = pred_soft.shape

        device = pred_seg.device
        total_loss = 0.0
        count = 0

        # Coordinate grid
        y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

        for b in range(B):
            for c in self.clses:
                mask = pred_soft[b, c:c+1]  # (1, H, W)
                m = mask
                A = m.sum() + self.eps

                mu_x = (m * x_coords[b]).sum() / A
                mu_y = (m * y_coords[b]).sum() / A

                xc = x_coords[b] - mu_x
                yc = y_coords[b] - mu_y

                sigma_xx = ((xc ** 2) * m).sum() / A
                sigma_yy = ((yc ** 2) * m).sum() / A
                sigma_xy = ((xc * yc) * m).sum() / A

                # Construct 2x2 covariance matrix
                cov = torch.stack([
                    torch.stack([sigma_xx, sigma_xy]),
                    torch.stack([sigma_xy, sigma_yy])
                ])  # shape (2, 2)

                # Eigenvalues of covariance matrix (λ1, λ2)
                eigvals = torch.linalg.eigvalsh(cov + self.eps * torch.eye(2, device=device))
                eigvals = torch.clamp(eigvals, min=self.eps)
                λ1, λ2 = eigvals[1], eigvals[0]  # λ1 ≥ λ2

                # Penalize extreme eccentricity: e = sqrt(1 - (λ2 / λ1))
                ecc = torch.sqrt(1 - (λ2 / λ1))  # eccentricity in [0, 1]
                loss = ecc ** 2  # smooth penalty

                total_loss += loss
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=device, requires_grad=True)

class CrossEntrophyWithConvexityLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, temperature=temperature, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, temperature=temperature, device=device)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        
        total_loss = ce_loss + conv_loss
        return total_loss

class CrossEntrophyWithConvexityWithIOULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, device=device, temperature=temperature, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device, temperature=temperature)
        self.iou_loss = IoULoss(weights=weights, device=device, temperature=temperature)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        iou_loss = self.iou_loss(pred, target)
        
        # Combine the losses with their respective weights
        total_loss = ce_loss + conv_loss + iou_loss
        return total_loss

class CrossEntrophyWithIOULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 eps: float = 1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        
        self.iou_loss = IoULoss(weights=weights, device=device, temperature=temperature)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device, temperature=temperature)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        
        total_loss = ce_loss + iou_loss
        return total_loss

class N3LLossWithConvexityLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        # Initialize both losses
        self.conv_loss = ConvexityLoss(weights=weights, device=device, temperature=temperature, eps=eps, clses=clses)
        self.n3_loss = N3LLoss(weights=weights, device=device, temperature=temperature)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.n3_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        
        total_loss = ce_loss + conv_loss
        return total_loss

class N3LLossWithConvexityWithIOULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, device=device, temperature=temperature, eps=eps, clses=clses)
        self.n3_loss = N3LLoss(weights=weights, device=device, temperature=temperature)
        self.iou_loss = IoULoss(weights=weights, device=device, temperature=temperature)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]],
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.n3_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        iou_loss = self.iou_loss(pred, target)        
        
        total_loss = ce_loss + conv_loss + iou_loss
        return total_loss

class N3LLossWithIOULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, temperature: float = 1.0,
                 eps: float = 1e-6, *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.eps = eps
        
        self.iou_loss = IoULoss(weights=weights, device=device, temperature=temperature)
        self.n3_loss = N3LLoss(weights=weights, device=device, temperature=temperature)

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]],
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.n3_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        
        total_loss = ce_loss + iou_loss
        return total_loss

class FocalWithConvexitylLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, temperature: float = 1.0,
                 alpha:float=1, gamma:float=3, device: str = None, clses: list = None,
                 *args, **kwargs):
        super().__init__(weights=weights, temperature=temperature, device=device)
        self.alpha = alpha  # Class balancing (e.g., [1.0, 2.0, 5.0, 10.0])
        self.gamma = gamma  # Focusing parameter
        self.ce_loss = nn.CrossEntropyLoss(weight=weights, reduction="none")
        self.conv_loss = ConvexityLoss(weights=weights, device=device, clses=clses, temperature=temperature)
        
    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        # ce_loss = F.cross_entropy(inputs, targets, self.weights, reduction="none")
        ce_loss = self.ce_loss(pred/self.temperature, target)  
        conv_loss = self.conv_loss(pred)
        
        pt = torch.exp(-ce_loss)  # p_t = probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        total_loss = focal_loss.mean() + conv_loss
        return total_loss

class DepthGLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Sobel kernels (3x3)
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Register as buffers so they move with the model and don't update
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def sobel_gradients(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_x, grad_y

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor) -> torch.Tensor:
        pred_dpt = pred['dpt']

        # Compute Sobel gradients
        grad_pred_x, grad_pred_y = self.sobel_gradients(pred_dpt)
        grad_gt_x, grad_gt_y = self.sobel_gradients(target)

        # L1 loss on gradients
        loss_x = F.l1_loss(grad_pred_x, grad_gt_x)
        loss_y = F.l1_loss(grad_pred_y, grad_gt_y)

        return loss_x + loss_y

class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor = None) -> torch.Tensor:
        pred_dpt = pred['dpt']
        loss = F.l1_loss(pred_dpt, target, reduction='mean')
        return loss

class L2DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor = None) -> torch.Tensor:
        pred_dpt = pred['dpt']
        loss = F.mse_loss(pred_dpt, target, reduction='mean')
        return loss

class ControledDepthLoss(nn.Module):
    def __init__(self, selected_clss: list[int] = [0, 1, 2, 3]):
        super().__init__()
        self.selected_clss = selected_clss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                seg_gt: torch.Tensor, depth_gt: torch.Tensor = None) -> torch.Tensor:
        
        pred_dpt = pred['dpt']  # [B, 1, H, W]

        # Create a mask for selected semantic classes
        class_mask = torch.zeros_like(seg_gt, dtype=torch.bool)
        for cls_id in self.selected_clss:
            class_mask |= (seg_gt == cls_id)
        class_mask = class_mask.unsqueeze(1)  # [B, 1, H, W]

        # Apply mask to predicted and ground truth depth
        slct_dpt_pred = pred_dpt[class_mask]
        slct_dpt_gt = depth_gt[class_mask]

        if slct_dpt_pred.numel() == 0:
            return torch.tensor(0.0, device=pred_dpt.device)  # No matching pixels

        loss = F.l1_loss(slct_dpt_pred, slct_dpt_gt, reduction='mean')
        return loss

class CrossEntrophyWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, 
                 device: str = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        dpt_loss = self.dpt_loss(pred, depth)
        total_loss = ce_loss + dpt_loss
        return total_loss

class WeightedCrossEntrophyWithDepthLoss(BaseLoss):  
    def __init__(self, weights: torch.Tensor = None, 
                 device: str = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_dpt = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dpt = self.dpt_loss(pred, depth)

        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce) * ce + self.log_sigma_ce +
                torch.exp(-self.log_sigma_dpt) * dpt + self.log_sigma_dpt)
        return loss
    

class WeightedCrossEntrophyWithDepthGLoss(BaseLoss):  
    def __init__(self, weights: torch.Tensor = None, 
                 device: str = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthGLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_dpt = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dpt = self.dpt_loss(pred, depth)

        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce) * ce + self.log_sigma_ce +
                torch.exp(-self.log_sigma_dpt) * dpt + self.log_sigma_dpt)
        return loss

class WeightedCrossEntrophyWithSSMIWithDepthLoss(BaseLoss):  
    def __init__(self, weights: torch.Tensor = None, 
                 device: str = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.ssmi_loss = StructuralConsistencyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_ssmi = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        ssmi_loss = self.ssmi_loss(pred)
        dpt = self.dpt_loss(pred, depth)

        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce) * ce + self.log_sigma_ce +
                torch.exp(-self.log_sigma_ssmi) * ssmi_loss + self.log_sigma_ssmi +
                torch.exp(-self.log_sigma_dpt) * dpt + self.log_sigma_dpt
            )
        return loss

class WeightedCrossEntrophyWithSSMILoss(BaseLoss):  
    def __init__(self, weights: torch.Tensor = None, 
                 device: str = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.ssmi_loss = StructuralConsistencyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_ssmi = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        ssmi_loss = self.ssmi_loss(pred)

        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce) * ce + self.log_sigma_ce +
                torch.exp(-self.log_sigma_ssmi) * ssmi_loss + self.log_sigma_ssmi
            )
        return loss

class WeightedLavarezWithDepthLoss(BaseLoss):  # Make sure it inherits from nn.Module
    def __init__(self, weights: torch.Tensor = None, 
                 device: str = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.dpt_loss = DepthLoss()
        self.lav_loss = WeightedLovaszSoftmaxLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_lvz = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        lvz = self.lav_loss(pred, target)
        dpt = self.dpt_loss(pred, depth)

        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_lvz) * lvz + self.log_sigma_lvz +
                torch.exp(-self.log_sigma_dpt) * dpt + self.log_sigma_dpt
            ) 
        return loss

class LavarezWithDepthLoss(BaseLoss):  # Make sure it inherits from nn.Module
    def __init__(self, weights: torch.Tensor = None, 
                 device: str = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.dpt_loss = DepthLoss()
        self.lav_loss = LovaszSoftmaxLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_lvz = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        lvz = self.lav_loss(pred, target)
        dpt = self.dpt_loss(pred, depth)

        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_lvz) * lvz + self.log_sigma_lvz +
                torch.exp(-self.log_sigma_dpt) * dpt + self.log_sigma_dpt
            ) 
        return loss

class WeightedCrossEntrophyWithConvexityWithIOULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.iou_loss = IoULoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_iou  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        iou_loss = self.iou_loss(pred, target)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce) * ce_loss + self.log_sigma_ce +
                torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv +
                torch.exp(-self.log_sigma_iou) * iou_loss + self.log_sigma_iou
        )
        return loss

class WeightedCrossEntrophyWithConvexityWithIOUWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.iou_loss = IoULoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_iou  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        iou_loss = self.iou_loss(pred, target)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce +
                torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv +
                torch.exp(-self.log_sigma_iou)  * iou_loss  + self.log_sigma_iou +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class WeightedConvexityWithIOULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.eps = eps
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.iou_loss = IoULoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_iou  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        conv_loss = self.conv_loss(pred)
        iou_loss = self.iou_loss(pred, target)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv +
                torch.exp(-self.log_sigma_iou)  * iou_loss  + self.log_sigma_iou
        )
        return loss

class WeightedConvexityWithIOUWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.iou_loss = IoULoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_iou  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        conv_loss = self.conv_loss(pred)
        iou_loss = self.iou_loss(pred, target)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv +
                torch.exp(-self.log_sigma_iou)  * iou_loss  + self.log_sigma_iou
        )
        return loss

class WeightedCrossEntrophyWithIOUWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 *args, **kwargs):
        super().__init__(weights=weights, device=device)

        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.iou_loss = IoULoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_iou  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce +
                torch.exp(-self.log_sigma_iou)  * iou_loss  + self.log_sigma_iou +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

##############

class WeightedCrossEntrophyWithConvexitySoftMorphologicalClosingLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexitySoftMorphologicalClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce +
                torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv
        )
        return loss

class WeightedCrossEntrophyWithConvexitySoftMorphologicalClosingWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexitySoftMorphologicalClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce +
                torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class WeightedCrossEntrophyWithConvexityLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce) * ce_loss + self.log_sigma_ce +
                torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv
        )
        return loss

class DepthAwareSegmentationLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, alpha=10.0, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        self.alpha = alpha
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_dpt = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

        self.register_buffer('sobel_x', 
                             torch.tensor([[1, 0, -1],
                                           [2, 0, -2],
                                           [1, 0, -1]], 
                                           dtype=torch.float32).view(1, 1, 3, 3)
                        )
        self.register_buffer('sobel_y', 
                             torch.tensor([[1, 2, 1],
                                           [0, 0, 0],
                                           [-1, -2, -1]], 
                                           dtype=torch.float32).view(1, 1, 3, 3)
                        )

    def _apply_sobel(self, img):
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad_mag
    
    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, C, H, W) raw model output before softmax
            target: (B, H, W) ground truth class indices
            depth: (B, 1, H, W) depth map aligned with input image
        Returns:
            total_loss: scalar tensor
        """

        # 1. Weighted segmentation cross-entropy loss
        seg_loss = F.cross_entropy(pred, target, weight=self.weights)

        # 2. Compute predicted probabilities
        probs = F.softmax(pred, dim=1)

        # 3. Get max predicted class probability per pixel (confidence)
        max_probs, _ = torch.max(probs, dim=1, keepdim=True)  # (B,1,H,W)

        # 4. Compute gradients with Sobel filters
        grad_s = self._apply_sobel(max_probs)  # segmentation edges
        grad_d = self._apply_sobel(depth)      # depth edges

        # 5. Depth-aware weighting (reduce penalty near depth edges)
        weight = torch.exp(-self.alpha * grad_d)

        # 6. Depth consistency loss: penalize segmentation edges where no depth edge
        depth_loss = (grad_s * weight).mean()

        # 7. Combine losses
        loss = (
            torch.exp(-self.log_sigma_ce) * seg_loss + self.log_sigma_ce +
            torch.exp(-self.log_sigma_dpt) * depth_loss + self.log_sigma_dpt
        )

        return loss

class WeightedCrossEntrophyWithEllipseLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.elp_loss = EllipseLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss  = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_elp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        elip_loss = self.elp_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        
        loss = (torch.exp(-self.log_sigma_ce)  * ce_loss  + self.log_sigma_ce +
        torch.exp(-self.log_sigma_elp) * elip_loss + self.log_sigma_elp)

        return loss

class WeightedCrossEntrophyWithConvexityWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce +
                torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class WeightedCrossEntrophyWithConvexityEnvelopeWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce +
                torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class WeightedCrossEntrophyWithConvexityEnvelopeLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_conv = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        conv_loss = self.conv_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce +
                torch.exp(-self.log_sigma_conv) * conv_loss + self.log_sigma_conv
        )
        return loss

class WeightedCrossEntrophyWithAllConvexityWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = ConvexitySoftMorphologicalClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class BalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = ConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class WeightedCrossEntrophyWithEnvelopeLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        nvlp_loss = self.conv_nvlp_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp
        )
        return loss

class WeightedCrossEntrophyWithCircularLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        
    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ
        )
        return loss

class MKBalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = MultiKernelConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class TrainableMKBalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = TrainableMultiKernelConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class WeightedCrossEntrophyWithClosingGapLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_clsg_loss = PixelwiseMultiScaleAttentionAreaAwareTrainableMultiKernelConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        
    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        clsg_loss = self.conv_clsg_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg
        )
        return loss

class WeightedCrossEntrophyWithGraphLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.grf_loss = GraphConvexityLoss(margin=0.5, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_grf = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        
    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        ce_loss = self.ce_loss(pred, target)
        clsg_loss = self.grf_loss(pred['seg'], target)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_grf) * clsg_loss + self.log_sigma_grf
        )
        return loss

class AreaAwareGradientTrainableMKBalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = AreaAwareTrainableMultiKernelConvexitySoftGradientBasedMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class AttentionAreaAwareGradientTrainableMKBalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = PixelwiseMultiScaleAttentionAreaAwareTrainableMultiKernelConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class MKBalancedWeightedCrossEntrophyWithAllConvexityWithControledDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = MultiKernelConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = ControledDepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, target, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class TrainableMKBalancedWeightedCrossEntrophyWithAllConvexityWithControledDepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = TrainableMultiKernelConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = ControledDepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, target, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class MKBalancedWeightedCrossEntrophyWithAllConvexityLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = MultiKernelConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg
        )
        return loss

class TrainableMKBalancedWeightedCrossEntrophyWithAllConvexityLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = TrainableMultiKernelConvexitySoftMorphologicalBalancedClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg
        )
        return loss

class WeightedCrossEntrophyWithAllConvexityWithL2DepthLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = ConvexitySoftMorphologicalClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.dpt_loss = L2DepthLoss()

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_dpt  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        dpt_loss = self.dpt_loss(pred, depth)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg +
                torch.exp(-self.log_sigma_dpt)  * dpt_loss  + self.log_sigma_dpt
        )
        return loss

class WeightedCrossEntrophyWithAllConvexityLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 eps: float = 1e-6, clses: list = None, *args, **kwargs):
        super().__init__(weights=weights, device=device)
        self.eps = eps
        if clses is None:
            raise ValueError("Class list 'clses' must be provided.")
        self.clses = clses
        
        self.conv_circ_loss = ConvexityLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_nvlp_loss = ConvexityEnvelopeLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.conv_clsg_loss = ConvexitySoftMorphologicalClosingLoss(weights=weights, device=device, eps=eps, clses=clses)
        self.ce_loss        = CrossEntrophyLoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce   = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_circ = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_nvlp = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss
        self.log_sigma_clsg = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        circ_loss = self.conv_circ_loss(pred)
        nvlp_loss = self.conv_nvlp_loss(pred)
        clsg_loss = self.conv_clsg_loss(pred)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce   +
                torch.exp(-self.log_sigma_circ) * circ_loss + self.log_sigma_circ +
                torch.exp(-self.log_sigma_nvlp) * nvlp_loss + self.log_sigma_nvlp +
                torch.exp(-self.log_sigma_clsg) * clsg_loss + self.log_sigma_clsg
        )
        return loss

##############

class WeightedCrossEntrophyWithIOULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None, 
                 *args, **kwargs):
        super().__init__(weights=weights, device=device)
        
        self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
        self.iou_loss = IoULoss(weights=weights, device=device)

        # Trainable log variance parameters for each loss component (more stable than direct weights)
        self.log_sigma_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
        self.log_sigma_iou = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

    def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
                target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        
        ce_loss = self.ce_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        
        # Weighted combination using learned uncertainties (Kendall et al. 2018)
        loss = (torch.exp(-self.log_sigma_ce)   * ce_loss   + self.log_sigma_ce +
                torch.exp(-self.log_sigma_iou)  * iou_loss  + self.log_sigma_iou 
        )
        return loss

# class PairwiseRankingLoss(BaseLoss): # Does nothing with weights
#     def __init__(self, weights: torch.Tensor = None, margin=1.0, 
#                  device: str = None, *args, **kwargs):
#         super().__init__(weights=None, device=device)
#         self.margin = margin  # Margin for ranking loss
    
#     def forward(self, pred, target):
#         """
#         pred: (batch_size, num_classes-1, H, W) - Raw logits
#         target: (batch_size, H, W) - Ordinal labels
#         """
#         batch_size, num_thresholds, H, W = pred.shape

#         # Expand target to create threshold labels
#         target_expanded = target.unsqueeze(1).expand(-1, num_thresholds, -1, -1)  # (B, K-1, H, W)
#         thresholds = torch.arange(num_thresholds, device=target.device).view(1, -1, 1, 1)
#         threshold_labels = (target_expanded > thresholds).float()  # Binary threshold indicators

#         # Compute pairwise ranking loss (Hinge loss)
#         loss = 0.0
#         for i in range(num_thresholds - 1):
#             diff = pred[:, i, :, :] - pred[:, i + 1, :, :]  # Difference between consecutive thresholds
#             hinge_loss = relu(self.margin - diff)  # Apply hinge loss
#             mask = (threshold_labels[:, i, :, :] != threshold_labels[:, i + 1, :, :]).float()  # Consider only differing pairs
#             loss += torch.mean(hinge_loss * mask)  # Apply mask to valid pairs
#             #loss += torch.mean(relu(self.margin - diff))  # Hinge loss to enforce ranking
        
#         return loss / (num_thresholds - 1)  # Normalize

# class OrdinalRegressionLoss(BaseLoss):
#     def __init__(self, weights:torch.Tensor=None, device:str=None, 
#                  *args, **kwargs):
#         super().__init__(weights=weights, device=device)

#         # If weights is provided, make sure it's a tensor and on the correct device
#         if weights is not None:
#             # Combine class weights to create threshold weights
#             threshold_weights = []
#             for i in range(len(self.weights) - 1):
#                 threshold_weights.append(self.weights[i] + self.weights[i + 1])
#             self.weights = torch.tensor(threshold_weights, dtype=torch.float32).to(device)
#             self.weights = self.weights.view(-1, 1, 1)  # Reshape for broadcasting
            
#         self.bce = nn.BCEWithLogitsLoss(pos_weight=self.weights)

#     def forward(self, pred, target):
#         """
#         pred: (batch_size, num_classes-1, H, W) - Sigmoid activations
#         target: (batch_size, H, W) - Ordinal labels
#         """
#         batch_size, num_thresholds, H, W = pred.shape

#         # Expand label to match prediction shape and create threshold labels
#         target_expanded = target.unsqueeze(1).expand(-1, num_thresholds, -1, -1) # (B, K-1, H, W)
#         thresholds = torch.arange(num_thresholds, device=target.device).view(1, -1, 1, 1)
#         threshold_labels = (target_expanded > thresholds).float().view(batch_size, num_thresholds, H, W)
#         loss = self.bce(pred, threshold_labels)
#         return loss
    
# class CombinedOrdinalLoss(BaseLoss):
#     def __init__(self, weights=None, margin=1.0, alpha=0.5, device=None,
#                  *args, **kwargs):
#         super().__init__(weights=weights, device=device)
#         self.wbce_loss = OrdinalRegressionLoss(weights, device)
#         self.ranking_loss = PairwiseRankingLoss(None, margin, device)
#         self.alpha = alpha  # Weighting factor

#     def forward(self, pred, target):
#         loss_bce = self.wbce_loss(pred, target)
#         loss_rank = self.ranking_loss(pred, target)
#         return loss_bce + self.alpha * loss_rank  # Combine both losses

# class WeightedCrossEntrophyWithSSMILoss(BaseLoss):  
#     def __init__(self, weights: torch.Tensor = None, 
#                  device: str = None, *args, **kwargs):
#         super().__init__(weights=weights, device=device)
#         self.ce_loss = CrossEntrophyLoss(weights=weights, device=device)
#         self.ssmi_loss = StructuralConsistencyLoss(weights=weights, device=device)

#         # Trainable log variance parameters for each loss component (more stable than direct weights)
#         self.log_sigma_ce  = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for CE loss
#         self.log_sigma_ssmi = nn.Parameter(torch.tensor(0.0))  # log(sigma^2) for depth loss

#     def forward(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], 
#                 target: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
#         ce = self.ce_loss(pred, target)
#         ssmi_loss = self.ssmi_loss(pred, depth=depth)

#         # Weighted combination using learned uncertainties (Kendall et al. 2018)
#         loss = (torch.exp(-self.log_sigma_ce) * ce + self.log_sigma_ce +
#                 torch.exp(-self.log_sigma_ssmi) * ssmi_loss + self.log_sigma_ssmi)
#         return loss

class Loss_Factory:
    # Create a dictionary that maps Loss names to Loss classes
    LOSS_FUNCS = {
        'ce': CrossEntrophyLoss,
        'ce-conv': CrossEntrophyWithConvexityLoss,
        'ce-conv-iou': CrossEntrophyWithConvexityWithIOULoss,
        'ce-iou': CrossEntrophyWithIOULoss,
        'iou': IoULoss,
        'dice': DiceLoss,
        'focal': FocalLoss,
        'nll': N3LLoss,
        'nll-iou': N3LLossWithIOULoss,
        'nll-conv-iou': N3LLossWithConvexityWithIOULoss,
        'nll-conv': N3LLossWithConvexityLoss,
        'nll-conv-iou': N3LLossWithConvexityWithIOULoss,
        'focal-conv': FocalWithConvexitylLoss,

        # Eval seperate impact
        'ce-dpt'  : CrossEntrophyWithDepthLoss,
        'w-ce-dpt': WeightedCrossEntrophyWithDepthLoss,
        'w-ce-gdpt': WeightedCrossEntrophyWithDepthGLoss,
        'w-ce-crc': WeightedCrossEntrophyWithCircularLoss,
        'w-ce-env': WeightedCrossEntrophyWithEnvelopeLoss,
        'w-ce-cgp': WeightedCrossEntrophyWithClosingGapLoss,
        'w-ce-grf': WeightedCrossEntrophyWithGraphLoss,

        'w-das': DepthAwareSegmentationLoss,
        
        'lvz': LovaszSoftmaxLoss,
        'lvz-dpt': WeightedLavarezWithDepthLoss,
        'w-lvz': WeightedLovaszSoftmaxLoss,
        'w-lvz-dpt': WeightedLavarezWithDepthLoss,

        'w-ce-conv-iou':WeightedCrossEntrophyWithConvexityWithIOULoss,
        'w-ce-conv-dpt': WeightedCrossEntrophyWithConvexityWithDepthLoss,
        'w-ce-envconv-dpt': WeightedCrossEntrophyWithConvexityEnvelopeWithDepthLoss,
        'w-ce-gapconv-dpt': WeightedCrossEntrophyWithConvexitySoftMorphologicalClosingWithDepthLoss,

        'w-ce-conv': WeightedCrossEntrophyWithConvexityLoss,
        'w-ce-envconv': WeightedCrossEntrophyWithConvexityEnvelopeLoss,
        'w-ce-gapconv': WeightedCrossEntrophyWithConvexitySoftMorphologicalClosingLoss,

        'w-ce-conv-iou-dpt': WeightedCrossEntrophyWithConvexityWithIOUWithDepthLoss,
        'w-conv-iou-dpt': WeightedConvexityWithIOUWithDepthLoss,
        'w-ce-iou': WeightedCrossEntrophyWithIOULoss,
        'w-ce-iou-dpt': WeightedCrossEntrophyWithIOUWithDepthLoss,
        'w-conv-iou': WeightedConvexityWithIOULoss,

        'w-ce-allconv-dpt': WeightedCrossEntrophyWithAllConvexityWithDepthLoss,
        'w-ce-ballconv-dpt': BalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss,
        
        'w-ce-ballconv-mk-dpt': MKBalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss,
        'w-ce-ballconv-tmk-dpt': TrainableMKBalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss,
        'w-ce-ballconv-agtmk-dpt': AreaAwareGradientTrainableMKBalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss,
        'w-ce-ballconv-agtmkatten-dpt': AttentionAreaAwareGradientTrainableMKBalancedWeightedCrossEntrophyWithAllConvexityWithDepthLoss,

        'w-ce-ballconv-mk-cont-dpt': MKBalancedWeightedCrossEntrophyWithAllConvexityWithControledDepthLoss,
        'w-ce-ballconv-tmk-cont-dpt': TrainableMKBalancedWeightedCrossEntrophyWithAllConvexityWithControledDepthLoss,
        
        'w-ce-ballconv-mk': MKBalancedWeightedCrossEntrophyWithAllConvexityLoss,
        'w-ce-ballconv-tmk': TrainableMKBalancedWeightedCrossEntrophyWithAllConvexityLoss,
        
        'w-ce-allconv': WeightedCrossEntrophyWithAllConvexityLoss,

        'w-ce-allconv-l2dpt': WeightedCrossEntrophyWithAllConvexityWithL2DepthLoss,

        'w-ce-ssmi-dpt': WeightedCrossEntrophyWithSSMIWithDepthLoss,
        'w-ce-ssmi': WeightedCrossEntrophyWithSSMILoss,

        'w-ce-elp': WeightedCrossEntrophyWithEllipseLoss,

        # Soon to be deprecated
        # 'pair-rank': PairwiseRankingLoss,
        # 'comb-ord': CombinedOrdinalLoss,
        # 'ord-reg': OrdinalRegressionLoss,
    }

    @classmethod
    def create_loss(cls, loss_type: str, **kwargs) -> BaseLoss:
        if loss_type not in cls.LOSS_FUNCS:
            raise ValueError(f"Loss function '{loss_type}' is not recognized.")
        return cls.LOSS_FUNCS[loss_type](**kwargs)



# def structural_consistency_loss(texture_depth:torch.Tensor, 
#                                 rgb_img:torch.Tensor)-> torch.Tensor:
#     texture_depth_3c = texture_depth.expand_as(rgb_img)
#     return 1 - pytorch_msssim.ssim(texture_depth_3c, rgb_img, data_range=1.0)

# class ConvexityLoss(BaseLoss):
#     def __init__(self, weights: torch.Tensor = None, device: str = None, 
#                  eps: float = 1e-6, clses:list=None, *args, **kwargs):
#         super().__init__(weights=weights, device=device)
#         self.eps = eps
#         self.clses = clses

#     def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
#         B, C, H, W = pred.shape
        
#         # Softmax along the class dimension (C)
#         pred_soft = torch.softmax(pred, dim=1)  # (B, C, H, W)

#         # Accumulate loss values
#         total_loss = 0.0
#         count = 0
        
#         # Vectorized processing of batches
#         for b in range(B):
#             for c in self.clses:
#                 prob_map = pred_soft[b, c]  # (H, W)

#                 # Compute area (sum of probabilities)
#                 #area = prob_map.sum()
#                 area = torch.clamp(prob_map.sum(), min=1e-12)


#                 # Efficient perimeter computation (using the sum of gradients)
#                 dy = prob_map[:, 1:] - prob_map[:, :-1]
#                 dx = prob_map[1:, :] - prob_map[:-1, :]
#                 #perimeter = torch.sum(torch.sqrt(dx[:, :-1] ** 2 + dy[:-1, :] ** 2))
#                 perimeter = torch.sum(torch.sqrt(torch.clamp(dx[:, :-1] ** 2 + dy[:-1, :] ** 2, min=1e-12)))

#                 # Convexity loss (penalizing irregular shape)
#                 convexity = (perimeter ** 2) / (area + self.eps)
#                 convexity = convexity / (H * W)  # <-- normalize here
#                 # Add convexity loss for the current class
#                 total_loss += convexity
#                 count += 1

#         # If no selected classes are processed, return 0 loss
#         if count == 0:
#             return torch.tensor(0.0, dtype=torch.float32, device=pred.device, requires_grad=True)

#         # Return average loss over the selected classes and batches
#         return total_loss / count


# class LovaszSoftmaxLoss(BaseLoss):
#     def __init__(self, weights: torch.Tensor = None, device: str = None, 
#                  temperature: float = 1.0, *args, **kwargs):
#         super().__init__(weights=weights, temperature=temperature, device=device)

#     def lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
#         p = len(gt_sorted)
#         gts = gt_sorted.sum(dim=0, keepdims=True)
#         intersection = gts - gt_sorted.cumsum(0)
#         union = gts + (1 - gt_sorted).cumsum(0)
#         jaccard = 1.0 - intersection / union
#         if p > 1:
#             jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#         return jaccard

#     def lovasz_softmax_flat(self, probabilities: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
#         batch_size, C, H, W = probabilities.shape
#         losses = []
        
#         # Flatten probabilities and ground truth (excluding batch dimension)
#         probabilities = probabilities.view(batch_size, C, -1)  # shape: [batch_size, C, H*W]
#         ground_truth = ground_truth.view(batch_size, -1)  # shape: [batch_size, H*W]
        
#         for c in range(C):
#             # Generate the foreground mask for class c
#             fg = (ground_truth == c).float()
            
#             if fg.sum() > 0:  # Only calculate loss if there are true pixels for this class
#                 errors = torch.abs(fg - probabilities[:, c, :])  # Absolute error for class c
                
#                 # Sort errors and foreground mask
#                 errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
#                 fg_sorted = fg[:, perm]
                
#                 # Apply Lovasz gradient
#                 loss = torch.sum(errors_sorted * self.lovasz_grad(fg_sorted))
#                 losses.append(loss)

#         return torch.mean(torch.stack(losses))

#     def forward(self, pred: dict[str, torch.Tensor], target:torch.Tensor=None, depth=None) -> torch.Tensor:
#         pred_seg = pred['seg']
#         pred_seg = F.softmax(pred_seg, dim=1)
#         return self.lovasz_softmax_flat(pred_seg, target.long())

# class LovaszSoftmaxLoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
#         p = len(gt_sorted)
#         gts = gt_sorted.sum(dim=0, keepdims=True)
#         intersection = gts - gt_sorted.cumsum(0)
#         union = gts + (1 - gt_sorted).cumsum(0)
#         jaccard = 1.0 - intersection / union
#         if p > 1:
#             jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#         return jaccard

#     def lovasz_softmax_flat(
#         self, 
#         probabilities: torch.Tensor, 
#         ground_truth: torch.Tensor
#     ) -> torch.Tensor:
#         C = probabilities.shape[1]
#         losses = []
#         for c in range(C):
#             fg = (ground_truth == c).float()
#             if fg.sum() > 0:
#                 errors = torch.abs(fg - probabilities[:, c, ...])
#                 errors_sorted, perm = torch.sort(errors.view(errors.shape[0], -1), dim=1, descending=True)
#                 fg_sorted = fg.view(fg.shape[0], -1)
#                 fg_sorted = fg_sorted[:, perm]
#                 losses.append(torch.dot(errors_sorted.float(), self.lovasz_grad(fg_sorted)))

#         return torch.mean(torch.stack(losses))

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         inputs = F.softmax(inputs, dim=1)
#         return self.lovasz_softmax_flat(inputs, targets.long())


# class LovaszSoftmaxLoss(BaseLoss):
#     def __init__(self, weights: torch.Tensor = None, device: str = None, 
#                  temperature: float = 1.0, *args, **kwargs):
#         super().__init__(weights=weights, temperature=temperature, device=device)
#         self.ce_loss = torch.nn.CrossEntropyLoss(weight=weights)

#     def lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
#         p = len(gt_sorted)
#         gts = gt_sorted.sum(dim=0, keepdims=True)
#         intersection = gts - gt_sorted.cumsum(0)
#         union = gts + (1 - gt_sorted).cumsum(0)
#         jaccard = 1.0 - intersection / union
#         if p > 1:
#             jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#         return jaccard

#     def lovasz_softmax_flat(self, pred_seg: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
#         batch_size, C, H, W = pred_seg.shape
#         losses = []

#         # Compute cross-entropy loss using unflattened tensors
#         ce_loss = self.ce_loss(pred_seg, ground_truth)

#         # Flatten pred_seg and ground truth (excluding batch dimension)
#         pred_seg = pred_seg.view(batch_size, C, -1)  # [B, C, H*W]
#         ground_truth = ground_truth.view(batch_size, -1)       # [B, H*W]

#         for c in range(C):
#             fg = (ground_truth == c).float()
#             # if fg.sum() > 0:
#             errors = torch.abs(fg - pred_seg[:, c, :])
#             errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
#             fg_sorted = fg.gather(1, perm)

#             class_loss = torch.sum(errors_sorted * self.lovasz_grad(fg_sorted))
            
#             # Apply class weight here
#             weight = self.weights[c] if self.weights is not None else 1.0
#             losses.append(weight * class_loss)

#         return ce_loss + torch.mean(torch.stack(losses)) if losses else ce_loss

#     def forward(self, pred: dict[str, torch.Tensor], target: torch.Tensor = None, depth=None) -> torch.Tensor:
#         pred_seg = pred['seg']
#         return self.lovasz_softmax_flat(pred_seg, target.long())