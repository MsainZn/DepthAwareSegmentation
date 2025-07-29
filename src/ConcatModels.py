import torch.nn as nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from transformers import AutoModelForDepthEstimation, AutoImageProcessor


from transformers import AutoImageProcessor, SegformerImageProcessor
from transformers import MobileNetV2ForSemanticSegmentation, SegformerForSemanticSegmentation, \
    MobileViTV2ForSemanticSegmentation, BeitForSemanticSegmentation, UperNetForSemanticSegmentation, \
    DPTForSemanticSegmentation

from transformers import SamModel, SamProcessor

# Grab Models 1) https://huggingface.co/docs/transformers/v4.27.0/en/tasks/semantic_segmentation
# Grab Models 2) https://huggingface.co/models?pipeline_tag=image-segmentation&library=pytorch,transformers&sort=downloads
# Construct UNET style architecture with native **PyTorch**

# POTENTIAL MODELS FOR LATER SegGptModel, CLIPSegForImageSegmentation, Data2VecVisionForSemanticSegmentation

class BaseUNet(nn.Module, ABC):
    def __init__(self,
                 in_channel_count:int,
                 num_class:int, 
                 img_size:int,
                 name:str="BaseUNet", 
                 is_ordinal:bool=False,
                 mdl_lbl:str = None):
        
        super().__init__()
        self.is_ordinal = is_ordinal
        self.in_channel_count = in_channel_count
        self.num_output = num_class - 1 if is_ordinal else num_class
        self.mdl_name = name    # The name Hugging face gives
        self.mdl_lbl = mdl_lbl  # The label based on user Config
        self.num_class = num_class
        self.img_size = img_size

    # def concat_channels(self, x:dict[str, torch.Tensor])-> torch.Tensor:
    #     return torch.cat([x[k] for k in x], dim=1)

    def calc_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad
                   and p.dim() > 1)  # Only count trainable parameters

    @abstractmethod
    def forward(self, x):
        pass  # Each subclass must implement its own forward method

    def predict_logits(self, x:dict[str, torch.Tensor], is_train:bool=False) -> torch.Tensor:
        self.train() if is_train else self.eval()
        # self.to(x.device)
        with torch.inference_mode():
            logits = self.forward(x)
        return logits['seg']
    
    def predict_with_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.is_ordinal:
            probs  = torch.sigmoid(logits)
            output = torch.sum(probs > 0.5, dim=1)
        else:
            probs  = F.softmax(logits, dim=1)
            output = torch.argmax(probs, dim=1) 
        return output

    def predict(self, x:dict[str, torch.Tensor], is_train:bool=False) -> torch.Tensor:
        logits = self.predict_logits(x, is_train)
        if self.is_ordinal:
            probs = torch.sigmoid(logits)
            return torch.sum(probs > 0.5, dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            return torch.argmax(probs, dim=1)

    def predict_one(self, x:dict[str, torch.Tensor], is_train:bool=False) -> torch.Tensor:
        return self.predict(x, is_train).squeeze(0)

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size=3, padding=1, Dropout:float=0.3):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(Dropout),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(Dropout),
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channel_count, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channel_count, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channel_count, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel_count, in_channel_count//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channel_count, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class CustomUNet(BaseUNet):
    def __init__(self, in_channel_count:int,num_class:int, img_size:int, 
                 name:str="CustomUNet", is_ordinal:bool=False, mdl_lbl:str = None):
        
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.down_conv_1 = DownSample(self.in_channel_count, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)
        self.down_conv_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv_1 = UpSample(1024, 512)
        self.up_conv_2 = UpSample(512, 256)
        self.up_conv_3 = UpSample(256, 128)
        self.up_conv_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=self.num_output, kernel_size=1)

    def forward (self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        down_1, p1 = self.down_conv_1(x_rgb)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)
        down_4, p4 = self.down_conv_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_conv_1(b, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)

        logits = self.out(up_4)
        return {
            'seg': logits, 
            'dpt': None
        }
 
class LightUNet(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, 
                 img_size:int, name:str="LightUNet", 
                 is_ordinal:bool=False, mdl_lbl:str = None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)
            
        self.down_conv_1 = DownSample(self.in_channel_count, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)

        self.bottle_neck = DoubleConv(256, 512)

        self.up_conv_1 = UpSample(512, 256)
        self.up_conv_2 = UpSample(256, 128)
        self.up_conv_3 = UpSample(128, 64)

        self.out = torch.nn.Conv2d(in_channels=64, 
                                   out_channels=self.num_output, 
                                   kernel_size=1)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        down_1, p1 = self.down_conv_1(x_rgb)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)

        b = self.bottle_neck(p3)

        up_1 = self.up_conv_1(b,    down_3)
        up_2 = self.up_conv_2(up_1, down_2)
        up_3 = self.up_conv_3(up_2, down_1)

        logits = self.out(up_3)

        return {
            'seg': logits, 
            'dpt': None
        }

# ..................Hugging Face Models Started..................

class MobileNetV2_DeepLab(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="google/deeplabv3_mobilenet_v2_1.0_513", is_ordinal:bool=False, mdl_lbl:str = None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        # Load model with the specified number of output classes
        self.model = MobileNetV2ForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        
        # Image processor (optional, if needed for pre/post-processing)
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        logits = self.model(x_rgb).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x_rgb.shape[2], x_rgb.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }

class Segformer_Face(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="jonathandinu/face-parsing", is_ordinal:bool=False, mdl_lbl:str = None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = SegformerImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        logits = self.model(x_rgb).logits  # Raw output (batch, num_class, H, W)
        # Resize logits to match input spatial dimensions
        target_size = (x_rgb.shape[2], x_rgb.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }

class Segformer_Nvidia(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", is_ordinal:bool=False, mdl_lbl:str = None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = SegformerImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        logits = self.model(x_rgb).logits  # Raw output (batch, num_class, H, W)
        
        # Resize logits to match input spatial dimensions
        target_size = (x_rgb.shape[2], x_rgb.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }

class Segformer_MITb0(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="nvidia/segformer-b0-finetuned-ade-512-512", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = SegformerImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        logits = self.model(x_rgb).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x_rgb.shape[2], x_rgb.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }
    
class MobileViTV2_Apple(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="apple/mobilevitv2-1.0-imagenet1k-256", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = MobileViTV2ForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        logits = self.model(x_rgb).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x_rgb.shape[2], x_rgb.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }
    
class DPT_INTEL(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="Intel/dpt-large-ade", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = DPTForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        logits = self.model(x_rgb).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x_rgb.shape[2], x_rgb.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }
    
class UperNet_Openmmlab(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="openmmlab/upernet-convnext-tiny", is_ordinal:bool=False, mdl_lbl:str=None):
        # in_channel_count is only 3
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True # THIS PART IS NOT DEFINED, num_channels=self.in_channel_count
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        logits = self.model(x_rgb).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x_rgb.shape[2], x_rgb.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }

class BEIT_Microsoft(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=640, 
                 name:str="microsoft/beit-base-finetuned-ade-640-640", 
                 is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)
        
        self.model = BeitForSemanticSegmentation.from_pretrained(
            self.mdl_name, image_size=self.img_size, num_labels=self.num_output, 
            ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        # x = self.concat_channels(x)
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        outputs = self.model(x_rgb)  # Raw output (batch, num_class, H, W)
        logits = outputs.logits
        # Resize logits to match input spatial dimensions
        target_size = (x_rgb.shape[2], x_rgb.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }
    
class SAM_MetaC(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SamModel.from_pretrained(self.mdl_name)
        self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
        self.encoder = self.model.vision_encoder
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Intermediate convolution
            torch.nn.BatchNorm2d(512),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),   # Further convolution
            torch.nn.BatchNorm2d(256),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )
        self.final_conv = torch.nn.Conv2d(256, self.num_output, kernel_size=1) 

    def forward(self, x):
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        features = self.encoder(x_rgb).last_hidden_state  
        dec_output = self.decoder(features)  
        outputs = self.final_conv(dec_output)  
        logits = F.interpolate(outputs, size=(1024, 1024), mode="bilinear", align_corners=False)
        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }

class SAM_MetaT(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SamModel.from_pretrained(self.mdl_name)
        self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
        self.encoder = self.model.vision_encoder
        
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(d_model=256, nhead=8),
            num_layers=6
        )
        self.final_conv = torch.nn.Conv2d(256, self.num_output, kernel_size=1)

        # Normalization layers
        self.decoder_norm = torch.nn.LayerNorm(256)  # Normalize the decoder output

    def forward(self, x):
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        # Encoder output is B, 256, 64, 64
        features = self.encoder(x_rgb).last_hidden_state  
        # Reshape for Transformer decoder: (B, C, H*W) → (H*W, B, C)
        B, C, H, W = features.shape
        features = features.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
        dec_output = self.decoder(features, features) 
        dec_output = self.decoder_norm(dec_output)  # Normalize the decoder output
        dec_output = dec_output.permute(1, 2, 0).view(B, C, H, W)

        dec_output = self.final_conv(dec_output) 
        logits = torch.functional.F.interpolate(dec_output, size=(1024, 1024), 
                                                mode="bilinear", align_corners=False)
        return {
            'seg': logits, # (batch, num_class, H, W)
            'dpt': None
        }

class SAM_MetaC_With_Depth(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SamModel.from_pretrained(self.mdl_name)
        self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
        self.encoder = self.model.vision_encoder
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Intermediate convolution
            torch.nn.BatchNorm2d(512),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),   # Further convolution
            torch.nn.BatchNorm2d(256),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )
        self.seg_final_conv = torch.nn.Conv2d(256, self.num_output, kernel_size=1) 
        self.dpt_final_conv = torch.nn.Conv2d(256, 1, kernel_size=1) 

    def forward(self, x):
        x_rgb:torch.Tensor = x['rgb']
        self.to(x_rgb.device)
        features = self.encoder(x_rgb).last_hidden_state  
        dec_output = self.decoder(features)  

        segmentation = self.seg_final_conv(dec_output)  
        logits_seg = F.interpolate(segmentation, size=(1024, 1024), 
                                     mode="bilinear", align_corners=False)
        
        depth_estimation = self.dpt_final_conv(dec_output)
        logits_dpt = F.interpolate(depth_estimation, size=(1024, 1024), 
                                     mode="bilinear", align_corners=False)
        return {
            'seg': logits_seg, # (batch, num_class, H, W)
            'dpt': logits_dpt
        }

class SAM_MetaC_With_Depth_Infused(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SamModel.from_pretrained(self.mdl_name)
        self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
        self.encoder = self.model.vision_encoder
        
        self.decoder_seg = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # Intermediate convolution
            torch.nn.BatchNorm2d(1024),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1),   # Further convolution
            torch.nn.BatchNorm2d(512),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )
        
        self.encoder_dpt = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Intermediate convolution
            torch.nn.BatchNorm2d(64),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Intermediate convolution
            torch.nn.BatchNorm2d(128),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),   # Further convolution
            torch.nn.BatchNorm2d(256),  # Added BatchNorm
            torch.nn.ReLU(),
        )

        self.seg_final_conv = torch.nn.Conv2d(512, self.num_output, kernel_size=1) 

    def forward(self, x):
        x_rgb:torch.Tensor = x['rgb']
        x_dpt:torch.Tensor = x['depth']
        self.to(x_rgb.device)
        seg_features = self.encoder(x_rgb).last_hidden_state  
        dpt_features = self.encoder_dpt(x_dpt)

        fused_features = torch.cat((seg_features, dpt_features), dim=1)  # Concatenate along the channel dimension

        dec_output = self.decoder_seg(fused_features)  
        segmentation = self.seg_final_conv(dec_output)  
        logits_seg = F.interpolate(segmentation, size=(1024, 1024), 
                                     mode="bilinear", align_corners=False)
        
        return {
            'seg': logits_seg # (batch, num_class, H, W)
        }

class TextureDiffusion(nn.Module):
    def __init__(self, in_channels:int=3, kernel_size:int=7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 
                              kernel_size=kernel_size, 
                              padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, depth:torch.Tensor, texture:torch.Tensor)-> torch.Tensor:
        texture_msg = self.conv(texture)
        fused = depth + texture_msg
        return self.relu(self.norm(fused))

class TextureDiffusionIterative(nn.Module):
    def __init__(self, in_channels=3, depth_channels=1, kernel_size=7, latent_dim=24, iterations=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.pad = kernel_size // 2

        # Encode depth to latent space
        self.depth_proj = nn.Conv2d(depth_channels, latent_dim, kernel_size=3, padding=1)

        # Predict spatial kernels from texture
        self.kernel_predictor = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim * kernel_size * kernel_size, kernel_size=3, padding=1),
            nn.Softmax(dim=1)  # normalize over kernel weights
        )

        # Final projection back to depth domain
        self.to_depth = nn.Conv2d(latent_dim, 1, kernel_size=1)

    def forward(self, depth: torch.Tensor, texture: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth.shape
        C = self.latent_dim
        K = self.kernel_size

        # Step 1: Project depth to latent space
        depth_latent = self.depth_proj(depth)  # (B, C, H, W)

        # Step 2: Generate spatial kernels from texture
        kernels = self.kernel_predictor(texture)  # (B, C*K*K, H, W)
        kernels = kernels.view(B, C, K * K, H, W)

        # Step 3: Pad input for convolutional unfolding
        unfolded = F.unfold(depth_latent, kernel_size=K, padding=self.pad)  # (B, C*K*K, H*W)
        unfolded = unfolded.view(B, C, K*K, H, W)

        # Step 4: Weighted sum (message passing)
        for _ in range(self.iterations):
            diffused = (kernels * unfolded).sum(dim=2)  # (B, C, H, W)
            unfolded = F.unfold(diffused, kernel_size=K, padding=self.pad).view(B, C, K*K, H, W)

        # Step 5: Project back to 1-channel
        fused_depth = self.to_depth(diffused)  # (B, 1, H, W)

        return fused_depth

class SAM_MetaC_With_Depth_Texture_Infused(BaseUNet):
    def __init__(self, in_channel_count: int, num_class: int, img_size: int = 1024,
                 name: str = "facebook/sam-vit-base", is_ordinal: bool = False, mdl_lbl: str = None, 
                 is_iterative:bool=True, *args, **kwargs):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)
        self.num_output = num_class
        self.mdl_name = name

        self.model = SamModel.from_pretrained(self.mdl_name)
        self.encoder = self.model.vision_encoder

        self.decoder_seg = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.depth_to_rgb = nn.Conv2d(1, 3, kernel_size=1) 

        if is_iterative:
            self.texture_diffuser = TextureDiffusionIterative(in_channels=3, kernel_size=7, 
                                                depth_channels=1, latent_dim=8, iterations=4)
        else:
            self.texture_diffuser = TextureDiffusion(in_channels=3, kernel_size=7)

        self.seg_final_conv = nn.Conv2d(256, self.num_output, kernel_size=1)
        self.dpt_final_conv = torch.nn.Conv2d(256, 1, kernel_size=1) 

    def _extract_texture_fft(self, rgb_img:torch.Tensor, alpha:float):
        B, C, H, W = rgb_img.shape
        fft = torch.fft.fft2(rgb_img)
        fft_shifted = torch.fft.fftshift(fft)

        mask = torch.ones_like(fft_shifted)
        r = int(min(H, W) * alpha)
        cy, cx = H // 2, W // 2
        mask[:, :, cy - r:cy + r, cx - r:cx + r] = 0

        filtered = fft_shifted * mask
        filtered = torch.fft.ifftshift(filtered)
        high_freq = torch.fft.ifft2(filtered).real
        return high_freq

    def forward(self, x):
        x_rgb: torch.Tensor = x['rgb']
        x_dpt: torch.Tensor = x['depth']
        self.to(x_rgb.device)

        # Step 1: Extract texture
        texture = self._extract_texture_fft(x_rgb, alpha=0.1)

        # Step 2: Diffuse texture into depth
        dpt_tex = self.texture_diffuser(x_dpt, texture)

        # Step 3: Resize and joint embed (element-wise add)
        dpt_tex_3ch = self.depth_to_rgb(dpt_tex)  # (B, 3, H, W)
        dpt_tex_3ch_hw = F.interpolate(dpt_tex_3ch, size=x_rgb.shape[-2:], 
                                   mode='bilinear', align_corners=False)
        rgbd_joint = x_rgb + dpt_tex_3ch_hw

        # Step 4: Pass through SAM encoder
        seg_features = self.encoder(rgbd_joint).last_hidden_state # THE OUTPUT IS TESTET B C H W
        # seg_features = seg_features.permute(0, 2, 1).reshape(x_rgb.size(0), 512, 64, 64)  # (B, C, H, W)

        dec_output   = self.decoder_seg(seg_features)
        segmentation = self.seg_final_conv(dec_output)

        logits_seg = F.interpolate(segmentation, size=(1024, 1024), 
                                   mode="bilinear", align_corners=False)
        
        depth_estimation = self.dpt_final_conv(dec_output)
        logits_dpt = F.interpolate(depth_estimation, size=(1024, 1024), 
                                     mode="bilinear", align_corners=False)

        return {
            'seg': logits_seg,  # (batch, num_class, H, W)
            'dpt_txt': dpt_tex_3ch_hw,
            'rgb': x_rgb,
            'dpt': logits_dpt
        }

class SAM_MetaC_With_Depth_Infused2(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SamModel.from_pretrained(self.mdl_name)
        self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
        self.encoder = self.model.vision_encoder
        
        self.decoder_seg = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # Intermediate convolution
            torch.nn.BatchNorm2d(1024),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1),   # Further convolution
            torch.nn.BatchNorm2d(512),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )
        
        self.encoder_dpt = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Intermediate convolution
            torch.nn.BatchNorm2d(64),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),   # Further convolution
            torch.nn.BatchNorm2d(256),  # Added BatchNorm
            torch.nn.ReLU(),
        )

        self.seg_final_conv = torch.nn.Conv2d(512, self.num_output, kernel_size=1) 

    def forward(self, x):
        x_rgb:torch.Tensor = x['rgb']
        x_dpt:torch.Tensor = x['depth']
        self.to(x_rgb.device)
        seg_features = self.encoder(x_rgb).last_hidden_state  
        dpt_features = self.encoder_dpt(x_dpt)

        fused_features = torch.cat((seg_features, dpt_features), dim=1)  # Concatenate along the channel dimension

        dec_output = self.decoder_seg(fused_features)  
        segmentation = self.seg_final_conv(dec_output)  
        logits_seg = F.interpolate(segmentation, size=(1024, 1024), 
                                     mode="bilinear", align_corners=False)
        return {
            'seg': logits_seg # (batch, num_class, H, W)
        }


# class TextureDiffusion(torch.nn.Module):  # Implement the Texture Diffusion module
#     def __init__(self):
#         super().__init__()
#         # Define layers for:
#         self.depth_to_latent = torch.nn.Conv2d(1, 24, kernel_size=3, padding=1) # Example
#         self.weight_prediction = torch.nn.Conv2d(256, 49, kernel_size=1)  # Output r*r (7*7) weights
#         self.relu = torch.nn.ReLU()

#     def forward(self, depth_map, texture_features):
#         # 1. Depth to Latent Space
#         latent_depth = self.depth_to_latent(depth_map)
        
#         # 2. Predict Diffusion Weights
#         weights = self.weight_prediction(texture_features)
#         weights = weights.view(depth_map.size(0), 1, 7, 7, depth_map.size(2), depth_map.size(3)) # Example: [B, C, r, r, H, W]
#         weights = F.softmax(weights, dim=2) # Normalize

#         # 3. Iterative Diffusion (Simplified - you might need more steps)
#         diffused_depth = latent_depth
#         for i in range(3):  # Example iterations
#             padded_depth = F.pad(diffused_depth, (3, 3, 3, 3), mode='reflect')  # Pad for kernel
#             diffused_depth = self.apply_diffusion_step(padded_depth, weights)
        
#         return diffused_depth

#     def apply_diffusion_step(self, padded_depth, weights):
#          b, c, h, w = padded_depth.shape
#          diffused_output = torch.zeros_like(padded_depth[:,:,3:-3,3:-3])
#          for i in range(7):
#            for j in range(7):
#              diffused_output += weights[:, :, i, j] * padded_depth[:,:, i:i+h-6, j:j+w-6]
#          return diffused_output

# class SAM_MetaC_With_Depth_FFT_Infused(BaseUNet):
#     def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
#                  name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
#         super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

#         self.model = SamModel.from_pretrained(self.mdl_name)
#         self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
#         self.encoder = self.model.vision_encoder
        
#         self.decoder_seg = torch.nn.Sequential(
#             torch.nn.Conv2d(3*256, 1024, kernel_size=3, padding=1),  # Intermediate convolution
#             torch.nn.BatchNorm2d(1024),  # Added BatchNorm
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),

#             torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1),   # Further convolution
#             torch.nn.BatchNorm2d(512),  # Added BatchNorm
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
#         )
        
#         self.encoder_dpt = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Intermediate convolution
#             torch.nn.BatchNorm2d(64),  # Added BatchNorm
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
            
#             torch.nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),   # Further convolution
#             torch.nn.BatchNorm2d(256),  # Added BatchNorm
#             torch.nn.ReLU(),
#         )
        
#         self.encoder_txt = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Intermediate convolution
#             torch.nn.BatchNorm2d(64),  # Added BatchNorm
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
            
#             torch.nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),   # Further convolution
#             torch.nn.BatchNorm2d(256),  # Added BatchNorm
#             torch.nn.ReLU(),
#         )

#         self.seg_final_conv = torch.nn.Conv2d(512, self.num_output, kernel_size=1) 

#     def forward(self, x):
#         x_rgb:torch.Tensor = x['rgb']
#         x_dpt:torch.Tensor = x['depth']
#         x_txt:torch.Tensor = x['texture']
#         self.to(x_rgb.device)
#         seg_features = self.encoder(x_rgb).last_hidden_state  
#         dpt_features = self.encoder_dpt(x_dpt)
#         txt_features = self.encoder_txt(x_txt)

#         fused_features = torch.cat((seg_features, dpt_features, txt_features), dim=1)  # Concatenate along the channel dimension

#         dec_output = self.decoder_seg(fused_features)  
#         segmentation = self.seg_final_conv(dec_output)  
#         logits_seg = F.interpolate(segmentation, size=(1024, 1024), 
#                                      mode="bilinear", align_corners=False)
        
#         return {
#             'seg': logits_seg # (batch, num_class, H, W)
#         }

# class SAM_MetaC_With_Depth_FFT_Infused_MultiHead(BaseUNet):
#     def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
#                  name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
#         super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

#         self.model = SamModel.from_pretrained(self.mdl_name)
#         self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
#         self.encoder = self.model.vision_encoder
        
#         self.decoder_seg = torch.nn.Sequential(
#             torch.nn.Conv2d(3*256, 1024, kernel_size=3, padding=1),  # Intermediate convolution
#             torch.nn.BatchNorm2d(1024),  # Added BatchNorm
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),

#             torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1),   # Further convolution
#             torch.nn.BatchNorm2d(512),  # Added BatchNorm
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
#         )
        
#         self.encoder_dpt = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Intermediate convolution
#             torch.nn.BatchNorm2d(64),  # Added BatchNorm
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
            
#             torch.nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),   # Further convolution
#             torch.nn.BatchNorm2d(256),  # Added BatchNorm
#             torch.nn.ReLU(),
#         )
        
#         self.encoder_txt = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Intermediate convolution
#             torch.nn.BatchNorm2d(64),  # Added BatchNorm
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
            
#             torch.nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),   # Further convolution
#             torch.nn.BatchNorm2d(256),  # Added BatchNorm
#             torch.nn.ReLU(),
#         )

#         self.seg_final_conv = torch.nn.Conv2d(512, self.num_output, kernel_size=1) 

#         self.dpt_final_conv = torch.nn.Conv2d(256, 1, kernel_size=1) 

#     def forward(self, x):
#         x_rgb:torch.Tensor = x['rgb']
#         x_dpt:torch.Tensor = x['depth']
#         x_txt:torch.Tensor = x['texture']
#         self.to(x_rgb.device)
#         seg_features = self.encoder(x_rgb).last_hidden_state  
#         dpt_features = self.encoder_dpt(x_dpt)
#         txt_features = self.encoder_txt(x_txt)

#         fused_features = torch.cat((seg_features, dpt_features, txt_features), dim=1)  # Concatenate along the channel dimension

#         dec_output = self.decoder_seg(fused_features)  
#         segmentation = self.seg_final_conv(dec_output)  
#         logits_seg = F.interpolate(segmentation, size=(1024, 1024), 
#                                      mode="bilinear", align_corners=False)
        
#         depth_estimation = self.dpt_final_conv(dec_output)
#         logits_dpt = F.interpolate(depth_estimation, size=(1024, 1024), 
#                                      mode="bilinear", align_corners=False)
        
        
#         return {
#             'seg': logits_seg, # (batch, num_class, H, W)
#             'dpt': logits_dpt
#         }

class _CrossModalMultiHeadAttentionFusion(nn.Module):
    def __init__(self, dim=256, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Projections
        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Normalization layers
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_dpt = nn.LayerNorm(dim)
        
        # MLP (feedforward) after attention
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(int(dim * mlp_ratio), dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        
        # Normalization layers after attention and MLP
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Learnable temperature parameter for scaling dot product attention
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        
        # Positional encoding: learnable 2D positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, dim, 1, 1))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def forward(self, rgb_feat: torch.Tensor, dpt_feat: torch.Tensor):
        B, C, H, W = rgb_feat.size()
        
        # Flatten spatial dims for LayerNorm, permute to (B, H, W, C) for LayerNorm
        rgb_feat_ln = rgb_feat.permute(0, 2, 3, 1)  # (B, H, W, C)
        dpt_feat_ln = dpt_feat.permute(0, 2, 3, 1)
        
        # Apply LayerNorm to last dimension (channels)
        rgb_feat_ln = self.norm_rgb(rgb_feat_ln).permute(0, 3, 1, 2)  # (B, C, H, W)
        dpt_feat_ln = self.norm_dpt(dpt_feat_ln).permute(0, 3, 1, 2)
        
        # Add positional encoding (broadcast to spatial dims)
        rgb_feat_ln = rgb_feat_ln + self.pos_embedding
        dpt_feat_ln = dpt_feat_ln + self.pos_embedding
        
        # Project queries, keys, values
        Q = self.query_proj(rgb_feat_ln).view(B, self.heads, self.head_dim, H * W).permute(0,1,3,2)  # (B, heads, HW, head_dim)
        K = self.key_proj(dpt_feat_ln).view(B, self.heads, self.head_dim, H * W).permute(0,1,3,2)
        V = self.value_proj(dpt_feat_ln).view(B, self.heads, self.head_dim, H * W).permute(0,1,3,2)
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / self.scale  # (B, heads, HW, HW)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Attention output
        fused = attn @ V  # (B, heads, HW, head_dim)
        fused = fused.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)  # (B, C, H, W)
        
        # Output projection
        fused = self.out_proj(fused)
        
        # Residual + Norm 1
        out = fused + rgb_feat
        out_ln = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out_ln = self.norm1(out_ln).permute(0, 3, 1, 2)
        
        # MLP + Residual + Norm 2
        mlp_out = self.mlp(out_ln)
        out2 = mlp_out + out_ln
        out2_ln = out2.permute(0, 2, 3, 1)
        out2_ln = self.norm2(out2_ln).permute(0, 3, 1, 2)
        
        return out2_ln

class _CrossModalMultiHeadAttentionFusionDPE(nn.Module):
    def __init__(self, dim=256, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Projections
        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Normalization layers
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_dpt = nn.LayerNorm(dim)
        
        # MLP (feedforward) after attention
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(int(dim * mlp_ratio), dim, kernel_size=1),
            nn.Dropout(dropout),
        )

        self.depth_embed = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # Normalization layers after attention and MLP
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Learnable temperature parameter for scaling dot product attention
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        
        # Positional encoding: learnable 2D positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, dim, 1, 1))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def forward(self, rgb_feat: torch.Tensor, dpt_feat: torch.Tensor, x_dpt: torch.Tensor):
        B, C, H, W = rgb_feat.size()
        
        # Flatten spatial dims for LayerNorm, permute to (B, H, W, C) for LayerNorm
        rgb_feat_ln = rgb_feat.permute(0, 2, 3, 1)  # (B, H, W, C)
        dpt_feat_ln = dpt_feat.permute(0, 2, 3, 1)
        
        # Apply LayerNorm to last dimension (channels)
        rgb_feat_ln = self.norm_rgb(rgb_feat_ln).permute(0, 3, 1, 2)  # (B, C, H, W)
        dpt_feat_ln = self.norm_dpt(dpt_feat_ln).permute(0, 3, 1, 2)
        
        # Create depth-informed positional encoding
        depth_pe = self.depth_embed(F.interpolate(x_dpt, size=rgb_feat.size()[2:], mode='bilinear', align_corners=False))

        # Add positional encodings
        rgb_feat_ln = rgb_feat_ln + self.pos_embedding + depth_pe
        dpt_feat_ln = dpt_feat_ln + self.pos_embedding + depth_pe
        
        # Project queries, keys, values
        Q = self.query_proj(rgb_feat_ln).view(B, self.heads, self.head_dim, H * W).permute(0,1,3,2)  # (B, heads, HW, head_dim)
        K = self.key_proj(dpt_feat_ln).view(B, self.heads, self.head_dim, H * W).permute(0,1,3,2)
        V = self.value_proj(dpt_feat_ln).view(B, self.heads, self.head_dim, H * W).permute(0,1,3,2)
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / self.scale  # (B, heads, HW, HW)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Attention output
        fused = attn @ V  # (B, heads, HW, head_dim)
        fused = fused.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)  # (B, C, H, W)
        
        # Output projection
        fused = self.out_proj(fused)
        
        # Residual + Norm 1
        out = fused + rgb_feat
        out_ln = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out_ln = self.norm1(out_ln).permute(0, 3, 1, 2)
        
        # MLP + Residual + Norm 2
        mlp_out = self.mlp(out_ln)
        out2 = mlp_out + out_ln
        out2_ln = out2.permute(0, 2, 3, 1)
        out2_ln = self.norm2(out2_ln).permute(0, 3, 1, 2)
        
        return out2_ln

class SpatialAttentionGate(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.conv(x)  # (B, 1, H, W)
        return x * attention

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.global_avg_pool(x).view(b, c)  # shape: (B, C)

        # Excitation: Fully Connected layers
        y = self.fc(y).view(b, c, 1, 1)         # shape: (B, C, 1, 1)

        # Scale: channel-wise multiplication
        return x * y

class SAM_MetaC_With_Depth_Attention_2way_dpt_inf_pe(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.num_output = num_class
        self.mdl_name = name
        self.model = SamModel.from_pretrained(name)
        self.image_processor = SamProcessor.from_pretrained(name)
        self.encoder = self.model.vision_encoder

        # Depth encoder
        self.encoder_dpt = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.depth_confidence = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Cross-modal attention to fuse RGB and depth features
        # self.attn_fusion = _CrossModalAttentionFusion(dim=256)
        self.attn_fusion_rgb2dpt = _CrossModalMultiHeadAttentionFusionDPE(dim=256, heads=8, 
                                                               mlp_ratio=4.0, dropout=0.1)
        
        self.attn_fusion_dpt2rgb = _CrossModalMultiHeadAttentionFusionDPE(dim=256, heads=8, 
                                                               mlp_ratio=4.0, dropout=0.1)

        # Decoder after fusion
        self.decoder_seg = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fusion_2ways = nn.Sequential(
            nn.Conv2d(2*256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            SpatialAttentionGate(256),
            nn.Dropout(0.3),
        )

        self.seg_final_conv = nn.Conv2d(256, self.num_output, kernel_size=1)
        self.seg_final_dpt = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x_rgb = x['rgb']
        x_dpt = x['depth']
        self.to(x_rgb.device)

        # Encode RGB using SAM
        rgb_feat = self.encoder(x_rgb).last_hidden_state

        # --- Dropout RGB features randomly during training ---
        if self.training:
            dropout_prob = 0.1
            if torch.rand(1).item() < dropout_prob:
                rgb_feat = torch.zeros_like(rgb_feat)
        
        x_dpt = (x_dpt - x_dpt.mean(dim=[2,3], keepdim=True)) / \
                 (x_dpt.std(dim=[2,3], keepdim=True) + 1e-5)

        # Encode depth
        dpt_feat = self.encoder_dpt(x_dpt)  # (B, 256, H, W)

        # # Compute depth confidence
        depth_conf = self.depth_confidence(x_dpt) 
        depth_conf_down = F.interpolate(depth_conf, size=dpt_feat.shape[2:], 
                                        mode='bilinear', align_corners=False)
        dpt_feat_conf = dpt_feat * depth_conf_down 

        # Fuse features via cross-modal attention
        fused_rgb2dpt = self.attn_fusion_rgb2dpt(rgb_feat, dpt_feat_conf, x_dpt)
        fused_dpt2rgb = self.attn_fusion_dpt2rgb(dpt_feat_conf, rgb_feat, x_dpt)

        # fused_rgb2dpt = self.attn_fusion_rgb2dpt(rgb_feat, dpt_feat_conf)
        # fused_dpt2rgb = self.attn_fusion_dpt2rgb(dpt_feat_conf, rgb_feat)
        
        # fused_rgb2dpt = self.attn_fusion_rgb2dpt(rgb_feat, dpt_feat)
        # fused_dpt2rgb = self.attn_fusion_dpt2rgb(dpt_feat, rgb_feat)

        fused_final = self.fusion_2ways(torch.cat((fused_rgb2dpt, fused_dpt2rgb), dim=1))

        # Decode
        dec_output = self.decoder_seg(fused_final)
        segmentation = self.seg_final_conv(dec_output)
        depth_layer = self.seg_final_dpt(fused_final)

        # segmentation = self.seg_final_conv(fused_rgb2dpt)
        # depth_layer = self.seg_final_dpt(fused_rgb2dpt)

        logits_seg = F.interpolate(segmentation, 
                                   size=(1024, 1024), 
                                   mode="bilinear", 
                                   align_corners=False)
        
        logits_dpt = F.interpolate(depth_layer, 
                                   size=(1024, 1024), 
                                   mode="bilinear", 
                                   align_corners=False)
        return {
            'seg': logits_seg,
            'dpt': logits_dpt  # (batch, num_class, H, W)
        }

class SAM_MetaC_With_Depth_Attention_2way(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.num_output = num_class
        self.mdl_name = name
        self.model = SamModel.from_pretrained(name)
        self.image_processor = SamProcessor.from_pretrained(name)
        self.encoder = self.model.vision_encoder

        # Depth encoder
        self.encoder_dpt = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.depth_confidence = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Cross-modal attention to fuse RGB and depth features
        # self.attn_fusion = _CrossModalAttentionFusion(dim=256)
        self.attn_fusion_rgb2dpt = _CrossModalMultiHeadAttentionFusion(dim=256, heads=8, 
                                                               mlp_ratio=4.0, dropout=0.1)
        
        self.attn_fusion_dpt2rgb = _CrossModalMultiHeadAttentionFusion(dim=256, heads=8, 
                                                               mlp_ratio=4.0, dropout=0.1)

        # Decoder after fusion
        self.decoder_seg = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fusion_2ways = nn.Sequential(
            nn.Conv2d(2*256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            SpatialAttentionGate(256),
            nn.Dropout(0.3),
        )

        self.seg_final_conv = nn.Conv2d(256, self.num_output, kernel_size=1)
        self.seg_final_dpt = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x_rgb = x['rgb']
        x_dpt = x['depth']
        self.to(x_rgb.device)

        # Encode RGB using SAM
        rgb_feat = self.encoder(x_rgb).last_hidden_state

        # --- Dropout RGB features randomly during training ---
        if self.training:
            dropout_prob = 0.1
            if torch.rand(1).item() < dropout_prob:
                rgb_feat = torch.zeros_like(rgb_feat)
                
        # Encode depth
        dpt_feat = self.encoder_dpt(x_dpt)  # (B, 256, H, W)

        # Compute depth confidence
        depth_conf = self.depth_confidence(x_dpt) 
        depth_conf_down = F.interpolate(depth_conf, size=dpt_feat.shape[2:], 
                                        mode='bilinear', align_corners=False)
        dpt_feat_conf = dpt_feat * depth_conf_down 

        # Fuse features via cross-modal attention
        fused_rgb2dpt = self.attn_fusion_rgb2dpt(rgb_feat, dpt_feat_conf)
        fused_dpt2rgb = self.attn_fusion_dpt2rgb(dpt_feat_conf, rgb_feat)
        
        # fused_rgb2dpt = self.attn_fusion_rgb2dpt(rgb_feat, dpt_feat)
        # fused_dpt2rgb = self.attn_fusion_dpt2rgb(dpt_feat, rgb_feat)

        fused_final = self.fusion_2ways(torch.cat((fused_rgb2dpt, fused_dpt2rgb), dim=1))

        # Decode
        dec_output = self.decoder_seg(fused_final)
        segmentation = self.seg_final_conv(dec_output)
        depth_layer = self.seg_final_dpt(fused_final)

        # segmentation = self.seg_final_conv(fused_rgb2dpt)
        # depth_layer = self.seg_final_dpt(fused_rgb2dpt)

        logits_seg = F.interpolate(segmentation, 
                                   size=(1024, 1024), 
                                   mode="bilinear", 
                                   align_corners=False)
        
        logits_dpt = F.interpolate(depth_layer, 
                                   size=(1024, 1024), 
                                   mode="bilinear", 
                                   align_corners=False)
        return {
            'seg': logits_seg,
            'dpt': logits_dpt  # (batch, num_class, H, W)
        }

class SAM_MetaC_With_Depth_Attention_2way_no_conf(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.num_output = num_class
        self.mdl_name = name
        self.model = SamModel.from_pretrained(name)
        self.image_processor = SamProcessor.from_pretrained(name)
        self.encoder = self.model.vision_encoder

        # Depth encoder
        self.encoder_dpt = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.depth_confidence = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Cross-modal attention to fuse RGB and depth features
        # self.attn_fusion = _CrossModalAttentionFusion(dim=256)
        self.attn_fusion_rgb2dpt = _CrossModalMultiHeadAttentionFusion(dim=256, heads=8, 
                                                               mlp_ratio=4.0, dropout=0.1)
        
        self.attn_fusion_dpt2rgb = _CrossModalMultiHeadAttentionFusion(dim=256, heads=8, 
                                                               mlp_ratio=4.0, dropout=0.1)

        # Decoder after fusion
        self.decoder_seg = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fusion_2ways = nn.Sequential(
            nn.Conv2d(2*256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            SpatialAttentionGate(256),
            nn.Dropout(0.3),
        )

        self.seg_final_conv = nn.Conv2d(256, self.num_output, kernel_size=1)
        self.seg_final_dpt = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x_rgb = x['rgb']
        x_dpt = x['depth']
        self.to(x_rgb.device)

        # Encode RGB using SAM
        rgb_feat = self.encoder(x_rgb).last_hidden_state

        # --- Dropout RGB features randomly during training ---
        if self.training:
            dropout_prob = 0.1
            if torch.rand(1).item() < dropout_prob:
                rgb_feat = torch.zeros_like(rgb_feat)
        
        x_dpt = (x_dpt - x_dpt.mean(dim=[2,3], keepdim=True)) / \
                 (x_dpt.std(dim=[2,3], keepdim=True) + 1e-5)

        # Encode depth
        dpt_feat = self.encoder_dpt(x_dpt)  # (B, 256, H, W)

        # Compute depth confidence
        # depth_conf = self.depth_confidence(x_dpt) 
        # depth_conf_down = F.interpolate(depth_conf, size=dpt_feat.shape[2:], 
        #                                 mode='bilinear', align_corners=False)
        # dpt_feat_conf = dpt_feat * depth_conf_down 

        # Fuse features via cross-modal attention
        fused_rgb2dpt = self.attn_fusion_rgb2dpt(rgb_feat, dpt_feat)
        fused_dpt2rgb = self.attn_fusion_dpt2rgb(dpt_feat, rgb_feat)

        fused_final = self.fusion_2ways(torch.cat((fused_rgb2dpt, fused_dpt2rgb), dim=1))

        # Decode
        dec_output = self.decoder_seg(fused_final)
        segmentation = self.seg_final_conv(dec_output)
        depth_layer = self.seg_final_dpt(fused_final)

        # segmentation = self.seg_final_conv(fused_rgb2dpt)
        # depth_layer = self.seg_final_dpt(fused_rgb2dpt)

        logits_seg = F.interpolate(segmentation, 
                                   size=(1024, 1024), 
                                   mode="bilinear", 
                                   align_corners=False)
        
        logits_dpt = F.interpolate(depth_layer, 
                                   size=(1024, 1024), 
                                   mode="bilinear", 
                                   align_corners=False)
        return {
            'seg': logits_seg,
            'dpt': logits_dpt  # (batch, num_class, H, W)
        }

class SAM_MetaC_With_Depth_Attention(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.num_output = num_class
        self.mdl_name = name
        self.model = SamModel.from_pretrained(name)
        self.image_processor = SamProcessor.from_pretrained(name)
        self.encoder = self.model.vision_encoder

        # Depth encoder
        self.encoder_dpt = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Cross-modal attention to fuse RGB and depth features
        # self.attn_fusion = _CrossModalAttentionFusion(dim=256)
        self.attn_fusion = _CrossModalMultiHeadAttentionFusion(dim=256, heads=8, 
                                                               mlp_ratio=4.0, dropout=0.1)

        # Decoder after fusion
        self.decoder_seg = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # self.fusion_conv = nn.Sequential(
        #     nn.Conv2d(2*256, 256, kernel_size=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        self.temp_seg = nn.Parameter(torch.ones(1))  # Initialized to 1.0
        self.temp_dpt = nn.Parameter(torch.ones(1))  # Initialized to 1.0

        self.seg_final_conv = nn.Conv2d(256, self.num_output, kernel_size=1)
        self.seg_final_dpt = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x_rgb = x['rgb']
        x_dpt = x['depth']
        self.to(x_rgb.device)

        # Encode RGB using SAM
        rgb_feat = self.encoder(x_rgb).last_hidden_state
        # Encode depth
        dpt_feat = self.encoder_dpt(x_dpt)  # (B, 256, H, W)
        # Fuse features via cross-modal attention
        fused = self.attn_fusion(rgb_feat, dpt_feat)

        # Decode
        dec_output = self.decoder_seg(fused)
        segmentation = self.seg_final_conv(dec_output)
        depth_layer = self.seg_final_dpt(dec_output)
        
        # Just Attention Fusion
        # segmentation = self.seg_final_conv(fused)
        # depth_layer = self.seg_final_dpt(fused)

        logits_seg = segmentation / (self.temp_seg + 1e-5)  # Avoid division by zero
        logits_dpt = depth_layer / (self.temp_dpt + 1e-5)  # Avoid division by zero

        logits_seg = F.interpolate(segmentation, 
                                   size=(1024, 1024), 
                                   mode="bilinear", 
                                   align_corners=False)
        
        logits_dpt = F.interpolate(depth_layer, 
                                   size=(1024, 1024), 
                                   mode="bilinear", 
                                   align_corners=False)
        return {
            'seg': logits_seg,
            'dpt': logits_dpt  # (batch, num_class, H, W)
        }

class Model_Factory:
    # Create a dictionary that maps model names to model classes
    MODEL_FACTORY: dict[str: BaseUNet] = {
        'LightUNet': LightUNet,
        'CustomUNet': CustomUNet,
        'BEIT_Microsoft': BEIT_Microsoft,
        'Segformer_Face': Segformer_Face,
        'DPT_INTEL': DPT_INTEL,
        'Segformer_MITb0': Segformer_MITb0,
        'Segformer_Nvidia': Segformer_Nvidia,
        'MobileNetV2_DeepLab': MobileNetV2_DeepLab,
        'MobileViTV2_Apple': MobileViTV2_Apple,
        'UperNet_Openmmlab': UperNet_Openmmlab,
        'SAM_MetaC': SAM_MetaC,
        'SAM_MetaT': SAM_MetaT,
        'SAM_MetaC_With_Depth': SAM_MetaC_With_Depth,
        'SAM_MetaC_With_Depth_Infused': SAM_MetaC_With_Depth_Infused,
        'SAM_MetaC_With_Depth_Texture_Infused': SAM_MetaC_With_Depth_Texture_Infused,
        'SAM_MetaC_With_Depth_Attention': SAM_MetaC_With_Depth_Attention,
        'SAM_MetaC_With_Depth_Infused2': SAM_MetaC_With_Depth_Infused2,
        'SAM_MetaC_With_Depth_Attention_2way': SAM_MetaC_With_Depth_Attention_2way,
        'SAM_MetaC_With_Depth_Attention_2way_dpt_inf_pe': SAM_MetaC_With_Depth_Attention_2way_dpt_inf_pe,
        'SAM_MetaC_With_Depth_Attention_2way_no_conf': SAM_MetaC_With_Depth_Attention_2way_no_conf,
        # 'SAM_MetaC_With_Depth_FFT_Infused': SAM_MetaC_With_Depth_FFT_Infused,
        # 'SAM_MetaC_With_Depth_FFT_Infused_MultiHead': SAM_MetaC_With_Depth_FFT_Infused_MultiHead
        # 'LightUNetResidualDepth': LightUNetResidualDepth,
        # 'LightUNetAttentionDepth': LightUNetAttentionDepth
    }

    @classmethod
    def create_model(cls, input_mdl:str, **kwargs)->BaseUNet:
        """
        Create a model based on the provided configuration.
        """
        # This method should be implemented to create and return a model instance
        if input_mdl not in cls.MODEL_FACTORY:
            raise ValueError(f"Model {input_mdl} not found in the factory.")
        return cls.MODEL_FACTORY[input_mdl](**kwargs)

    @classmethod
    def get_model_list(cls)->list[str]:
        """
        Get the list of available models in the factory.
        """
        return list(cls.MODEL_FACTORY.keys())




# class _CrossModalAttentionFusion(nn.Module):
#     def __init__(self, dim:int=256):
#         super().__init__()
#         self.query_proj = nn.Conv2d(dim, dim, 1)
#         self.key_proj = nn.Conv2d(dim, dim, 1)
#         self.value_proj = nn.Conv2d(dim, dim, 1)
#         self.softmax = nn.Softmax(dim=-1)
#         self.norm_rgb = nn.BatchNorm2d(dim)
#         self.norm_dpt = nn.BatchNorm2d(dim)

#     def forward(self, rgb_feat: torch.Tensor, dpt_feat: torch.Tensor):
#         B, C, H, W = rgb_feat.size()

#         rgb_feat = self.norm_rgb(rgb_feat)
#         dpt_feat = self.norm_dpt(dpt_feat)

#         Q = self.query_proj(rgb_feat).flatten(2).permute(0, 2, 1)  # (B, H*W, C)
#         K = self.key_proj(dpt_feat).flatten(2).permute(0, 2, 1)    # (B, H*W, C)
#         V = self.value_proj(dpt_feat).flatten(2).permute(0, 2, 1)  # (B, H*W, C)
#         d_k = Q.size(-1)

#         scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)       # (B, H*W, H*W)
#         attn = self.softmax(scores)                                # (B, H*W, H*W)
#         fused = (attn @ V)                                         # (B, H*W, C)

#         fused = fused.permute(0, 2, 1).contiguous().view(B, C, H, W)
#         return fused
