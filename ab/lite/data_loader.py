import os
import torch
import glob
from PIL import Image
import numpy as np
from typing import Iterator, Tuple, List

class RepresentativeDataset:
    def __init__(self, data_dir: str, size: int = 224, batch_size: int = 1, limit: int = 100):
        """
        Args:
            data_dir: Directory containing images
            size: Input size (assumes square)
            batch_size: Batch size for calibration
            limit: Maximum number of samples to use
        """
        self.data_dir = data_dir
        self.size = size
        self.batch_size = batch_size
        self.limit = limit
        self.image_paths = self._find_images()
        
    def _find_images(self) -> List[str]:
        if not os.path.exists(self.data_dir):
            return []
            
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(self.data_dir, ext)))
            images.extend(glob.glob(os.path.join(self.data_dir, ext.upper())))
            
        return sorted(images)[:self.limit]

    def _preprocess(self, image_path: str) -> torch.Tensor:
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.size, self.size), Image.BILINEAR)
            
            # Convert to tensor and normalize
            # Assuming standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # But for TFLite, we often just want 0-1 or -1 to 1 depending on model.
            # PyTorch models usually expect normalized inputs.
            
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Normalize
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = (img_np - mean) / std
            
            # HWC to CHW if needed, but ai_edge_torch might handle it. 
            # The conversion script wraps model with NHWCWrapper which permutes input.
            # So if we provide NHWC here (which PIL/numpy is), the wrapper expects NHWC and permutes to NCHW.
            # Wait, let's check NHWCWrapper in torch2tflite-all.py.
            # class NHWCWrapper(torch.nn.Module):
            #     def forward(self, x):
            #         return self.model(x.permute(0, 3, 1, 2).contiguous())
            # So the wrapper expects NHWC (Batch, Height, Width, Channels).
            # Our numpy array is HWC. We just need to add batch dim.
            
            return torch.from_numpy(img_np)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def __call__(self) -> Iterator[Tuple[torch.Tensor]]:
        if not self.image_paths:
            # Fallback to random data if no images found (for testing/unblocked flow)
            # In production this should probably be an error or warning.
            print(f"Warning: No images found in {self.data_dir}. Using random data for calibration.")
            for _ in range(min(10, self.limit)):
                yield (torch.randn(self.batch_size, self.size, self.size, 3),)
            return

        batch = []
        for img_path in self.image_paths:
            tensor = self._preprocess(img_path)
            if tensor is not None:
                batch.append(tensor)
                
            if len(batch) == self.batch_size:
                # Stack to create batch: (B, H, W, C)
                batch_tensor = torch.stack(batch)
                yield (batch_tensor,)
                batch = []
        
        # Yield remaining
        if batch:
            batch_tensor = torch.stack(batch)
            yield (batch_tensor,)
