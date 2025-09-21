import sys
import os
import gc
from pathlib import Path
import torch
import ai_edge_torch
from ab.nn.api import data
import importlib
import logging
import traceback
from typing import Tuple, Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NHWCWrapper(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, x): return self.model(x.permute(0, 3, 1, 2).contiguous())

class TFLiteConverter:
    def __init__(self):
        self.device = torch.device("cpu"); self.max_batch_size = 32
        self.failed_models, self.successful_models, self.skipped_models = [], [], []

    def parse_arguments(self) -> Tuple[List[str], int, Path]:
        if len(sys.argv) < 2:
            print("Usage: python generate_tflite.py <model_name>"); sys.exit(1)
        
        model_names = [name.strip() for name in sys.argv[1].split(',')]
        num_classes = 100
        
        # *** Output directory is now a dedicated folder. ***
        output_dir = Path("generated_tflite_files")
        output_dir.mkdir(parents=True, exist_ok=True)
        return model_names, num_classes, output_dir

    def is_mobile_friendly_config(self, config: Dict[str, Any]) -> bool:
        prm = config.get("prm", {}); batch = int(prm.get("batch", 1))
        if batch > self.max_batch_size: return False
        model_name = config.get("nn", ""); problematic = ["BayesianNet", "GAN", "Transformer"]
        return not any(p.lower() in model_name.lower() for p in problematic)

    def get_mobile_friendly_config(self, model_name: str) -> Dict[str, Any]:
        df = data(nn=model_name)
        if df.empty: raise ValueError(f"No entries found for model '{model_name}'")
        configs = [r.to_dict() for _, r in df.iterrows() if self.is_mobile_friendly_config(r.to_dict())]
        if not configs:
            df_small = df[df['prm'].apply(lambda x: int(x.get('batch', 1)) <= 4)]
            if df_small.empty: raise ValueError(f"No mobile-friendly configs for '{model_name}'")
            configs = [df_small.sort_values("duration").iloc[0].to_dict()]
        configs.sort(key=lambda x: x.get('duration', float('inf')))
        return configs[0]

    def extract_model_params(self, row: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any]]:
        prm = row["prm"]
        try:
            size = int(str(prm.get("transform", "")).split("_")[-1]) if prm.get("transform", "") else 224
        except (ValueError, IndexError): size = 224
        batch = min(int(prm.get("batch", 1)), self.max_batch_size)
        return size, batch, prm

    def instantiate_model(self, config: Dict[str, Any], num_classes: int):
        model_name = config["nn"]; size, batch, prm = self.extract_model_params(config)
        in_shape, out_shape = (batch, 3, size, size), (batch, num_classes)
        try:
            module = importlib.import_module(f"ab.nn.nn.{model_name}")
            Net = getattr(module, "Net")
            model = Net(in_shape, out_shape, prm, self.device); model.eval()
            for param in model.parameters(): param.requires_grad = False
            return model, size, batch, prm
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import/instantiate model '{model_name}': {e}")

    def convert_single_model(self, model, output_file: Path, size: int, batch: int, model_name: str) -> bool:
        try:
            wrapped_model = NHWCWrapper(model); wrapped_model.eval()
            sample_input = torch.randn(min(batch, 4), size, size, 3)
            with torch.no_grad():
                edge_model = ai_edge_torch.convert(wrapped_model, (sample_input,))
            edge_model.export(str(output_file)); logger.info(f"✅ SUCCESS: {model_name} → {output_file}")
            return True
        except Exception as e:
            logger.error(f"❌ FAILED to convert '{model_name}': {e}"); return False

    def cleanup_memory(self):
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def process_all_models(self, model_names: List[str], num_classes: int, output_dir: Path):
        for model_name in model_names:
            logger.info(f"\n--- Processing model: '{model_name}' ---")
            try:
                output_file = output_dir / f"{model_name}.tflite"
                if output_file.exists():
                    self.skipped_models.append(model_name); continue
                config = self.get_mobile_friendly_config(model_name)
                model, size, batch, prm = self.instantiate_model(config, num_classes)
                if self.convert_single_model(model, output_file, size, batch, model_name):
                    self.successful_models.append(model_name)
                else: self.failed_models.append(model_name)
                del model; self.cleanup_memory()
            except Exception as e:
                logger.error(f"Failed to process '{model_name}': {e}"); self.failed_models.append(model_name)
        self.print_summary()

    def print_summary(self):
        logger.info("\n🎯 CONVERSION SUMMARY:"); logger.info(f"   ✅ Successful: {len(self.successful_models)}")
        logger.info(f"   ❌ Failed: {len(self.failed_models)}"); logger.info(f"   ⏭️  Skipped: {len(self.skipped_models)}")
        if self.failed_models: logger.info(f"\n❌ Failed models: {', '.join(self.failed_models)}")

def main():
    converter = TFLiteConverter()
    try:
        model_names, num_classes, output_dir = converter.parse_arguments()
        converter.process_all_models(model_names, num_classes, output_dir)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}\n{traceback.format_exc()}"); sys.exit(1)

if __name__ == "__main__":
    main()
