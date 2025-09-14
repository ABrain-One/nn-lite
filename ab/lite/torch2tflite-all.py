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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NHWCWrapper(torch.nn.Module):
    """
    Wraps a PyTorch model to handle NHWC to NCHW format conversion for TFLite.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # NHWC -> NCHW conversion
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.model(x)

class TFLiteConverter:
    """Main class to handle TFLite conversion operations."""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.max_batch_size = 32  # Limit batch size for mobile-friendly models
        self.failed_models = []
        self.successful_models = []
        self.skipped_models = []
    
    def parse_arguments(self) -> Tuple[List[str], int, Path]:
        """Parses command-line arguments and returns them."""
        if len(sys.argv) < 2:
            print("Usage: python create_tflite.py <model1,model2,model3>")
            print("Example: python create_tflite.py resnet50,mobilenet,efficientnet")
            print("Or single model: python create_tflite.py resnet50")
            print("Or all models: python create_tflite.py all")
            sys.exit(1)
        
        # Parse model names (can be comma-separated for multiple models)
        model_names_str = sys.argv[1]
        model_names = [name.strip() for name in model_names_str.split(',')]
        
        num_classes = 100
        output_dir = Path("../../Kotlinapplication/app/src/main/assets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return model_names, num_classes, output_dir
    
    def is_mobile_friendly_config(self, config: Dict[str, Any]) -> bool:
        """Check if a model configuration is mobile-friendly."""
        prm = config.get("prm", {})
        batch = int(prm.get("batch", 1))
        
        # Skip very large batch sizes for mobile deployment
        if batch > self.max_batch_size:
            return False
        
        # Skip known problematic model types
        model_name = config.get("nn", "")
        problematic_patterns = [
            "BayesianNet",  # Dynamic weight mutation
            "GAN",          # Complex architectures
            "Transformer",  # Large memory requirements
        ]
        
        for pattern in problematic_patterns:
            if pattern.lower() in model_name.lower():
                return False
        
        return True
    
    def get_mobile_friendly_config(self, model_name: str) -> Dict[str, Any]:
        """Gets the fastest mobile-friendly configuration for a single model."""
        df = data(nn=model_name)
        if df.empty:
            raise ValueError(f"No entries found for model '{model_name}'")
        
        # Filter for mobile-friendly configurations
        mobile_configs = []
        for _, row in df.iterrows():
            config = row.to_dict()
            if self.is_mobile_friendly_config(config):
                mobile_configs.append(config)
        
        if not mobile_configs:
            # If no mobile-friendly configs, try the smallest batch size available
            df_small_batch = df[df['prm'].apply(lambda x: int(x.get('batch', 1)) <= 4)]
            if df_small_batch.empty:
                raise ValueError(f"No mobile-friendly configurations found for model '{model_name}'")
            mobile_configs = [df_small_batch.sort_values("duration").iloc[0].to_dict()]
        
        # Sort by duration and get the fastest mobile-friendly config
        mobile_configs.sort(key=lambda x: x.get('duration', float('inf')))
        fastest_config = mobile_configs[0]
        
        logger.info(f"Selected mobile-friendly config for '{model_name}' (batch: {fastest_config['prm'].get('batch', 1)}, duration: {fastest_config.get('duration', 'N/A')})")
        return fastest_config
    
    def extract_model_params(self, row: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any]]:
        """Extracts model parameters from database row."""
        prm = row["prm"]
        
        try:
            # Extract size from transform parameter
            transform = prm.get("transform", "")
            if transform and "_" in str(transform):
                size = int(str(transform).split("_")[-1])
            else:
                size = 224  # Default mobile-friendly size
        except (ValueError, IndexError):
            size = 224
            logger.warning(f"Could not extract size from transform, using default: {size}")
        
        batch = int(prm.get("batch", 1))
        
        # Ensure mobile-friendly batch size
        if batch > self.max_batch_size:
            logger.warning(f"Batch size {batch} too large for mobile, reducing to {self.max_batch_size}")
            batch = min(batch, self.max_batch_size)
        
        return size, batch, prm
    
    def instantiate_model(self, config: Dict[str, Any], num_classes: int):
        """Dynamically imports and instantiates the model."""
        model_name = config["nn"]
        size, batch, prm = self.extract_model_params(config)
        
        in_shape = (batch, 3, size, size)
        out_shape = (batch, num_classes)
        
        try:
            # Dynamically import the model module and class
            module = importlib.import_module(f"ab.nn.nn.{model_name}")
            Net = getattr(module, "Net")
            
            # Instantiate model
            model_instance = Net(in_shape, out_shape, prm, self.device)
            model_instance.eval()
            
            # Force evaluation mode and disable gradients
            for param in model_instance.parameters():
                param.requires_grad = False
            
            return model_instance, size, batch, prm
            
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import or instantiate model '{model_name}': {e}")
    
    def convert_single_model(self, model_instance, output_file: Path, 
                           size: int, batch: int, model_name: str) -> bool:
        """Converts a single model instance to TFLite."""
        try:
            # Ensure model is in eval mode
            model_instance.eval()
            
            # Wrap model for NHWC conversion
            wrapped_model = NHWCWrapper(model_instance)
            wrapped_model.eval()
            
            # Create mobile-friendly dummy input (smaller batch for memory efficiency)
            mobile_batch = min(batch, 4)  # Use smaller batch for conversion
            sample_input = torch.randn(mobile_batch, size, size, 3, dtype=torch.float32)
            
            # Convert to TFLite with memory-efficient settings
            logger.info(f"Converting '{model_name}' with input shape: {sample_input.shape}")
            
            # Use torch.no_grad() to reduce memory usage
            with torch.no_grad():
                edge_model = ai_edge_torch.convert(wrapped_model, (sample_input,))
            
            # Export model
            edge_model.export(str(output_file))
            logger.info(f"✅ SUCCESS: {model_name} → {output_file}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ FAILED: Could not convert '{model_name}': {error_msg}")
            
            # Log specific error types
            if "Mutating module attribute" in error_msg:
                logger.error(f"   → Model '{model_name}' has dynamic weights (not TFLite compatible)")
            elif "out of memory" in error_msg.lower():
                logger.error(f"   → Out of memory error for '{model_name}' - try smaller batch size")
            
            return False
    
    def cleanup_memory(self):
        """Aggressive memory cleanup between model conversions."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def process_all_models(self, model_names: List[str], num_classes: int, output_dir: Path):
        """Processes all specified models one by one using their mobile-friendly configuration."""
        total_models = len(model_names)
        logger.info(f"Processing {total_models} models for mobile deployment")
        
        for i, model_name in enumerate(model_names, 1):
            logger.info(f"\n--- Processing model {i}/{total_models}: '{model_name}' ---")
            
            try:
                # Check if output file already exists
                potential_files = list(output_dir.glob(f"{model_name}_*.tflite"))
                if potential_files:
                    logger.info(f"⏭️  SKIPPED: {model_name} (already exists: {potential_files[0].name})")
                    self.skipped_models.append(model_name)
                    continue
                
                # Get mobile-friendly configuration for this model
                config = self.get_mobile_friendly_config(model_name)
                
                # Instantiate model with mobile-friendly config
                model_instance, size, batch, prm = self.instantiate_model(config, num_classes)
                
                # Create output filename with mobile-friendly suffix
                output_file = output_dir / f"{model_name}_mobile_batch{batch}_size{size}.tflite"
                
                # Convert model
                if self.convert_single_model(model_instance, output_file, size, batch, model_name):
                    self.successful_models.append(model_name)
                else:
                    self.failed_models.append(model_name)
                
                # Clean up memory immediately after each model
                del model_instance
                self.cleanup_memory()
                
            except Exception as e:
                logger.error(f"Failed to process model '{model_name}': {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.failed_models.append(model_name)
                continue
        
        # Final summary
        self.print_summary()
        
        return len(self.successful_models) > 0
    
    def print_summary(self):
        """Print conversion summary statistics."""
        total = len(self.successful_models) + len(self.failed_models) + len(self.skipped_models)
        
        logger.info(f"\n🎯 CONVERSION SUMMARY:")
        logger.info(f"   ✅ Successful: {len(self.successful_models)}")
        logger.info(f"   ❌ Failed: {len(self.failed_models)}")
        logger.info(f"   ⏭️  Skipped: {len(self.skipped_models)}")
        logger.info(f"   📊 Total: {total}")
        
        if self.failed_models:
            logger.info(f"\n❌ Failed models: {', '.join(self.failed_models[:10])}{'...' if len(self.failed_models) > 10 else ''}")
    
    def get_all_available_models(self) -> List[str]:
        """Gets all available model names from the database."""
        try:
            # Get all data without filtering
            df = data()
            if df.empty:
                logger.warning("No models found in database")
                return []
            
            # Get unique model names and filter out problematic ones
            all_models = df['nn'].unique().tolist()
            
            # Filter for mobile-friendly models
            mobile_models = []
            for model in all_models:
                if not any(pattern.lower() in model.lower() for pattern in ["BayesianNet"]):
                    mobile_models.append(model)
            
            logger.info(f"Found {len(mobile_models)} mobile-friendly models out of {len(all_models)} total")
            return mobile_models
            
        except Exception as e:
            logger.error(f"Could not retrieve available models: {e}")
            return []

def main():
    """Main execution pipeline."""
    converter = TFLiteConverter()
    
    try:
        model_names, num_classes, output_dir = converter.parse_arguments()
        
        # If user wants all models, get mobile-friendly ones from database
        if len(model_names) == 1 and model_names[0].lower() == 'all':
            logger.info("Retrieving all mobile-friendly models from database...")
            model_names = converter.get_all_available_models()
            if not model_names:
                logger.error("No mobile-friendly models found!")
                sys.exit(1)
        
        logger.info(f"Starting mobile-friendly TFLite conversion for {len(model_names)} model(s)")
        logger.info(f"Using {num_classes} classes and saving to: {output_dir}")
        
        success = converter.process_all_models(model_names, num_classes, output_dir)
        
        if not success:
            logger.warning("No models were successfully converted, but process completed")
            
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        converter.print_summary()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()