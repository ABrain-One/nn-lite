# continuous_processor.py
import sys
import os
import gc
import time
import subprocess
import json
from ab.nn.util.Const import out_dir
from typing import List, Dict, Any, Tuple
import torch
import ai_edge_torch
from ab.nn.api import data
import importlib
import logging
import traceback

out_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(out_dir / 'continuous_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NHWCWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x.permute(0, 3, 1, 2).contiguous())

class ContinuousProcessor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.max_batch_size = 32
        # Maximum allowed model parameter size (MB) before skipping conversion
        # Can be overridden with environment variable MAX_PARAM_MB
        try:
            self.max_param_mb = int(os.environ.get('MAX_PARAM_MB', '500'))
        except Exception:
            self.max_param_mb = 500
        # Restart after processing N models (to free memory)
        # Can be overridden with environment variable RESTART_EVERY
        try:
            self.restart_every = int(os.environ.get('RESTART_EVERY', '50'))
        except Exception:
            self.restart_every = 50
        # Counter for models processed since last restart
        self.models_since_restart = 0
        self.state_file = out_dir / 'processing_state.json'
        self.tflite_dir = out_dir / 'generated_tflite_files'
        self.reports_dir = out_dir / 'benchmark_reports'
        self.tflite_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Android configuration
        self.package_name = "com.example.App"
        self.device_model_dir = "/data/local/tmp"
        self.device_report_dir = f"/storage/emulated/0/Android/data/{self.package_name}/cache"
        self.android_project_path = "App"
        
        # Track progress
        self.processed_models = []
        self.failed_models = []
        self.current_model = None
        self.current_avd_name = None
        # Selected adb serial (None -> no explicit -s passed)
        self.adb_serial = None

    def run_adb(self, args, **kwargs):
        """Run adb command and automatically add -s <serial> if a device is selected.

        Args are the remaining adb arguments (e.g. ['shell', 'getprop']).
        Additional kwargs are forwarded to subprocess.run.
        """
        base = ['adb']
        if self.adb_serial:
            base.extend(['-s', self.adb_serial])
        base.extend(args)
        try:
            return subprocess.run(base, **kwargs)
        except Exception:
            # In rare cases fallback to raw subprocess.run to preserve existing behavior
            return subprocess.run(base, **kwargs)

    def get_connected_devices(self) -> List[Tuple[str, str]]:
        """Return list of tuples (serial, state) from `adb devices`."""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return []
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            devices = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    devices.append((parts[0], parts[1]))
            return devices
        except Exception:
            return []

    def select_target_device(self) -> bool:
        """Select connected Android device to use.

        Preference order:
        1. Real (non-emulator) device with state 'device'
        2. Any emulator device (serial starts with 'emulator-')
        3. None (no device connected)
        """
        devices = self.get_connected_devices()
        if not devices:
            logger.info("No adb devices connected")
            self.adb_serial = None
            return False

        # Prefer real devices (serials not starting with 'emulator-')
        for serial, state in devices:
            if state == 'device' and not serial.startswith('emulator-'):
                self.adb_serial = serial
                logger.info(f"✅ Selected real device: {serial}")
                return True

        # Fallback to any emulator device
        for serial, state in devices:
            if state == 'device' and serial.startswith('emulator-'):
                self.adb_serial = serial
                logger.info(f"ℹ️ Selected emulator device: {serial}")
                return True

        # No usable devices
        self.adb_serial = None
        logger.info("No usable adb devices found (no 'device' state)")
        return False
    def collect_device_analytics(self) -> Dict[str, Any]:
        """Collect device analytics including RAM and CPU information"""
        analytics = {
            "timestamp": time.time(),
            "memory_info": {},
            "cpu_info": {}
        }
        
        try:
            # Get memory information
            mem_result = self.run_adb(['shell', 'cat', '/proc/meminfo'], capture_output=True, text=True, timeout=10)
            
            if mem_result.returncode == 0:
                mem_lines = mem_result.stdout.split('\n')
                mem_data = {}
                for line in mem_lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        mem_data[key.strip()] = value.strip()
                
                analytics["memory_info"] = {
                    "total_ram_kb": mem_data.get('MemTotal', 'Unknown'),
                    "free_ram_kb": mem_data.get('MemFree', 'Unknown'),
                    "available_ram_kb": mem_data.get('MemAvailable', 'Unknown'),
                    "cached_kb": mem_data.get('Cached', 'Unknown')
                }

            # Get CPU information - enhanced for both Intel and ARM
            cpu_result = self.run_adb(['shell', 'cat', '/proc/cpuinfo'], capture_output=True, text=True, timeout=10)
            
            if cpu_result.returncode == 0:
                cpu_lines = cpu_result.stdout.split('\n')
                cpu_cores = 0
                processor_info = []
                current_cpu = {}
                
                # ARM-specific fields (will remain empty on Intel)
                arm_architecture = {
                    "processor_architecture": "",
                    "hardware": "",
                    "features": "",
                    "cpu_implementer": "",
                    "cpu_architecture": "",
                    "cpu_variant": "",
                    "cpu_part": "",
                    "cpu_revision": ""
                }
                
                for line in cpu_lines:
                    if 'processor' in line and ':' in line:
                        if current_cpu:
                            processor_info.append(current_cpu)
                        current_cpu = {}
                        cpu_cores += 1
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        key_clean = key.strip().lower()
                        current_cpu[key_clean] = value.strip()
                        
                        # Capture ARM-specific fields (case-insensitive)
                        key_lower = key.strip().lower()
                        if key_lower == "processor" and ("aarch64" in value or "arm" in value.lower()):
                            arm_architecture["processor_architecture"] = value.strip()
                        elif key_lower == "hardware":
                            arm_architecture["hardware"] = value.strip()
                        elif key_lower == "features":
                            arm_architecture["features"] = value.strip()
                        elif key_lower == "cpu implementer":
                            arm_architecture["cpu_implementer"] = value.strip()
                        elif key_lower == "cpu architecture":
                            arm_architecture["cpu_architecture"] = value.strip()
                        elif key_lower == "cpu variant":
                            arm_architecture["cpu_variant"] = value.strip()
                        elif key_lower == "cpu part":
                            arm_architecture["cpu_part"] = value.strip()
                        elif key_lower == "cpu revision":
                            arm_architecture["cpu_revision"] = value.strip()
                
                if current_cpu:
                    processor_info.append(current_cpu)
                
                # Only include ARM architecture if we found ARM-specific data
                arm_data = arm_architecture if any(arm_architecture.values()) else None
                
                analytics["cpu_info"] = {
                    "cpu_cores": cpu_cores,
                    "processors": processor_info[:4] if processor_info else [],
                    "arm_architecture": arm_data
                }

            # Best-effort detection of compute units (GPU/NPU/CPU names)
            try:
                units = self.detect_compute_units()
                analytics['compute_units'] = units
            except Exception:
                analytics['compute_units'] = {}

            logger.info("✅ Device analytics collected successfully")
            
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Timeout while collecting device analytics")
        except Exception as e:
            logger.warning(f"⚠️ Could not collect device analytics: {e}")
        
        return analytics

    def detect_compute_units(self) -> Dict[str, Any]:
        """Best-effort detection of compute units (CPU/GPU/NPU) on the Android device.

        Returns a dict with keys like 'cpu', 'gpu', 'npu' and human-readable names when found.
        This is best-effort because vendors expose different sysfs paths and properties.
        """
        units = {"cpu": None, "gpu": None, "npu": None, "other": []}
        try:
            # CPU model from getprop or /proc/cpuinfo
            cpu_prop = self.run_adb(['shell', 'getprop', 'ro.product.cpu.abi'], capture_output=True, text=True)
            if cpu_prop.returncode == 0 and cpu_prop.stdout.strip():
                units['cpu'] = cpu_prop.stdout.strip()
            else:
                cpuinfo = self.run_adb(['shell', 'cat', '/proc/cpuinfo'], capture_output=True, text=True)
                if cpuinfo.returncode == 0 and cpuinfo.stdout:
                    # take first 'model name' or 'Hardware' line
                    for line in cpuinfo.stdout.splitlines():
                        if ':' in line:
                            k, v = line.split(':', 1)
                            key = k.strip().lower()
                            if key in ('model name', 'hardware', 'processor') and v.strip():
                                units['cpu'] = v.strip()
                                break

            # GPU detection: try common sysfs paths (Adreno: kgsl, Mali: mali)
            try_paths = [
                '/sys/class/kgsl/kgsl-3d0/gpu_model',
                '/sys/class/kgsl/kgsl-3d0/gpu_id',
                '/sys/class/misc/mali0/name',
                '/sys/class/misc/mali0/device/name',
            ]
            for p in try_paths:
                res = self.run_adb(['shell', 'cat', p], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    units['gpu'] = res.stdout.strip()
                    break

            # Fallback: try dumpsys SurfaceFlinger or GLES strings
            if not units['gpu']:
                sf = self.run_adb(['shell', 'dumpsys', 'SurfaceFlinger'], capture_output=True, text=True)
                if sf.returncode == 0 and sf.stdout:
                    out = sf.stdout
                    for marker in ('Adreno', 'Mali', 'PVR', 'PowerVR', 'Apple', 'Intel'):
                        if marker.lower() in out.lower():
                            units['gpu'] = marker
                            break

            # NPU / DSP detection - check for common vendor sysfs/driver names
            npu_paths = [
                '/sys/class/nn', '/sys/class/vsi', '/sys/class/mediatek_npu',
                '/sys/class/arm-npu', '/sys/class/mlx', '/dev/hexagon'
            ]
            for p in npu_paths:
                res = self.run_adb(['shell', 'ls', p], capture_output=True, text=True)
                if res.returncode == 0:
                    units['npu'] = p
                    break

            # As a last resort, look for 'hexagon' or 'npu' keywords in cpuinfo or dmesg
            if not units['npu']:
                cpuinfo = self.run_adb(['shell', 'cat', '/proc/cpuinfo'], capture_output=True, text=True)
                if cpuinfo.returncode == 0 and ('hexagon' in cpuinfo.stdout.lower() or 'hexagon' in cpuinfo.stderr.lower()):
                    units['npu'] = 'Hexagon (DSP)'

        except Exception:
            pass

        return units

    def get_avd_name(self) -> str:
        """Get the AVD name for use in filename"""
        try:
            # If we already have the AVD name from starting the emulator, use it
            if self.current_avd_name:
                return self.current_avd_name
            
            # Try to get AVD name from running emulator
            result = self.run_adb(['emu', 'avd', 'name'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                avd_name = result.stdout.strip()
                # Sanitize the AVD name for filename
                avd_name = "".join(c for c in avd_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                avd_name = avd_name.replace(' ', '_')
                return avd_name
            
            # Fallback: try to get device model
            result = self.run_adb(['shell', 'getprop', 'ro.product.model'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                device_name = result.stdout.strip()
                device_name = "".join(c for c in device_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                device_name = device_name.replace(' ', '_')
                return device_name
            
            return "unknown_avd"
            
        except Exception as e:
            logger.warning(f"⚠️ Could not get AVD name: {e}")
            return "unknown_avd"

    def load_state(self) -> Dict[str, Any]:
        """Load processing state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return {"processed_models": [], "failed_models": [], "current_model": None}
    
    def save_state(self):
        """Save current processing state"""
        state = {
            "processed_models": self.processed_models,
            "failed_models": self.failed_models,
            "current_model": self.current_model,
            "timestamp": time.time()
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def get_all_available_models(self) -> List[str]:
        """Get models from command line arguments or database"""
        # Check if specific models are provided as command line arguments
        if len(sys.argv) > 1:
            specific_models = []
            for arg in sys.argv[1:]:
                # Handle comma-separated models: model1,model2,model3
                if ',' in arg:
                    specific_models.extend([m.strip() for m in arg.split(',')])
                else:
                    specific_models.append(arg.strip())
            
            if specific_models:
                logger.info(f"Using command line models: {', '.join(specific_models)}")
                return specific_models
        
        # Fall back to getting all models from database
        try:
            df = data()
            if df.empty:
                logger.warning("No models found in database")
                return []
            
            all_models = df['nn'].unique().tolist()
            mobile_models = []
            
            for model in all_models:
                # Filter out problematic models
                if not any(pattern.lower() in model.lower() 
                          for pattern in ["BayesianNet", "GAN", "Transformer"]):
                    mobile_models.append(model)
            
            logger.info(f"Found {len(mobile_models)} mobile-friendly models")
            return mobile_models
            
        except Exception as e:
            logger.error(f"Could not retrieve models: {e}")
            return []
    
    def is_mobile_friendly_config(self, config: Dict[str, Any]) -> bool:
        prm = config.get("prm", {})
        batch = int(prm.get("batch", 1))
        if batch > self.max_batch_size:
            return False
        model_name = config.get("nn", "")
        problematic = ["BayesianNet", "GAN", "Transformer"]
        return not any(p.lower() in model_name.lower() for p in problematic)
    
    def get_mobile_friendly_config(self, model_name: str) -> Dict[str, Any]:
        df = data(nn=model_name)
        if df.empty:
            raise ValueError(f"No entries found for model '{model_name}'")
        
        configs = [r.to_dict() for _, r in df.iterrows() 
                  if self.is_mobile_friendly_config(r.to_dict())]
        
        if not configs:
            df_small = df[df['prm'].apply(lambda x: int(x.get('batch', 1)) <= 4)]
            if df_small.empty:
                raise ValueError(f"No mobile-friendly configs for '{model_name}'")
            configs = [df_small.sort_values("duration").iloc[0].to_dict()]
        
        configs.sort(key=lambda x: x.get('duration', float('inf')))
        return configs[0]
    
    def extract_model_params(self, row: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any]]:
        prm = row["prm"]
        try:
            size = int(str(prm.get("transform", "")).split("_")[-1]) if prm.get("transform", "") else 224
        except (ValueError, IndexError):
            size = 224
        batch = min(int(prm.get("batch", 1)), self.max_batch_size)
        return size, batch, prm
    
    def instantiate_model(self, config: Dict[str, Any], num_classes: int):
        model_name = config["nn"]
        size, batch, prm = self.extract_model_params(config)
        in_shape, out_shape = (batch, 3, size, size), (batch, num_classes)
        
        try:
            module = importlib.import_module(f"ab.nn.nn.{model_name}")
            Net = getattr(module, "Net")
            model = Net(in_shape, out_shape, prm, self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model, size, batch, prm
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import/instantiate model '{model_name}': {e}")
    
    def convert_model(self, model_name: str) -> bool:
        """Convert a single model to TFLite"""
        try:
            logger.info(f"Converting model: {model_name}")
            config = self.get_mobile_friendly_config(model_name)
            model, size, batch, prm = self.instantiate_model(config, 100)

            # Quick safety: estimate model size from parameter count (float32 assumed)
            try:
                num_params = sum(p.numel() for p in model.parameters())
                est_bytes = int(num_params) * 4
                est_mb = est_bytes / (1024.0 * 1024.0)
            except Exception:
                num_params = None
                est_mb = None

            if est_mb is not None:
                logger.info(f"Model params: {num_params:,} (~{est_mb:.1f} MB)")
                if est_mb > float(self.max_param_mb):
                    logger.error(f"❌ Skipping conversion: estimated model size {est_mb:.1f} MB exceeds threshold {self.max_param_mb} MB")
                    return False
            
            # Wrap and convert
            wrapped_model = NHWCWrapper(model)
            wrapped_model.eval()
            sample_input = torch.randn(min(batch, 4), size, size, 3)
            
            with torch.no_grad():
                edge_model = ai_edge_torch.convert(wrapped_model, (sample_input,))
            
            output_file = self.tflite_dir / f"{model_name}.tflite"
            edge_model.export(str(output_file))
            
            del model, wrapped_model, edge_model
            self.cleanup_memory()
            
            logger.info(f"✅ Converted: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {model_name}: {e}")
            return False

    def get_available_avds(self) -> List[str]:
        """Get list of available Android Virtual Devices"""
        try:
            result = subprocess.run(['emulator', '-list-avds'], capture_output=True, text=True)
            if result.returncode == 0:
                avds = [avd.strip() for avd in result.stdout.split('\n') if avd.strip()]
                logger.info(f"Found {len(avds)} available AVDs: {avds}")
                return avds
            else:
                logger.error("Failed to list AVDs")
                return []
        except Exception as e:
            logger.error(f"Error listing AVDs: {e}")
            return []

    def is_emulator_running(self) -> bool:
        """Check if any emulator is already running"""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # Look for emulator devices (lines containing 'emulator-')
                emulators = [line for line in lines if 'emulator-' in line and 'device' in line]
                if emulators:
                    logger.info(f"Found running emulator(s): {emulators}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking emulator status: {e}")
            return False

    def ensure_emulator_running(self) -> bool:
        """Ensure an emulator is running - use any available AVD"""
        try:
            # First check if any emulator is already running
            if self.is_emulator_running():
                logger.info("✅ Emulator is already running")
                # Try to get the AVD name of the running emulator
                self.current_avd_name = self.get_avd_name()
                return True
            
            logger.info("🚀 No emulator running, starting one...")
            
            # Get available AVDs
            available_avds = self.get_available_avds()
            if not available_avds:
                logger.error("❌ No Android Virtual Devices (AVDs) found.")
                logger.info("💡 Please create an AVD in Android Studio first")
                return False
            
            # Use the first available AVD
            target_avd = available_avds[0]
            self.current_avd_name = target_avd
            logger.info(f"📱 Starting AVD: '{target_avd}'")
            
            # Start emulator in background
            process = subprocess.Popen(
                ['emulator', '-avd', target_avd, '-no-audio',  '-no-window' ],    # if add -no-window  as emulator argument it will keep running
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for device to connect (3 minute timeout)
            logger.info("⏳ Waiting for device to connect...")
            wait_time = 0
            while wait_time < 180:  # 3 minute timeout
                if self.is_emulator_running():
                    break
                time.sleep(5)
                wait_time += 5
                logger.info(f"   ... waited {wait_time}s")
            
            if not self.is_emulator_running():
                logger.error("❌ Emulator failed to start within timeout")
                process.terminate()
                return False
            
            # Wait for OS to boot completely (2 minute timeout)
            logger.info("⏳ Waiting for OS to boot completely...")
            boot_time = 0
            while boot_time < 120:  # 2 minute timeout
                try:
                    result = self.run_adb(['shell', 'getprop', 'sys.boot_completed'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip() == "1":
                        logger.info("✅ Emulator is fully booted and ready")
                        return True
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    pass
                
                time.sleep(5)
                boot_time += 5
            
            logger.error("❌ Emulator boot timeout")
            return False
            
        except Exception as e:
            logger.error(f"❌ Emulator startup failed: {e}")
            return False
    
    def install_android_app(self) -> bool:
        """Install Android benchmark app using gradlew"""
        try:
            logger.info("📦 Installing Android app...")
            
            # Ensure gradlew is executable
            gradlew_path = os.path.join(self.android_project_path, "gradlew")
            if os.path.exists(gradlew_path):
                os.chmod(gradlew_path, 0o755)
            
            result = subprocess.run(
                ['./gradlew', 'installDebug'],
                cwd=self.android_project_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ Android app installed successfully")
                return True
            else:
                logger.error(f"❌ App installation failed: {result.stderr}")
                # Attempt fallback: use adb install directly
                try:
                    apk_path = os.path.join(self.android_project_path, 'app', 'build', 'outputs', 'apk', 'debug', 'app-debug.apk')
                    if os.path.exists(apk_path):
                        logger.info(f"🔁 Attempting fallback install via adb of '{apk_path}'")
                        # Ensure we have a target device selected
                        if not self.adb_serial:
                            self.select_target_device()

                        # Start logcat capture to help diagnose device-side restriction
                        try:
                            ts = int(time.time())
                            log_file = out_dir / f"install_logcat_{ts}.log"
                            logger.info(f"📥 Capturing device logcat to: {log_file}")
                            adb_base = ['adb']
                            if self.adb_serial:
                                adb_base.extend(['-s', self.adb_serial])
                            logcat_proc = subprocess.Popen(adb_base + ['logcat', '-v', 'time'], stdout=open(log_file, 'w'), stderr=subprocess.DEVNULL)
                        except Exception as e:
                            logcat_proc = None
                            logger.warning(f"⚠️ Could not start logcat capture: {e}")

                        # Try adb install -r
                        adb_install = self.run_adb(['install', '-r', apk_path], capture_output=True, text=True)
                        if adb_install.returncode == 0:
                            logger.info("✅ APK installed via adb fallback")
                            if logcat_proc:
                                logcat_proc.terminate()
                            return True
                        else:
                            logger.warning(f"⚠️ adb install returned: {adb_install.stderr}")
                            # Try push + pm install as a last resort
                            try:
                                remote_path = f"/data/local/tmp/{os.path.basename(apk_path)}"
                                push_res = self.run_adb(['push', apk_path, remote_path], capture_output=True, text=True)
                                if push_res.returncode == 0:
                                    pm_res = self.run_adb(['shell', 'pm', 'install', '-r', remote_path], capture_output=True, text=True)
                                    if pm_res.returncode == 0:
                                        logger.info("✅ APK installed via push+pm fallback")
                                        if logcat_proc:
                                            logcat_proc.terminate()
                                        return True
                                    else:
                                        logger.warning(f"⚠️ pm install failed: {pm_res.stderr}")
                                else:
                                    logger.warning(f"⚠️ adb push failed: {push_res.stderr}")
                            except Exception as e:
                                logger.warning(f"⚠️ Fallback push+pm install failed: {e}")

                        # Stop logcat capture and point user to the file
                        if logcat_proc:
                            try:
                                logcat_proc.terminate()
                                time.sleep(0.5)
                            except Exception:
                                pass
                            logger.info(f"🔍 Saved device logcat to: {log_file}")
                    else:
                        logger.warning(f"⚠️ APK not found for fallback install: {apk_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Error during fallback install: {e}")

                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ App installation timeout")
            return False
        except Exception as e:
            logger.error(f"❌ App installation error: {e}")
            return False
    
    def force_stop_emulator(self):
        """Force stop the emulator"""
        try:
            logger.info("🛑 Force stopping emulator...")
            
            # Kill emulator process
            self.run_adb(['emu', 'kill'], capture_output=True, timeout=30)
            
            # Wait a bit for process to terminate
            time.sleep(10)
            
            # Force kill any remaining emulator processes
            subprocess.run(['pkill', '-f', 'emulator'], capture_output=True)
            
            logger.info("✅ Emulator stopped")
            
        except Exception as e:
            logger.error(f"⚠️ Error stopping emulator: {e}")
    
    def handle_benchmark_failure(self, model_name: str):
        """Handle benchmark failure by waiting, closing everything, and restarting"""
        logger.error(f"🔄 Benchmark failed for {model_name}, initiating recovery...")
        
        # Step 1: Wait for 3 minutes
        logger.info("⏳ Waiting 3 minutes before recovery...")
        for i in range(180, 0, -10):
            logger.info(f"   ... {i} seconds remaining")
            time.sleep(10)
        
        # Step 2: Close everything
        logger.info("🛑 Closing all processes...")
        
        # Force stop emulator
        self.force_stop_emulator()
        
        # Kill any ADB processes
        try:
            subprocess.run(['pkill', '-f', 'adb'], capture_output=True)
            logger.info("✅ ADB processes killed")
        except:
            pass
        
        # Additional cleanup
        self.cleanup_memory()
        
        # Step 3: Wait a bit more
        logger.info("⏳ Final wait before restart...")
        time.sleep(10)
        
        logger.info("🔄 Restarting process...")
        
        # Restart the script
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    def run_benchmark(self, model_name: str) -> bool:
        """Run benchmark on Android device and retrieve results"""
        try:
            tflite_file = self.tflite_dir / f"{model_name}.tflite"
            if not tflite_file.exists():
                logger.error(f"❌ TFLite file not found: {tflite_file}")
                return False
            
            # Get AVD name for filename
            avd_name = self.get_avd_name()
            if not avd_name:
                logger.warning("⚠️ Could not get AVD name, using 'unknown_avd'")
                avd_name = "unknown_avd"
            
            # Get task from config and create task_modelName directory
            config = self.get_mobile_friendly_config(model_name)
            task = config.get('task', 'unknown_task')
            task_model_dir = self.reports_dir / f"{task}_{model_name}"
            task_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Push model to device
            logger.info(f"📤 Pushing model to device: {model_name}")
            push_result = self.run_adb(['push', str(tflite_file), f"{self.device_model_dir}/{model_name}.tflite"], capture_output=True, text=True)
            
            if push_result.returncode != 0:
                logger.error(f"❌ Failed to push model: {push_result.stderr}")
                return False
            
            logger.info("✅ Model pushed successfully")
            
            # Stop previous instance
            self.run_adb(['shell', 'am', 'force-stop', self.package_name], capture_output=True)
            
            # Launch benchmark
            logger.info("🎯 Launching benchmark...")
            launch_result = self.run_adb(['shell', 'am', 'start', '-n', f"{self.package_name}/.MainActivity", '--es', 'model_filename', f"{model_name}.tflite"], capture_output=True, text=True)
            
            if launch_result.returncode != 0:
                logger.error(f"❌ Failed to launch benchmark: {launch_result.stderr}")
                return False
            
            logger.info("✅ Benchmark launched successfully")
            
            # Wait for completion
            logger.info("⏳ Waiting 20 seconds for benchmark completion...")
            time.sleep(20)
            
            # Collect device analytics before retrieving report
            logger.info("📊 Collecting device analytics...")
            device_analytics = self.collect_device_analytics()
            
            # Retrieve report with new structure
            device_report = f"{self.device_report_dir}/{model_name}.json"
            local_report = task_model_dir / f"android_{avd_name}.json"
            
            pull_result = self.run_adb(['pull', device_report, str(local_report)], capture_output=True, text=True)
            
            # Enhance the report with analytics if successfully pulled
            if pull_result.returncode == 0 and local_report.exists():
                try:
                    with open(local_report, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    # Add device analytics to the benchmark report
                    benchmark_data["device_analytics"] = device_analytics
                    # Add a concise 'unit' field (preference: NPU -> GPU -> CPU)
                    try:
                        cu = device_analytics.get('compute_units', {}) if isinstance(device_analytics, dict) else {}
                        unit = cu.get('npu') or cu.get('gpu') or cu.get('cpu')
                        if isinstance(unit, list) and unit:
                            unit = unit[0]
                        benchmark_data['unit'] = unit
                    except Exception:
                        benchmark_data['unit'] = None
                    
                    # Save the enhanced report
                    with open(local_report, 'w') as f:
                        json.dump(benchmark_data, f, indent=2)
                    
                    logger.info("✅ Device analytics added to benchmark report")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not enhance report with analytics: {e}")
            
            # Cleanup device
            self.run_adb(['shell', 'rm', f"{self.device_model_dir}/{model_name}.tflite", device_report], capture_output=True)
            
            if pull_result.returncode == 0 and local_report.exists():
                logger.info(f"✅ Benchmark completed and report retrieved: {model_name}")
                logger.info(f"📁 Report saved to: {local_report}")
                return True
            else:
                logger.error(f"❌ Failed to retrieve benchmark report for {model_name}")
                # Handle benchmark failure by waiting, closing, and restarting
                self.handle_benchmark_failure(model_name)
                return False
                
        except Exception as e:
            logger.error(f"❌ Benchmark execution error for {model_name}: {e}")
            # Handle benchmark failure by waiting, closing, and restarting
            self.handle_benchmark_failure(model_name)
            return False
    
    def cleanup_memory(self):
        """Clean up memory between conversions"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def process_models_continuously(self):
        """Main processing loop with resume capability"""
        # Load previous state
        state = self.load_state()
        self.processed_models = state.get("processed_models", [])
        self.failed_models = state.get("failed_models", [])
        last_model = state.get("current_model")
        
        # Get all models
        all_models = self.get_all_available_models()
        if not all_models:
            logger.error("❌ No models found to process")
            return
        
        # Filter out already processed models
        remaining_models = [
            m for m in all_models 
            if m not in self.processed_models and m not in self.failed_models
        ]
        
        # Resume from last model if needed
        if last_model and last_model in remaining_models:
            remaining_models.remove(last_model)
            remaining_models.insert(0, last_model)
            logger.info(f"🔄 Resuming from model: {last_model}")
        
        if not remaining_models:
            logger.info("✅ All models have been processed already!")
            self.print_summary()
            return
        
        logger.info(f"🚀 Starting continuous processing of {len(remaining_models)} models")
        logger.info(f"   Remaining models: {', '.join(remaining_models)}")
        
        # Ensure emulator is running (using any available AVD)
        # Prefer a connected real device; fall back to emulator if needed
        logger.info("🔎 Selecting target device (real device preferred)...")
        has_device = self.select_target_device()
        if has_device and self.adb_serial and not str(self.adb_serial).startswith('emulator-'):
            logger.info(f"✅ Using real device: {self.adb_serial}")
        else:
            logger.info("📱 No real device found or using emulator. Ensuring emulator is running...")
            if not self.ensure_emulator_running():
                logger.error("❌ Cannot proceed without emulator or connected device")
                return
            # re-select device after emulator startup
            self.select_target_device()
        
        # Install app once
        if not self.install_android_app():
            logger.error("❌ App installation failed")
            return
        
        # Process each model
        total_models = len(remaining_models)
        for idx, model_name in enumerate(remaining_models, 1):
            self.current_model = model_name
            self.save_state()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Processing {idx}/{total_models}: {model_name}")
            logger.info(f"{'='*60}")
            
            success = True
            
            # Step 1: Convert to TFLite
            logger.info("🔄 Converting model to TFLite...")
            if not self.convert_model(model_name):
                logger.error(f"❌ Conversion failed for {model_name}")
                self.failed_models.append(model_name)
                success = False
            else:
                # Step 2: Run benchmark
                logger.info("📊 Running benchmark on device...")
                if not self.run_benchmark(model_name):
                    logger.error(f"❌ Benchmark failed for {model_name}")
                    self.failed_models.append(model_name)
                    success = False
                else:
                    self.processed_models.append(model_name)
                    logger.info(f"✅ Successfully processed {model_name}")
            
            # Update state
            self.current_model = None
            self.save_state()

            # Check if we should restart the process to free memory
            if self.restart_every > 0:
                self.models_since_restart += 1
                if self.models_since_restart >= self.restart_every:
                    logger.info(f"🔄 Processed {self.models_since_restart} models; restarting process to free memory...")
                    time.sleep(2)
                    # Gracefully restart by re-executing the script with the same arguments
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            
            # Small delay between models
            if idx < total_models:
                logger.info("⏳ Waiting 3 seconds before next model...")
                time.sleep(3)
        
        # Final summary
        self.print_summary()
        
        # Cleanup state file on successful completion
        if not self.failed_models and os.path.exists(self.state_file):
            try:
                os.remove(self.state_file)
                logger.info("🧹 Cleaned up state file")
            except:
                pass
    
    def print_summary(self):
        """Print comprehensive summary"""
        total_attempted = len(self.processed_models) + len(self.failed_models)
        
        logger.info(f"\n{'='*70}")
        logger.info("🎯 PROCESSING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"✅ Successfully processed: {len(self.processed_models)} models")
        logger.info(f"❌ Failed: {len(self.failed_models)} models")
        logger.info(f"📊 Total attempted: {total_attempted}")
        
        if self.processed_models:
            logger.info(f"\n✅ Successful models ({len(self.processed_models)}):")
            for i, model in enumerate(self.processed_models[:10], 1):
                logger.info(f"   {i:2d}. {model}")
            if len(self.processed_models) > 10:
                logger.info(f"   ... and {len(self.processed_models) - 10} more")
        
        if self.failed_models:
            logger.info(f"\n❌ Failed models ({len(self.failed_models)}):")
            for i, model in enumerate(self.failed_models, 1):
                logger.info(f"   {i:2d}. {model}")
            logger.info("\n💡 You can rerun the script to retry failed models")
        
        logger.info(f"\n📁 Reports saved to: {self.reports_dir.absolute()}")
        logger.info(f"📁 Models saved to: {self.tflite_dir.absolute()}")

def main():
    """Main entry point with error handling"""
    processor = ContinuousProcessor()
    
    try:
        processor.process_models_continuously()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Processing interrupted by user")
        processor.save_state()
        logger.info("💾 Progress saved. Run the script again to resume.")
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        logger.error(f"📝 Stack trace: {traceback.format_exc()}")
        processor.save_state()
        logger.info("💾 State saved for recovery")

if __name__ == "__main__":
    main()
