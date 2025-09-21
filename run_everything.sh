#!/bin/bash

# This script fully automates the TFLite benchmarking process.
# It automatically launches an emulator, generates the model, runs the benchmark,
# retrieves the results, and cleans up.

set -e

# --- CONFIGURATION ---
ANDROID_PROJECT_PATH="Kotlinapplication"
EMULATOR_NAME="Pixel_5_API_33"
PACKAGE_NAME="com.example.kotlinapplication"
TFLITE_GENERATOR_SCRIPT="generate_tflite.py"
DEVICE_MODEL_DIR="/data/local/tmp"
TFLITE_FILES_DIR="generated_tflite_files"
JSON_REPORTS_DIR="benchmark_reports"
# --- END CONFIGURATION ---

# ==============================================================================
# SECTION 0: PREREQUISITE CHECKS
# ==============================================================================
echo "🚀 Initializing TFLite Benchmark Automation..."

for cmd in python3 adb emulator; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "❌ Error: Required command '$cmd' is not found in your PATH."
        if [ "$cmd" == "emulator" ] || [ "$cmd" == "adb" ]; then
            echo "   Please ensure your Android SDK 'emulator' and 'platform-tools' directories are in your PATH."
            echo "   Example for .bashrc: export PATH=\$PATH:\$HOME/Android/Sdk/emulator:\$HOME/Android/Sdk/platform-tools"
        fi
        exit 1
    fi
done

if [ $# -eq 0 ]; then
    echo "❌ Error: No model name(s) provided."
    echo "Usage: ./run_everything.sh <model_name_1> [model_name_2] ..."
    exit 1
fi

mkdir -p "$TFLITE_FILES_DIR"
mkdir -p "$JSON_REPORTS_DIR"

# ==============================================================================
# SECTION 1: EMULATOR MANAGEMENT
# ==============================================================================
if ! adb get-state 1>/dev/null 2>&1 || [ "$(adb get-state)" != "device" ]; then
    echo "[1/5] No running device found. Starting emulator..."
    AVAILABLE_AVDS=($(emulator -list-avds))
    if [ ${#AVAILABLE_AVDS[@]} -eq 0 ]; then
        echo "   ❌ ERROR: No Android Virtual Devices (AVDs) found."
        exit 1
    fi
    TARGET_AVD=${EMULATOR_NAME:-${AVAILABLE_AVDS[0]}}
    if [[ ! " ${AVAILABLE_AVDS[@]} " =~ " ${TARGET_AVD} " ]]; then
        echo "   ⚠️ WARNING: AVD '$TARGET_AVD' not found. Defaulting to '${AVAILABLE_AVDS[0]}'."
        TARGET_AVD=${AVAILABLE_AVDS[0]}
    fi
    echo "   -> Launching AVD: '$TARGET_AVD'. This can take a few minutes..."
    emulator -avd "$TARGET_AVD" -no-snapshot-load &
    echo "   -> Waiting for the device to connect..."
    adb wait-for-device
    echo "   -> Waiting for the OS to boot completely..."
    until [[ "$(adb shell getprop sys.boot_completed 2>/dev/null)" == "1" ]]; do
        sleep 5; echo "      ...";
    done
    echo "   -> Emulator is fully booted and ready."
else
    echo "[1/5] Found an already running device. Proceeding."
fi

# ==============================================================================
# SECTION 2: BUILD AND INSTALL ANDROID APP
# ==============================================================================
echo "[2/5] Building and installing the generic benchmark APK..."
(cd "$ANDROID_PROJECT_PATH" && ./gradlew --quiet installDebug)
MAIN_ACTIVITY="$PACKAGE_NAME.MainActivity"

# ==============================================================================
# SECTION 3: MODEL PROCESSING LOOP
# ==============================================================================
for model_name in "$@"; do
    echo ""
    echo "===================================================="
    echo "▶️  Processing Model: $model_name"
    echo "===================================================="

    TFLITE_FILE_PATH="$TFLITE_FILES_DIR/${model_name}.tflite"
    LOCAL_JSON_PATH="$JSON_REPORTS_DIR/${model_name}_report.json"

    DEVICE_MODEL_PATH="${DEVICE_MODEL_DIR}/${model_name}.tflite"
    JSON_REPORT_FILENAME="${model_name}_report.json"
    
    # *** This is the new, reliable path for the report on the device ***
    DEVICE_JSON_PATH="/storage/emulated/0/Android/data/$PACKAGE_NAME/cache/$JSON_REPORT_FILENAME"

    echo "[3/5] Generating model: '$model_name'..."
    python3 "$TFLITE_GENERATOR_SCRIPT" "$model_name"
    if [ ! -f "$TFLITE_FILE_PATH" ]; then
        echo "   ❌ FAILED: Script did not create '$TFLITE_FILE_PATH'. Skipping."
        continue
    fi

    echo "[4/5] Pushing model to device and launching benchmark..."
    adb push "$TFLITE_FILE_PATH" "$DEVICE_MODEL_DIR/"
    echo "   -> Pushed to $DEVICE_MODEL_PATH"

    adb shell am force-stop "$PACKAGE_NAME"
    adb shell am start -n "$PACKAGE_NAME/$MAIN_ACTIVITY" --es model_filename "${model_name}.tflite"
    
    echo "   -> Benchmark started. Waiting 45 seconds for completion..."
    sleep 45

    echo "[5/5] Retrieving report and cleaning up..."
    adb pull "$DEVICE_JSON_PATH" "$LOCAL_JSON_PATH" || true

    echo "----------------------------------------------------"
    echo "Model Name: $model_name"
    if [ -f "$LOCAL_JSON_PATH" ]; then
        if grep -q '"model_run_successfully": true' "$LOCAL_JSON_PATH"; then
            echo "Model Run Successfully: Yes"
        else
            echo "Model Run Successfully: No (Check JSON report for error)"
        fi
        echo "JSON File created successfully to the path: $LOCAL_JSON_PATH"
    else
        echo "Model Run Successfully: No"
        echo "JSON File created successfully: No (Failed to retrieve report)."
        echo "   -> TIP: Check logs with 'adb logcat -s TFLiteRunner' to debug."
    fi
    echo "----------------------------------------------------"

    adb shell rm "$DEVICE_JSON_PATH" > /dev/null 2>&1 || true
    adb shell rm "$DEVICE_MODEL_PATH" > /dev/null 2>&1 || true
done

# ==============================================================================
# SECTION 4: FINAL CLEANUP
# ==============================================================================
echo ""
echo "✅ All benchmark tasks complete."
echo "   -> Shutting down emulator..."
adb emu kill
echo "===================================================="
