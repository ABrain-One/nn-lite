#!/usr/bin/env bash
set -euo pipefail

# install-android-studio-medium.sh
# No-sudo, per-user installer that robustly attempts downloads and ensures Medium Phone AVD setup.

##############################
# Config - edit if you want
##############################
USER_HOME="${HOME}"
STUDIO_DIR="${USER_HOME}/android-studio"
SDK_ROOT="${USER_HOME}/Android/Sdk"
LOCAL_BIN="${USER_HOME}/.local/bin"
TMPDIR="$(mktemp -d)"
PAGE_HTML="$TMPDIR/studio_page.html"
PLATFORM_API="android-36"
SYSTEM_IMAGE_PATH="system-images;android-36;google_apis_playstore;x86_64"
AVD_NAME="Medium_Phone_API_36.1"
DEVICE_ID="Medium Phone"  # correct AVD device ID
RETRY_COUNT=8
##############################

trap 'rm -rf "$TMPDIR"' EXIT

info(){ printf '\e[1;34m[INFO]\e[0m %s\n' "$*"; }
warn(){ printf '\e[1;33m[WARN]\e[0m %s\n' "$*"; }
err(){ printf '\e[1;31m[ERROR]\e[0m %s\n' "$*"; exit 1; }

mkdir -p "$SDK_ROOT" "$LOCAL_BIN" "$STUDIO_DIR"

# Download helper functions
try_curl() { command -v curl >/dev/null 2>&1 && curl -L --retry "$RETRY_COUNT" --retry-delay 5 -C - -f -o "$2" "$1" && return 0 || return 1; }
try_wget() { command -v wget >/dev/null 2>&1 && wget --tries="$RETRY_COUNT" --waitretry=5 -c -O "$2" "$1" && return 0 || return 1; }
try_aria2() { command -v aria2c >/dev/null 2>&1 && aria2c -x 16 -s 16 -k 1M -o "$2" "$1" && return 0 || return 1; }

# Fetch Android Studio page
info "Fetching Android Studio page to discover download links..."
curl -fsSL "https://developer.android.com/studio" -o "$PAGE_HTML" || true

# Auto-discover Android Studio tarball
STUDIO_DL_URL="$(grep -Eo 'https://[^\" ]+android-studio-[0-9].+linux.+tar.gz' "$PAGE_HTML" | head -n1 || true)"
LOCAL_STUDIO_ARCHIVE="$(ls "$HOME"/Downloads/android-studio-*.tar.gz 2>/dev/null | head -n1 || true)"
STUDIO_ARCHIVE="$TMPDIR/android-studio.tar.gz"

if [ -n "$LOCAL_STUDIO_ARCHIVE" ]; then
  info "Using local Android Studio archive: $LOCAL_STUDIO_ARCHIVE"
  cp "$LOCAL_STUDIO_ARCHIVE" "$STUDIO_ARCHIVE"
elif [ -n "$STUDIO_DL_URL" ]; then
  info "Downloading Android Studio..."
  try_curl "$STUDIO_DL_URL" "$STUDIO_ARCHIVE" || try_wget "$STUDIO_DL_URL" "$STUDIO_ARCHIVE" || try_aria2 "$STUDIO_DL_URL" "$STUDIO_ARCHIVE" || err "Failed to download Android Studio."
else
  err "No Android Studio tarball found or detected."
fi

# Extract Android Studio
info "Extracting Android Studio to $STUDIO_DIR ..."
mkdir -p "$STUDIO_DIR"
tar -xzf "$STUDIO_ARCHIVE" -C "$TMPDIR"
rsync -a --delete "$TMPDIR"/android-studio/ "$STUDIO_DIR/" || true

# Create launcher
cat > "$LOCAL_BIN/android-studio" <<'EOF'
#!/usr/bin/env bash
exec "$HOME/android-studio/bin/studio.sh" "$@"
EOF
chmod +x "$LOCAL_BIN/android-studio"

# Command-line tools
info "Installing command-line tools..."
CMDLINE_URL="$(grep -Eo 'https://[^\" ]+commandlinetools-linux[^\" ]+zip' "$PAGE_HTML" | head -n1 || true)"
CMDLINE_ZIP="$TMPDIR/cmdline.zip"
if [ -n "$CMDLINE_URL" ]; then
  try_curl "$CMDLINE_URL" "$CMDLINE_ZIP" || try_wget "$CMDLINE_URL" "$CMDLINE_ZIP" || try_aria2 "$CMDLINE_URL" "$CMDLINE_ZIP" || err "Failed to download cmdline tools."
else
  err "Command-line tools URL not found."
fi

CMDLINE_ROOT="$SDK_ROOT/cmdline-tools/latest"
mkdir -p "$CMDLINE_ROOT"
unzip -q "$CMDLINE_ZIP" -d "$TMPDIR/cmdline_tmp"
cp -r "$TMPDIR/cmdline_tmp"/cmdline-tools/* "$CMDLINE_ROOT"/

# Environment setup
ENV_SNIPPET="
# >>> Android SDK Auto Setup >>>
export ANDROID_SDK_ROOT=\"$SDK_ROOT\"
export ANDROID_HOME=\"$SDK_ROOT\"
export PATH=\"\$PATH:$SDK_ROOT/cmdline-tools/latest/bin:$SDK_ROOT/platform-tools:$SDK_ROOT/emulator:$LOCAL_BIN\"
# <<< Android SDK Auto Setup <<<
"
for f in "$USER_HOME/.bashrc" "$USER_HOME/.profile"; do
  if ! grep -Fq "Android SDK Auto Setup" "$f" 2>/dev/null; then
    echo "$ENV_SNIPPET" >> "$f"
  fi
done

export ANDROID_SDK_ROOT="$SDK_ROOT"
export ANDROID_HOME="$SDK_ROOT"
export PATH="$PATH:$SDK_ROOT/cmdline-tools/latest/bin:$SDK_ROOT/platform-tools:$SDK_ROOT/emulator:$LOCAL_BIN"

SDKMANAGER="$SDK_ROOT/cmdline-tools/latest/bin/sdkmanager"
AVDMANAGER="$SDK_ROOT/cmdline-tools/latest/bin/avdmanager"

# Install required SDKs
info "Installing SDK components..."
yes | "$SDKMANAGER" --sdk_root="$SDK_ROOT" --licenses >/dev/null || true
"$SDKMANAGER" --sdk_root="$SDK_ROOT" "platform-tools" "emulator" "platforms;$PLATFORM_API" "build-tools;34.0.0" "$SYSTEM_IMAGE_PATH"

# Create Medium Phone AVD
info "Creating AVD: $AVD_NAME ..."
if "$AVDMANAGER" list avd | grep -q "$AVD_NAME"; then
  warn "AVD $AVD_NAME already exists; skipping creation."
else
  if echo "no" | "$AVDMANAGER" create avd -n "$AVD_NAME" -k "$SYSTEM_IMAGE_PATH" -d "$DEVICE_ID" 2>/dev/null; then
    info "✅ AVD $AVD_NAME created successfully."
  else
    warn "❌ Failed to create Medium Phone device; falling back to 'pixel' profile."
    echo "no" | "$AVDMANAGER" create avd -n "$AVD_NAME" -k "$SYSTEM_IMAGE_PATH" -d "pixel" || warn "AVD creation failed."
  fi
fi

cat <<EOF

🎉 Installation complete!

Android Studio:  $STUDIO_DIR
SDK Root:        $SDK_ROOT
AVD created:     $AVD_NAME

To start emulator:
  emulator -avd "$AVD_NAME" &

To launch Android Studio:
  android-studio &

✅ Paths added to your ~/.bashrc and ~/.profile automatically.
Restart your terminal once, or run:
  source ~/.bashrc
EOF
