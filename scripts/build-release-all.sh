#!/usr/bin/env bash
# NALA-ModelForge-RS Tri-OS Native Cross-Compile & Release Factory
# Builds & verifies:
#   - macOS Apple Silicon (ARM64) .app + .dmg
#   - macOS Intel (x86_64) .app + .dmg
#   - Ubuntu / Debian Linux (amd64 & arm64) static musl ELF .deb + .tar.gz
#   - Windows 11 (x86_64) Native PE32+ .exe + .zip
#   - CycloneDX 1.5 SBOM + SHA256SUMS.txt + sHeriFF Zero-PII Binary Gate
set -euo pipefail

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.."
ROOT="$PWD"
VERSION="$(tr -d ' \n\r' < VERSION)"
DIST="$ROOT/dist"
rm -rf "$DIST"
mkdir -p "$DIST"

BASE_RUSTFLAGS="-D warnings --remap-path-prefix=$HOME=/build --remap-path-prefix=$ROOT=/src"
RUST_LLD="$(find "$HOME/.rustup" -name "rust-lld" 2>/dev/null | head -n 1 || true)"

echo "============================================================"
echo " NALA-ModelForge-RS v${VERSION} — Tri-OS Native Build"
echo "============================================================"

echo "==> [Gate 0] Validating embedded JS syntax in ui/index.html ..."
node -e '
const fs = require("fs");
const html = fs.readFileSync("ui/index.html", "utf8");
const m = html.match(/<script[^>]*>([\s\S]*?)<\/script>/);
if (!m) { console.error("No <script> block found"); process.exit(1); }
new Function(m[1]);
console.log("PASS_JS_SYNTAX");
'

echo "==> [Gate 1] Running cargo test (strict -D warnings) ..."
RUSTFLAGS="$BASE_RUSTFLAGS" cargo test --all-targets

echo "==> [Gate 2] Packaging NALA-ModelForge App Icon (.png -> .icns) & 5-Language PDF Guides ..."
mkdir -p assets
if [ ! -f assets/icon_1024.png ]; then
  swift scripts/make_icon.swift assets/icon_1024.png
fi
rm -rf assets/AppIcon.iconset
mkdir -p assets/AppIcon.iconset
for s in 16 32 64 128 256 512; do
  sips -z $s $s assets/icon_1024.png --out "assets/AppIcon.iconset/icon_${s}x${s}.png" >/dev/null
  s2=$((s * 2))
  sips -z $s2 $s2 assets/icon_1024.png --out "assets/AppIcon.iconset/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns assets/AppIcon.iconset -o assets/AppIcon.icns
rm -rf assets/AppIcon.iconset
swift scripts/generate_pdfs.swift "$ROOT"
cp docs/NALA-ModelForge-Guide-*.pdf "$DIST/"

check_zero_pii_bin() {
  local bin="$1"
  if strings "$bin" | grep -qE '(/Users/[a-zA-Z0-9._-]+/|/home/[a-zA-Z0-9._-]+/|192\.168\.[0-9]+\.[0-9]+|100\.(6[4-9]|[7-9][0-9]|1[0-2][0-7])\.[0-9]+\.[0-9]+|gh[op]_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_-]{16,})'; then
    echo "FATAL: sHeriFF Zero-PII Gate FAILED on $bin" >&2
    exit 1
  fi
  echo "  ✓ sHeriFF Zero-PII Binary PASS: $(basename "$bin") ($(du -h "$bin" | awk '{print $1}'))"
}

build_macos_bundle() {
  local RT="$1" ST="$2" ARCH_LABEL="$3"
  local STAGE="$DIST/stage-macos-${ARCH_LABEL}"
  local APP="$STAGE/NALA-ModelForge.app"
  local DMG="$DIST/NALA-ModelForge-${VERSION}-macOS-${ARCH_LABEL}.dmg"
  echo "==> [macOS ${ARCH_LABEL}] Compiling Rust (${RT}) + Swift/AppKit (${ST}) ..."
  RUSTFLAGS="$BASE_RUSTFLAGS" cargo build --release --target "$RT"
  rm -rf "$STAGE"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cp "target/${RT}/release/nala-modelforge-rs" "$APP/Contents/MacOS/nala-modelforge-engine"
  swiftc -O -target "$ST" macos_ui/main.swift -framework Cocoa -framework WebKit -o "$APP/Contents/MacOS/NALA-ModelForge"
  swiftc -O -target "$ST" macos_ui/metal_runner.swift -framework Metal -framework Foundation -o "$APP/Contents/MacOS/nala-modelforge-metal"
  cp assets/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
  cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleExecutable</key><string>NALA-ModelForge</string>
  <key>CFBundleIdentifier</key><string>ch.nala.modelforge-rs</string>
  <key>CFBundleName</key><string>NALA-ModelForge</string>
  <key>CFBundleDisplayName</key><string>NALA-ModelForge</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
EOF
  codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
  check_zero_pii_bin "$APP/Contents/MacOS/nala-modelforge-engine"
  check_zero_pii_bin "$APP/Contents/MacOS/NALA-ModelForge"
  check_zero_pii_bin "$APP/Contents/MacOS/nala-modelforge-metal"
  ln -s /Applications "$STAGE/Applications"
  hdiutil create -volname "NALA-ModelForge ${VERSION}" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
  hdiutil verify "$DMG" >/dev/null
  echo "  ✓ DMG verified: $(basename "$DMG") ($(du -h "$DMG" | awk '{print $1}'))"

  # Keep local AppleSilicon .app copy for /Applications installation
  if [ "$ARCH_LABEL" = "AppleSilicon-ARM64" ]; then
    rm -rf "$DIST/NALA-ModelForge.app"
    cp -R "$APP" "$DIST/NALA-ModelForge.app"
  fi
  rm -rf "$STAGE"
}

build_linux_deb() {
  local RT="$1" DEB_ARCH="$2" TAR_ARCH="$3"
  echo "==> [Ubuntu/Debian ${DEB_ARCH}] Compiling static musl Rust (${RT}) ..."
  RUSTFLAGS="$BASE_RUSTFLAGS -C linker=$RUST_LLD -C linker-flavor=ld.lld -C target-feature=+crt-static" \
    cargo build --release --target "$RT"
  local BIN="target/${RT}/release/nala-modelforge-rs"
  file "$BIN"
  check_zero_pii_bin "$BIN"

  local TGZ_DIR="$DIST/stage-tgz-${TAR_ARCH}/nala-modelforge-rs-${VERSION}-linux-${TAR_ARCH}"
  mkdir -p "$TGZ_DIR"
  cp "$BIN" "$TGZ_DIR/nala-modelforge-rs"
  cp README.md LICENSE install.sh "$TGZ_DIR/"
  COPYFILE_DISABLE=1 tar --format ustar --no-xattrs --no-mac-metadata --uid 0 --gid 0 --uname root --gname root -czf "$DIST/nala-modelforge-rs-${VERSION}-linux-${TAR_ARCH}.tar.gz" -C "$DIST/stage-tgz-${TAR_ARCH}" "$(basename "$TGZ_DIR")"
  rm -rf "$DIST/stage-tgz-${TAR_ARCH}"

  local WORK="$DIST/stage-deb-${DEB_ARCH}"
  local DATA="$WORK/data"
  local CTRL="$WORK/control"
  mkdir -p "$DATA/usr/bin" "$DATA/usr/share/applications" "$DATA/usr/share/icons/hicolor/256x256/apps" "$CTRL"
  cp "$BIN" "$DATA/usr/bin/nala-modelforge-rs"
  chmod 0755 "$DATA/usr/bin/nala-modelforge-rs"
  sips -z 256 256 assets/icon_1024.png --out "$DATA/usr/share/icons/hicolor/256x256/apps/nala-modelforge.png" >/dev/null
  cat > "$DATA/usr/share/applications/nala-modelforge.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=NALA-ModelForge
Comment=100% Native Safe Rust LLM Precision Quantizer, Abliteration & Multi-Agent Studio
Exec=/usr/bin/nala-modelforge-rs --open
Icon=nala-modelforge
Terminal=false
Categories=Development;Science;
EOF
  local INSTALLED_KB
  INSTALLED_KB="$(du -sk "$DATA" | awk '{print $1}')"
  cat > "$CTRL/control" <<EOF
Package: nala-modelforge-rs
Version: ${VERSION}
Section: devel
Priority: optional
Architecture: ${DEB_ARCH}
Installed-Size: ${INSTALLED_KB}
Maintainer: NALA-OS Core <opensource@nala.box>
Description: 100% Native Safe Rust LLM Precision Quantizer, Abliteration & MCP Workstation
 Domain-selective mixed-precision quantization (Coding FAT + Bonsai 1-Bit / Gyro-Coder),
 asymmetric multi-user KV-cache planner, norm-preserving abliteration, multi-LoRA merger,
 NALA-RAG & NALA-Grid connectors, and 28-tool JSON-RPC 2.0 MCP server.
EOF
  printf "2.0\n" > "$WORK/debian-binary"
  (cd "$CTRL" && COPYFILE_DISABLE=1 tar --format ustar --no-xattrs --no-mac-metadata --uid 0 --gid 0 --uname root --gname root -czf "$WORK/control.tar.gz" .)
  (cd "$DATA" && COPYFILE_DISABLE=1 tar --format ustar --no-xattrs --no-mac-metadata --uid 0 --gid 0 --uname root --gname root -czf "$WORK/data.tar.gz" .)
  local DEB="$DIST/nala-modelforge-rs_${VERSION}_${DEB_ARCH}.deb"
  ./target/aarch64-apple-darwin/release/nala-modelforge-rs --pack-deb "$WORK" "$DEB" >/dev/null
  local DEB_BYTES
  DEB_BYTES="$(wc -c < "$DEB" | tr -d ' ')"
  if [ "$DEB_BYTES" -lt 300000 ]; then
    echo "FATAL: Anti-Empty-Image Gate FAILED — $DEB is only $DEB_BYTES bytes!" >&2
    exit 1
  fi
  echo "  ✓ DEB verified (Pure-Rust POSIX ar + >=300KB Gate): $(basename "$DEB") ($(du -h "$DEB" | awk '{print $1}'))"
  rm -rf "$WORK"
}

build_windows_x64() {
  local RT="x86_64-pc-windows-gnu"
  echo "==> [Windows x86_64] Compiling Native PE32+ (${RT}) ..."
  RUSTFLAGS="$BASE_RUSTFLAGS" \
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc \
    cargo build --release --target "$RT"
  local BIN="target/${RT}/release/nala-modelforge-rs.exe"
  file "$BIN"
  check_zero_pii_bin "$BIN"
  cp "$BIN" "$DIST/NALA-ModelForge-${VERSION}-windows-x86_64.exe"

  local WSTAGE="$DIST/stage-win-x64/NALA-ModelForge-${VERSION}-windows-x86_64"
  mkdir -p "$WSTAGE"
  cp "$BIN" "$WSTAGE/NALA-ModelForge.exe"
  cp README.md LICENSE install.ps1 "$WSTAGE/"
  (cd "$DIST/stage-win-x64" && zip -qr "$DIST/NALA-ModelForge-${VERSION}-windows-x86_64.zip" "$(basename "$WSTAGE")")
  rm -rf "$DIST/stage-win-x64"
  echo "  ✓ Windows x64 EXE + ZIP built"
}

build_macos_bundle aarch64-apple-darwin arm64-apple-macos13.0 AppleSilicon-ARM64
build_macos_bundle x86_64-apple-darwin x86_64-apple-macos13.0 Intel-x86_64
build_linux_deb x86_64-unknown-linux-musl amd64 x86_64
build_linux_deb aarch64-unknown-linux-musl arm64 aarch64
build_windows_x64

echo "==> [Gate 3] Running Pure-Rust Deslop Gate, Self-Test & Generating CycloneDX 1.5 SBOM + SHA256SUMS ..."
./target/aarch64-apple-darwin/release/nala-modelforge-rs --deslop-audit
./target/aarch64-apple-darwin/release/nala-modelforge-rs --self-test
./target/aarch64-apple-darwin/release/nala-modelforge-rs --emit-sbom > "$DIST/SBOM_SLSA_v${VERSION}.json"

(cd "$DIST" && find . -maxdepth 1 -type f ! -name "SHA256SUMS.txt" -exec shasum -a 256 {} + > SHA256SUMS.txt)
echo "============================================================"
echo " ALL TRI-OS RELEASE ARTIFACTS VERIFIED IN $DIST:"
ls -lh "$DIST"
echo "============================================================"
