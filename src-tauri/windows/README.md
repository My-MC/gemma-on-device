The MSI template is copied from Tauri CLI 2.12.0:
https://github.com/tauri-apps/tauri/blob/tauri-cli-v2.12.0/crates/tauri-bundler/src/bundle/windows/msi/main.wxs

The only change is `CompressionLevel="none"` on the embedded CAB's `Media` element.
The upstream template is used under the MIT license in `LICENSE-MIT`.
Review changes against the matching upstream template when updating Tauri CLI.
