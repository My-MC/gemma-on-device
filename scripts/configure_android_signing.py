"""Configure the generated Android Gradle project for release signing."""

from pathlib import Path


def main():
    path = Path("src-tauri/gen/android/app/build.gradle.kts")
    source = path.read_text()
    android_block = "android {\n"
    release_block = '        getByName("release") {\n'
    if android_block not in source or release_block not in source:
        raise SystemExit(f"Android Gradle signing blocks not found in {path}")
    signing = '''val keystorePropertiesFile = rootProject.file("keystore.properties")
val keystoreProperties = java.util.Properties().apply {
    keystorePropertiesFile.inputStream().use { load(it) }
}

android {
    signingConfigs {
        create("release") {
            keyAlias = keystoreProperties["keyAlias"] as String
            keyPassword = keystoreProperties["keyPassword"] as String
            storeFile = file(keystoreProperties["storeFile"] as String)
            storePassword = keystoreProperties["password"] as String
        }
    }
'''
    source = source.replace(android_block, signing, 1)
    source = source.replace(
        release_block,
        release_block + '            signingConfig = signingConfigs.getByName("release")\n',
        1,
    )
    path.write_text(source)


if __name__ == "__main__":
    main()
