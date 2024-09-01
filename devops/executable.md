# Programming Executable Files for Windows
* There are multiple ways to create an executable file

## 1. Batch Conversion
* First we create a batch file that has the commands we have to execute in order to run the program.

## 2. Rust bin file Cross-compilation
* I usually work in MacOS, so if I locally compile my rust code it won't run in Windows.
* Fortunately, there is a **cross-platform support that lets us compile windows executable from MacOS**

1) Add cross compile toolchain

```bash
brew install mingw-w64
```

2) add windows OS in the rustup target binary:
```bash
rustup target add x86_64-pc-windows-gnu
```