# Securing Transformer Models using Fully Homomorphic Encryption

This repository demonstrates how to perform homomorphic encryption on a BERT‑Tiny embedding using the OpenFHE (Boolean) Python wrapper. The example encrypts a sentence embedding, applies a secure dot‑product with random weights and bias, and decrypts the result.

## Prerequisites

- Linux: Ubuntu LTS 20.04, 22.04 or 24.04 (only these versions are currently supported) ([github.com](https://github.com/openfheorg/openfhe-python?utm_source=chatgpt.com))
- C++ compiler (GCC ≥ 9 or Clang ≥ 10)
- CMake ≥ 3.16
- Make, Git, Python 3.8+
- `pip` for Python package management

## 1. Install OpenFHE C++ Library

```bash
# Clone the OpenFHE core library
git clone https://github.com/openfheorg/openfhe.git
cd openfhe

# Create and enter a build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j    # Use -j$(nproc) to parallelize

# Install to /usr/local (requires sudo)
sudo make install
```
This installs the OpenFHE headers and shared libraries to your system. ([openfheorg.github.io](https://openfheorg.github.io/openfhe-python/html/index.html?utm_source=chatgpt.com), [openfhe-development.readthedocs.io](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html?utm_source=chatgpt.com))

## 2. Install the OpenFHE Python Wrapper

### Option A: Pre‑built via PyPI

```bash
pip install openfhe
```
Supports Ubuntu LTS 20.04/22.04/24.04 with precompiled wheels. ([github.com](https://github.com/openfheorg/openfhe-python?utm_source=chatgpt.com))

### Option B: Build from Source

```bash
# Clone the Python wrapper
git clone https://github.com/openfheorg/openfhe-python.git
cd openfhe-python

# Install pybind11 for C++ bindings
pip install "pybind11[global]"

# Build and install the wrapper
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/usr/local
make -j
sudo make install
```
If you installed OpenFHE to a non-default location, pass its path via `-DCMAKE_PREFIX_PATH` and ensure the `.so` library is on your `LD_LIBRARY_PATH` or `PYTHONPATH`. ([git-crysp.uwaterloo.ca](https://git-crysp.uwaterloo.ca/iang/openfhe-python-fork?utm_source=chatgpt.com))

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 4. Usage

```bash
# Run on a custom sentence
# Add desired sentence to be tested by changing sentence in line 178
python openfhe_bert.py
```

Expected output:
```
[INFO] Plaintext BERT‑Tiny embedding…
[INFO] Setting up BinFHE…
[INFO] Homomorphic dot‑product ( 128 dimensions)…
Encrypted output int : <integer>
De‑quantised value  : <float>
Prediction          : positive|negative
FHE compute time    : <seconds>s
```



