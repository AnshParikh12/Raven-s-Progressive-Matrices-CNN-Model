# Raven's Progressive Matrices CNN Model

A PyTorch-based implementation for reasoning over Raven's Progressive Matrices.

## Project Structure

- `src/`: source code for dataset loading, model definitions, and training.
- `tests/`: runnable example/test scripts.
- `data/`: local dataset files for experimentation.
- `requirements.txt`: Python dependencies.
- `.gitignore`: files and folders excluded from version control.

## Local dataset

This repository does not include the full RAVEN dataset.
The current local dataset contains `100` generated Raven problem files under:

```text
data/raven_test/distribute_nine
```

Each file is a separate problem instance generated using the RAVEN dataset code.
If you want to generate a larger local dataset (for example, 10,000+ problems), follow the official RAVEN repository instructions and store the generated files in the same local directory.

### Credit

The dataset is generated using the RAVEN dataset project:

- RAVEN: A Dataset for Relational and Analogical Visual rEasoNing
- Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, Song-Chun Zhu
- CVPR 2019

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Local dataset

This repository does not bundle the full RAVEN dataset.
The current local dataset contains `10000` generated Raven problem files in:

```text
data/raven_test/distribute_nine
```

To obtain a larger dataset:

1. Clone or download the official RAVEN repository separately.
2. Follow the RAVEN repo instructions to generate problem files.
3. Store the generated `.npz` files in:

```text
data/raven_test/distribute_nine
```


## Usage

Run a test script from the project root:

```powershell
.\.venv\Scripts\python.exe tests\test_dataloader.py
```

If you need package-style execution with `src/` imports:

```powershell
.\.venv\Scripts\python.exe -m src.tests.test_dataloader
```


## Result

Current best accuracy received is 28.75%. Simple CNN is not able to achieve a better result. A future experiment could be to use ResNet or other pretrained model that has already learned basic features like lines, shapes, points, etc. and then train it to learn patterns and transformations.