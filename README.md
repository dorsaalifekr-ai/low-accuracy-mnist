# MNIST — Minimal Model (Intentionally Low Accuracy) 

This repository contains a single Python script, `low_accuracy_mnist.py`, that **intentionally** trains a very small model on MNIST to demonstrate a basic deep‑learning workflow (Load → Train → Evaluate → Visualize).  
Because the architecture is tiny and the training subset is limited, the resulting performance is **deliberately low** (useful for teaching/experiments).

> If you need higher accuracy, see **Improve Accuracy (Optional)** below.


## Repository Contents
- `low_accuracy_mnist.py` — main training/evaluation script.
- `README.md` — this file.


## Requirements
Python 3.8+ recommended. Install dependencies:
```bash
pip install torch torchvision matplotlib
```
> On macOS or headless servers (e.g., Colab), the default Matplotlib backend (`TkAgg`) may not be available. See **Common Issues**.


## Quick Start (Local)
1. Clone/download and enter the folder:
```bash
git clone <YOUR-REPO-URL>
cd <YOUR-REPO-FOLDER>
```
2. Install dependencies:
```bash
pip install torch torchvision matplotlib
```
3. Run:
```bash
python low_accuracy_mnist.py
```
The MNIST dataset will be downloaded automatically into `data/` on first run. The script prints **Train Accuracy** per epoch and, at the end, **Test Accuracy**. It also shows a small grid of sample predictions.


## Quick Start (Google Colab)
- Upload `low_accuracy_mnist.py` or copy the code into a Colab cell.
- Install requirements:
```python
!pip install torch torchvision matplotlib
```
- Since Colab is headless, either **comment out** the Matplotlib backend line or switch it to `Agg`:
```python
# matplotlib.use("TkAgg")  # comment/remove in Colab, or:
# import matplotlib
# matplotlib.use("Agg")
```
- Replace `plt.show()` with `plt.savefig("preds.png", dpi=150)` if you need an image file rather than a pop‑up window.


## Code Structure (What the script does)
- **Model**: `SimpleDigitNet` with a single tiny hidden layer (5 units) → intentionally underpowered.
- **Data**: Trains on a **subset** of the MNIST training set (e.g., first 6,000 samples) to keep accuracy low on purpose.
- **Optimizer**: `SGD` with a small learning rate (e.g., 0.005).
- **Visualization**: `show_predictions` creates a simple grid of images with predicted labels.

These choices make it easier to contrast with stronger baselines later.


## Expected Results
Results vary slightly by environment and library versions. Do **not** expect typical MNIST accuracies (>97%). This project is for **didactic** purposes.


## Improve Accuracy (Optional)
If you want to turn this into a stronger baseline, consider the following incremental changes:
1. **Wider/Deeper MLP**
   - Increase the hidden layer (e.g., from 5 → 128 or 256).
   - Add one or two additional fully connected layers.
2. **Use the full training set** instead of a 6k subset.
3. **Optimizer**: switch to `Adam` with `lr=1e-3`.
4. **Train longer** (20–30 epochs) with early stopping.
5. **Use a small CNN** (Conv2d + MaxPool) instead of a plain MLP.
6. Add **Dropout/BatchNorm** and/or a **learning‑rate scheduler**.

Example tweak:
```python
self.layer1 = nn.Linear(28*28, 128)
self.output = nn.Linear(128, 10)
```


## Common Issues
- **Matplotlib backend error (`TkAgg`) on Colab/servers**  
  - Comment out `matplotlib.use("TkAgg")` or set:
    ```python
    import matplotlib
    matplotlib.use("Agg")
    ```
  - Use `plt.savefig("preds.png", dpi=150)` instead of `plt.show()`.
- **CUDA not available**  
  - The script will fall back to CPU automatically. For speed, run on a CUDA‑enabled machine.


## Tips for Reports/Slides
- Print cleaner logs (average loss per epoch, epoch time).
- Save the best checkpoint:
```python
torch.save(model.state_dict(), "mnist_simple.pt")
```
- Save the predictions grid:
```python
plt.savefig("preds.png", dpi=150)
```


## License
MIT. Add a `LICENSE` file if needed.


## Citation
If you use this repo for teaching/research, a simple reference to the repository/script is sufficient. For the dataset, please cite LeCun et al. (1998).
