# 🚀 ResNet Training Notebooks

## 🌟 Overview
This repository contains **Jupyter notebooks** for training a **ResNet** model on a dataset over different numbers of epochs. The models were trained for **50, 100, and 600 epochs**, respectively, to analyze performance improvements with increased training time. The goal was to evaluate how **training duration impacted accuracy, loss, and generalization ability**.

---

## 📂 Files

1. 📌 **resnet_train_test(50 epochs).ipynb**  
   - 🏃‍♂️ Trained a **ResNet model for 50 epochs**.  
   - ⚡ Quick experimentation and establishing a **baseline performance**.
   - 🔍 Recommended for **early-stage model evaluation**.

2. 📌 **resnet_train_test(100 epochs).ipynb**  
   - 🔬 Trained a **ResNet model for 100 epochs**.  
   - ⚖️ Balanced **training time vs model performance**.
   - 📈 Helped analyze improvements over the **50-epoch model**.

3. 📌 **resnet_train_test(600 epochs).ipynb**  
   - 🔥 Trained a **ResNet model for 600 epochs**.  
   - 🎯 Focused on **extensive training for optimal accuracy**.
  

---

## ⚙️ Dependencies
Before running the notebooks, make sure the following dependencies were installed:

```bash
pip install torch torchvision numpy matplotlib
```

---

## 🛠️ Usage
1. **Open the desired notebook** in Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook resnet_train_test(50 epochs).ipynb
   ```
2. **Run all cells** sequentially to train the **ResNet model**.
3. **Modify hyperparameters** as needed for experimentation:
   - 🎯 **Batch Size**: Adjust based on **GPU memory** availability.
   - ⚡ **Learning Rate**: Default was typically `0.001`, but could be fine-tuned.
   - 🏋️ **Optimizer**: Used **Adam, SGD**, or other optimizers as needed.
   - 📊 **Data Augmentation**: Could be added for **better generalization**.

---

## 📊 Results & Observations
### 🔹 **Key Insights from Running the Notebooks**:

1. **⏳ Training Time vs Accuracy**  
   - ✅ The **50-epoch model** reached a moderate accuracy but may have **underfitted**.
   - ⚖️ The **100-epoch model** provided **more stable performance**.
   - 🚀 The **600-epoch model** achieved the **best accuracy** but risked **overfitting** if not regularized.

2. **📉 Loss Reduction**  
   - Loss **progressively decreased** over epochs.
   - If loss **stagnated or increased**, hyperparameters needed **adjustments**.

3. **🛡️ Overfitting Analysis**
   - 🚨 Higher epochs improved training accuracy, but **validation accuracy needed monitoring**.
   - 🛠️ Techniques like **dropout, batch normalization, and learning rate scheduling** helped reduce **overfitting**.

---



