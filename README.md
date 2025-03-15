Overview

This repository contained Jupyter notebooks for training a ResNet model on a given dataset for different numbers of epochs. The models were trained for 50, 100, and 600 epochs, respectively, to analyze performance improvements with increased training time. The objective was to evaluate how training duration impacted model accuracy, loss, and generalization ability.

Files

resnet_train_test(50 epochs).ipynb

Trained a ResNet model for 50 epochs.

Suitable for quick experimentation and establishing a baseline performance.

Recommended for early-stage model evaluation.

resnet_train_test(100 epochs).ipynb

Trained a ResNet model for 100 epochs.

Provided a balance between training time and model performance.

Helped analyze improvements over the 50-epoch model.

resnet_train_test(600 epochs).ipynb

Trained a ResNet model for 600 epochs.

Intended for extensive training to achieve optimal accuracy and generalization.

Required more computational resources and time.

Dependencies

The following dependencies were required before running the notebooks:

pip install torch torchvision numpy matplotlib

Additional optional dependencies for logging and visualization:

pip install seaborn tensorboard

Usage

Opened the desired notebook in Jupyter Notebook or Jupyter Lab:

jupyter notebook resnet_train_test(50 epochs).ipynb

Ran all cells sequentially to train the ResNet model.

Modified hyperparameters as needed for experimentation:

Batch Size: Adjusted depending on GPU memory availability.

Learning Rate: Default was typically 0.001, but could be fine-tuned.

Optimizer: Used Adam, SGD, or other optimizers as needed.

Data Augmentation: Could be added to improve generalization.

Results & Observations

The following key insights were derived from running the notebooks:

Training Time vs Accuracy:

The 50-epoch model reached a moderate accuracy level but may have underfitted.

The 100-epoch model provided a more stable performance.

The 600-epoch model generally had the best accuracy but may have overfitted if not regularized.

Loss Reduction:

Loss progressively decreased over epochs.

If loss stagnated or increased, hyperparameters needed adjustments.

Overfitting Analysis:

Higher epochs improved training accuracy, but validation accuracy needed to be monitored.

Techniques like dropout, batch normalization, and learning rate scheduling helped mitigate overfitting.

Future Work

Implementing advanced data augmentation techniques to enhance model robustness.

Experimenting with different learning rate schedules such as Cosine Annealing or Cyclical Learning Rates.

Fine-tuning with pre-trained models for transfer learning to improve efficiency on specific datasets.

Hyperparameter tuning using Grid Search or Bayesian Optimization.
