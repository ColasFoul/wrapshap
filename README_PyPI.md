This package allows you to perform SHAP-based feature selection and exploration for high-dimensional data, with both CPU and GPU support.

**Paper:** [Furger et al. (2026), *Wrapshap: a high dimension feature selection framework to explore biological systems through machine learning explanations*](https://www.nature.com/articles/s41540-026-00779-z).

## How to install Wrapshap

In order to install the CPU implementation of Wrapshap, simply run:

```bash
pip install wrapshap
```

If you can afford it, the GPU implementation is much faster and comes with additional methods. To install it instead, please run:

```bash
pip install "wrapshap[gpu]"
```

You should now be able to import wrapshap utilities.

## **Warning: If you are encountering import issues, please try using Python version 3.9**

### For the CPU implementation:

```python
from wrapshap import Wrapshap
```
or for the already wrapped models:

```python
from wrapshap.already_wrapped import WrappedBinaryClassifierXGB, WrappedRegressorXGB
```

### For the GPU implementation:

```python
from wrapshap.gpu_implementation import WrapshapGPU
```
or for the already wrapped models:

```python
from wrapshap.gpu_implementation.already_wrapped_gpu import WrappedRegressorXGB_GPU, WrappedBinaryClassifierXGB_GPU
```

For more details about how to use the package, please head over to our [GitHub repository](https://github.com/fefurger/wrapshap) for example notebooks.
