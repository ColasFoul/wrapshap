from setuptools import setup, find_packages

from pathlib import Path

README = Path(__file__).parent / "README_PyPI.md"

setup(
    name='wrapshap',
    version='0.1.1',
    author='Félix Furger',
    author_email='fefurger@hotmail.com',
    description='SHAP-based feature selection and exploration for high-dimensional data',
    packages=find_packages(),
    long_description=README.read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/fefurger/wrapshap',
    install_requires=[
        'numpy>=1.26.4',
        'pandas>=2.2.2',
        'shap>=0.45.0',
        'scikit-learn>=1.4.2',
        'scipy>=1.13.0',
        'xgboost>=2.0.3',
        'lightgbm>=4.6.0',
        'seaborn>=0.13.2',
        'matplotlib>=3.8.4'
    ],
    extras_require={
        'gpu': [
            'tensorflow>=2.16.1, <=2.20'
        ]
    },
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license_files=("LICENSE",),
    python_requires=">=3.8",
    data_files=[("", ["README_PyPI.md"])],
)