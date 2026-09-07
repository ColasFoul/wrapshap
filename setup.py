from setuptools import setup, find_packages

setup(
    name='wrapshap',
    version='0.1',
    packages=find_packages(),
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
            'tensorflow>=2.16.1'
        ]
    },
    author='Félix Furger',
    author_email='fefurger@hotmail.com',
    description=''
)
