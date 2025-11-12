from setuptools import setup, find_packages

setup(
    name="jane-street-prediction",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.3.3",
        "numpy>=2.3.4",
        "scikit-learn>=1.7.2",
        "xgboost>=3.1.1",
        "tensorflow>=2.20.0",
    ],
    python_requires=">=3.8",
)
