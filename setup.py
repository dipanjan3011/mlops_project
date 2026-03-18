from setuptools import setup, find_packages

setup(
    name="mlops_project",
    version="0.1.0",
    description="Telco Customer Churn Prediction - MLOps Demo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
)
