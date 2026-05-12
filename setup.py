from setuptools import setup, find_packages

setup(
    name="hybrid_forecaster",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # main requirements
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
    ],
    author="Sharif Jubaid Redwan Rusho",
    description="esoc hybrid project",
)