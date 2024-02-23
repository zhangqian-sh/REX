from setuptools import setup, find_packages

setup(
    name="rex",
    version="0.1.0",
    author="Qian Zhang",
    author_email="qian_zhang1@brown.edu",
    description="Run Experiments with JAX",
    long_description="Run Experiments with JAX",
    packages=find_packages(),
    install_requires=["jax", "equinox", "optax", "ml_collections", "h5py"],
)
