from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlrpc",
    author="Sri Tikkireddy",
    author_email="sri.tikkireddy@databricks.com",
    description="Deploy FastAPI applications on MLFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        'mlrpc': ['**/*.html'],
    },
    url="https://github.com/stikkireddy/mlflow-rpc",
    packages=find_packages(),
    install_requires=["pathspec",
                      "mlflow-skinny[databricks]",
                      "fastapi",
                      "pandas",
                      "httpx",
                      "databricks-sdk"],
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'dbtunnel = dbtunnel.cli.cli:cli',
        ],
    },
    python_requires=">=3.10",
)
