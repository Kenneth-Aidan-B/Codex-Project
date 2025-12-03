"""Setup configuration for the Genomic Hearing Screening Platform."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="genomic-hearing-screening",
    version="1.0.0",
    author="Genomic Hearing Screening Platform Team",
    author_email="contact@hearingscreening.org",
    description="AI-powered genomic screening platform for newborn hearing loss",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/Codex-Project",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.7.0",
            "isort>=5.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hearing-screening=api.app:main",
        ],
    },
)
