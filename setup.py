from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="6404voroninpm-lab5",
    version="1.0.0",
    author="Voronin",
    author_email="your.email@example.com",
    description="Асинхронная обработка изображений животных (Лабораторная работа 5)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/6404voroninpm-lab5",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Education :: Computer Science"
    ],
    python_requires=">=3.7",
    install_requires=[
        "aiohttp>=3.8.0",
        "aiofiles>=23.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "process-cats=6404voroninpm_lab5.main_lab5:main_lab5",
        ],
    },
    package_data={
        "6404voroninpm_lab5": ["*.py"],
    },
    keywords="async image-processing cats api edge-detection",
    project_urls={
        "Source": "https://github.com/yourusername/6404voroninpm-lab5",
        "Bug Reports": "https://github.com/yourusername/6404voroninpm-lab5/issues",
    },
)