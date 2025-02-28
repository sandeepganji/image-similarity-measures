import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="image-similarity-measures",
    version="0.3.5",
    description="Evaluation metrics to assess the similarity between two images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandeepganji/image-similarity-measures",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable"
    ],
    extras_require={
        "rasterio": ["rasterio"],
        "speedups": ["pyfftw"],
    },
    install_requires=["numpy", "scikit-image", "opencv-python", "phasepack", "nibabel"],
    python_requires=">=3.7, <3.10",
    entry_points = {
        'console_scripts': ['image-similarity-measures=image_similarity_measures.evaluate:main'],
    },
)

# setuptools.setup(
    # name="image-similarity-measures",
    # version="0.3.5",
    # author="UP42",
    # author_email="support@up42.com",
    # description="Evaluation metrics to assess the similarity between two images.",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="https://github.com/up42/image-similarity-measures",
    # packages=setuptools.find_packages(),
    # license="MIT",
    # classifiers=[
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
        # "Intended Audience :: Science/Research",
        # "Development Status :: 5 - Production/Stable"
    # ],
    # extras_require={
        # "rasterio": ["rasterio"],
        # "speedups": ["pyfftw"],
    # },
    # install_requires=["numpy", "scikit-image", "opencv-python", "phasepack", "nibabel"],
    # python_requires=">=3.7, <3.10",
    # entry_points = {
        # 'console_scripts': ['image-similarity-measures=image_similarity_measures.evaluate:main'],
    # },
# )
