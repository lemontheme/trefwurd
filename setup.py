from setuptools import setup

setup(
    name="trefwurd",
    version="0.0.1-alpha",
    author="lemontheme (adriaan lemmens)",
    email="lemontheme@gmail.com",
    license="LICENSE",
    description="Rule-based lemmatization for Dutch.",
    install_requires=[
        "tqdm"
    ],
    long_description_content_type="text/markdown",
    include_package_data=True
)

