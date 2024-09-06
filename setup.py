from setuptools import setup

# with open("README.rst", "r", encoding="utf-8") as f:
#     __long_description__ = f.read()

if __name__ == "__main__":
    setup(
        name = "SpaGT",
        version = "1.0.0",
        description = "Spatialyinformed Graph Transformer",
        url = "https://github.com/xy428/SpaGT.git",
        author = "Xinyu Bao",
        author_email = "xy",
        license = "MIT",
        packages = ["SpaGT"],
        install_requires = ["requests"],
        zip_safe = False,
        include_package_data = True,
        long_description = """ Long Description """,
        long_description_content_type="text/markdown",
    )
