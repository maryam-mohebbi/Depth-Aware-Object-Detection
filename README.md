## Configuration and Setup
To get started with the project, you'll need to set up your environment and install necessary dependencies. This guide will walk you through the steps using Poetry, a tool for dependency management and packaging in Python.

### Installing Poetry

[Poetry](https://python-poetry.org/) is a tool for dependency management and packaging in Python. To install Poetry, execute the following command in your terminal:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

This command retrieves and executes the Poetry installation script.

### Setting Up the Project Environment

After installing Poetry, you can set up the project's environment and install its dependencies. Ensure your Python version is `3.10.10` as it is the version used for this project.

1. **Install Dependencies**

   Run the following command in the project directory to install the required dependencies:

   ```bash
   poetry install
   ```

2. **Activate the Environment**

   To activate the Poetry-managed virtual environment, use:

   ```bash
   poetry shell
   ```

### Post-Activation Steps

Due to version conflicts between dependencies, certain libraries need to be installed using pip after activating the environment. Execute the commands below to install these specific libraries:

```bash
pip install ultralytics install super-gradients
```

### Pretrained Model Download

Download the pretrained "Segment Anything Model" and place it in the `data/sam_weights` folder. This model is essential for the project's functionality. Use the command below to download the model:

```bash
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P data/sam_weights/
```

Make sure you have the `wget` tool installed on your system to execute the download command successfully.