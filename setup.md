
# Setup Instructions: Anaconda and Environment Creation

## Prerequisites

Before starting, ensure that you have access to the internet and administrative privileges on your computer.

## Step 1: Install Anaconda

Follow the instructions below based on your operating system to install Anaconda.

### Windows

1. Download the Anaconda installer for Windows from the official website: [Anaconda Windows Installer](https://www.anaconda.com/products/individual).
2. Run the downloaded installer and follow the prompts to complete the installation.
   - Ensure that you check the option to **Add Anaconda to the PATH environment variable** (optional but recommended).
   - Install for "Just Me" unless you require a system-wide installation.
3. Once the installation is complete, open the **Anaconda Prompt** from the Start menu.

### macOS and Linux

1. Download the Anaconda installer for macOS or Linux from the official website: [Anaconda macOS/Linux Installer](https://www.anaconda.com/products/individual).
2. Open a terminal and run the following command (replace `filename` with the name of the downloaded installer):

   ```bash
   bash ~/Downloads/filename.sh
   ```

3. Follow the prompts to complete the installation.
4. After installation, initialize Conda by running:

   ```bash
   conda init
   ```

5. Close and reopen your terminal to activate Conda.

## Step 2: Create a New Environment from `NIRenv.yml`

Once Anaconda is installed, you can create a new environment using the provided `NIRenv.yml` file.

1. Place the `NIRenv.yml` file in a directory of your choice.
2. Open the **Anaconda Prompt** (Windows) or your terminal (macOS/Linux).
3. Navigate to the directory containing the `NIRenv.yml` file. For example:

   ```bash
   cd path/to/your/directory
   ```

4. Run the following command to create the environment:

   ```bash
   conda env create -f NIRenv.yml
   ```

   This will create a new Conda environment with the settings and packages specified in the `NIRenv.yml` file.

5. Once the environment is created, activate it by running:

   ```bash
   conda activate NIRenv
   ```

## Step 3: Verify the Environment

To ensure the environment is set up correctly, you can check the installed packages by running:

```bash
conda list
```

This will display the list of packages installed in the `NIRenv` environment.

## Troubleshooting

- **"conda command not found"**: If you encounter this issue, ensure that Anaconda is added to your system’s PATH variable or try running `conda init` again.
- **Missing dependencies**: If some dependencies are not installed correctly, try updating Conda and recreating the environment:

  ```bash
  conda update conda
  conda env create -f NIRenv.yml --force
  ```

## Additional Information

- [Anaconda Documentation](https://docs.anaconda.com/)
- [Conda Documentation](https://docs.conda.io/)
