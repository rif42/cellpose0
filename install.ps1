# 1. Install Python and Git using Windows Package Manager
winget install -e --id Python.Python.3.10
winget install -e --id Git.Git

# Refresh Path
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# 2. Clone Repo
git clone https://github.com/rif42/cellpose0.git
cd cellpose0

# 3. Create and access virtual environment
python -m venv venv
.\cellpose0\cellpose\Scripts\Activate.ps1

# 4. Install dependencies and cellpose
pip install --upgrade pip
pip install "cellpose[gui]"

# 5. Execute
python -m cellpose