# setup_github.ps1 - Windows PowerShell script

# Clone the repository
git clone https://github.com/mzoxolombini/hsi-computational-intel.git
cd hsi-computational-intel

# Create directory structure
New-Item -ItemType Directory -Force -Path src\models, src\optimization, src\evaluation, src\utils
New-Item -ItemType Directory -Force -Path data\raw, data\processed
New-Item -ItemType Directory -Force -Path configs
New-Item -ItemType Directory -Force -Path notebooks
New-Item -ItemType Directory -Force -Path results\figures, results\tables
New-Item -ItemType Directory -Force -Path tests

# Create __init__.py files
New-Item -ItemType File -Force -Path src\__init__.py
New-Item -ItemType File -Force -Path src\models\__init__.py
New-Item -ItemType File -Force -Path src\optimization\__init__.py
New-Item -ItemType File -Force -Path src\evaluation\__init__.py
New-Item -ItemType File -Force -Path src\utils\__init__.py
New-Item -ItemType File -Force -Path tests\__init__.py

# Create requirements.txt
@"
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.4.0
pandas>=1.3.0
opencv-python>=4.5.0
jupyter>=1.0.0
"@ | Out-File -FilePath requirements.txt -Encoding utf8

Write-Host "Directory structure created successfully!"