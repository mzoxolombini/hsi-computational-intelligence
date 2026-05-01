# setup_repository.py
import os


def create_directory_structure():
    """Create the complete repository structure"""

    # Directories to create
    directories = [
        'src/models',
        'src/optimization',
        'src/evaluation',
        'src/utils',
        'data/raw',
        'data/processed',
        'configs',
        'notebooks',
        'results/figures',
        'results/tables',
        'tests'
    ]

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")

    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/optimization/__init__.py',
        'src/evaluation/__init__.py',
        'src/utils/__init__.py',
        'tests/__init__.py'
    ]

    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization\n')
        print(f"✓ Created: {init_file}")

    # Create requirements.txt
    requirements = """numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.4.0
pandas>=1.3.0
opencv-python>=4.5.0
jupyter>=1.0.0
"""
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✓ Created: requirements.txt")

    print("\n✅ Repository structure created successfully!")
    print("\nNext steps:")
    print("1. Copy the Python code files to their respective directories")
    print("2. Run: git add .")
    print("3. Run: git commit -m 'Add implementation'")
    print("4. Run: git push origin main")


if __name__ == "__main__":
    create_directory_structure()