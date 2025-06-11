#!/bin/bash

# Titans API Setup Script
# This script sets up the complete environment for the Titans API

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to install sudo if missing
install_sudo_if_needed() {
    if ! command_exists sudo; then
        print_status "sudo not found, installing it first..."
        local os=$(detect_os)
        
        case $os in
            "linux")
                if command_exists apt-get; then
                    apt-get update
                    apt-get install -y sudo
                elif command_exists yum; then
                    yum install -y sudo
                else
                    print_error "Cannot install sudo - package manager not detected"
                    exit 1
                fi
                print_success "sudo installed successfully"
                ;;
            *)
                print_error "Cannot install sudo on this system automatically"
                exit 1
                ;;
        esac
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    # Install sudo first if needed
    install_sudo_if_needed
    
    local os=$(detect_os)
    print_status "Installing system dependencies for $os..."
    
    case $os in
        "linux")
            if command_exists apt-get; then
                print_status "Installing dependencies with apt-get..."
                sudo apt-get update
                sudo apt-get install -y \
                    python3 \
                    python3-pip \
                    python3-venv \
                    python3-dev \
                    build-essential \
                    portaudio19-dev \
                    espeak-ng \
                    postgresql \
                    postgresql-contrib \
                    libpq-dev \
                    ffmpeg \
                    git
            elif command_exists yum; then
                print_status "Installing dependencies with yum..."
                sudo yum update -y
                sudo yum install -y \
                    python3 \
                    python3-pip \
                    python3-devel \
                    gcc \
                    gcc-c++ \
                    portaudio-devel \
                    espeak-ng \
                    postgresql \
                    postgresql-server \
                    postgresql-devel \
                    ffmpeg \
                    git
            else
                print_warning "Package manager not detected. Please install dependencies manually."
            fi
            ;;
        "macos")
            if command_exists brew; then
                print_status "Installing dependencies with Homebrew..."
                brew update
                brew install \
                    python3 \
                    portaudio \
                    espeak-ng \
                    postgresql \
                    ffmpeg \
                    git
            else
                print_error "Homebrew not found. Please install Homebrew first:"
                print_error "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            ;;
        "windows")
            print_warning "Windows detected. Please ensure you have:"
            print_warning "- Python 3.8+ installed"
            print_warning "- PostgreSQL installed"
            print_warning "- Git installed"
            print_warning "- Visual Studio Build Tools or Visual Studio with C++ support"
            ;;
        *)
            print_warning "Unknown OS. Please install dependencies manually."
            ;;
    esac
}

# Function to setup PostgreSQL
setup_postgresql() {
    print_status "Setting up PostgreSQL..."
    
    local os=$(detect_os)
    
    # Check if we're in a Docker container
    local in_docker=false
    if [ -f /.dockerenv ] || grep -q 'docker\|lxc' /proc/1/cgroup 2>/dev/null; then
        in_docker=true
        print_status "Detected Docker container environment"
    fi
    
    # Start PostgreSQL service
    case $os in
        "linux")
            if [ "$in_docker" = true ]; then
                print_status "Starting PostgreSQL in Docker container..."
                # In Docker, we need to initialize and start PostgreSQL manually
                if [ ! -d "/var/lib/postgresql/data" ] || [ -z "$(ls -A /var/lib/postgresql/data 2>/dev/null)" ]; then
                    print_status "Initializing PostgreSQL database..."
                    # Create postgres user if it doesn't exist
                    if ! id postgres >/dev/null 2>&1; then
                        useradd -m -s /bin/bash postgres
                    fi
                    
                    # Initialize database
                    mkdir -p /var/lib/postgresql/data
                    chown postgres:postgres /var/lib/postgresql/data
                    chmod 700 /var/lib/postgresql/data
                    
                    # Initialize the database cluster
                    sudo -u postgres /usr/lib/postgresql/*/bin/initdb -D /var/lib/postgresql/data
                fi
                
                # Start PostgreSQL
                print_status "Starting PostgreSQL server..."
                sudo -u postgres /usr/lib/postgresql/*/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/data/logfile start
                
                # Wait a moment for PostgreSQL to start
                sleep 3
                
            elif command_exists systemctl; then
                sudo systemctl start postgresql
                sudo systemctl enable postgresql
            elif command_exists service; then
                sudo service postgresql start
            else
                print_warning "Could not start PostgreSQL service automatically"
            fi
            ;;
        "macos")
            if command_exists brew; then
                brew services start postgresql
            fi
            ;;
    esac
    
    # Check if database exists, create if not
    if command_exists psql; then
        print_status "Checking PostgreSQL connection..."
        
        # Try to connect as postgres user
        local max_attempts=5
        local attempt=1
        local connected=false
        
        while [ $attempt -le $max_attempts ] && [ "$connected" = false ]; do
            print_status "Connection attempt $attempt/$max_attempts..."
            if sudo -u postgres psql -c '\q' 2>/dev/null; then
                connected=true
                print_success "PostgreSQL is running"
            else
                print_status "Waiting for PostgreSQL to start..."
                sleep 2
                attempt=$((attempt + 1))
            fi
        done
        
        if [ "$connected" = true ]; then
            # Create database and user if they don't exist
            print_status "Setting up database and user..."
            sudo -u postgres psql -c "CREATE DATABASE titans_db;" 2>/dev/null || print_warning "Database titans_db may already exist"
            sudo -u postgres psql -c "CREATE USER titans_user WITH PASSWORD 'titans_password';" 2>/dev/null || print_warning "User titans_user may already exist"
            sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE titans_db TO titans_user;" 2>/dev/null
            sudo -u postgres psql -c "ALTER USER titans_user CREATEDB;" 2>/dev/null
            
            # Create admin superuser and titansdb database
            print_status "Creating admin superuser..."
            sudo -u postgres createuser admin --superuser 2>/dev/null || print_warning "User admin may already exist"
            
            print_status "Creating titansdb database..."
            sudo -u postgres psql -c "CREATE DATABASE titansdb;" 2>/dev/null || print_warning "Database titansdb may already exist"
            
            print_success "Database setup completed"
        else
            print_warning "Could not connect to PostgreSQL after $max_attempts attempts."
            print_warning "Please check PostgreSQL installation and try starting it manually:"
            print_warning "sudo -u postgres /usr/lib/postgresql/*/bin/pg_ctl -D /var/lib/postgresql/data start"
        fi
    else
        print_warning "PostgreSQL not found. Please install and configure it manually."
    fi
}

# Function to create .env file if it doesn't exist
create_env_file() {
    if [ ! -f .env ]; then
        print_status "Creating .env file with default values..."
        cat > .env << 'EOF'
# Required - OpenAI API Key for embeddings and LLM inference
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DB_NAME=titans_db
DB_USER=titans_user
DB_PASSWORD=titans_password
DB_HOST=localhost
DB_PORT=5432

# Optional Model Paths
GGUF_MODEL_PATH=./models/your-model.gguf
MODELS_DIR=./models
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=text-embedding-ada-002

# Optional API Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_OPENAI_MODEL=gpt-3.5-turbo

# Optional Hugging Face Configuration (for private/gated models)
HUGGINGFACE_API_KEY=your_huggingface_token_here
EOF
        print_success "Created .env file with default values"
        print_warning "Please edit .env file and add your API keys!"
    else
        print_success ".env file already exists"
    fi
}

# Function to check Python version and find correct executable
find_python_executable() {
    local required_version="3.10.16"
    local required_major_minor="3.10"
    
    # List of possible Python executables to check
    local python_candidates=(
        "python3.10.16"
        "python3.10"
        "python3"
        "python"
    )
    
    for python_cmd in "${python_candidates[@]}"; do
        if command_exists "$python_cmd"; then
            local version=$($python_cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
            
            # Check if this is exactly the version we want
            if [[ "$version" == "$required_version" ]]; then
                echo "$python_cmd"
                return 0
            fi
            
            # Check if this is at least the right major.minor version
            local major_minor=$(echo "$version" | grep -oE '^[0-9]+\.[0-9]+')
            if [[ "$major_minor" == "$required_major_minor" ]]; then
                echo "$python_cmd"
                return 0
            fi
        fi
    done
    
    return 1
}

# Function to install Python 3.10.16 if not available
install_python_3_10() {
    # Install sudo first if needed
    install_sudo_if_needed
    
    local os=$(detect_os)
    print_status "Installing Python 3.10.16..."
    
    case $os in
        "linux")
            if command_exists apt-get; then
                print_status "Installing Python 3.10 via apt-get..."
                sudo apt-get update
                sudo apt-get install -y software-properties-common
                sudo add-apt-repository ppa:deadsnakes/ppa -y
                sudo apt-get update
                sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
                
                # Install pip for Python 3.10
                if ! command_exists python3.10; then
                    print_error "Failed to install Python 3.10"
                    return 1
                fi
                
                # Install pip if not available
                if ! python3.10 -m pip --version >/dev/null 2>&1; then
                    print_status "Installing pip for Python 3.10..."
                    curl https://bootstrap.pypa.io/get-pip.py | python3.10
                fi
                
            elif command_exists yum; then
                print_status "Installing Python 3.10 via yum..."
                sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel
                
                # Build from source for CentOS/RHEL
                cd /tmp
                wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tgz
                tar xzf Python-3.10.16.tgz
                cd Python-3.10.16
                ./configure --enable-optimizations
                make altinstall
                cd -
                rm -rf /tmp/Python-3.10.16*
            fi
            ;;
        "macos")
            if command_exists brew; then
                print_status "Installing Python 3.10 via Homebrew..."
                brew install python@3.10
            else
                print_error "Homebrew not found. Please install Python 3.10.16 manually."
                print_error "You can download it from: https://www.python.org/downloads/release/python-31016/"
                return 1
            fi
            ;;
        "windows")
            print_error "Please install Python 3.10.16 manually from:"
            print_error "https://www.python.org/downloads/release/python-31016/"
            return 1
            ;;
        *)
            print_error "Please install Python 3.10.16 manually for your system."
            return 1
            ;;
    esac
}

# Function to install Rust (required for some Python packages like deepfilternet)
install_rust() {
    print_status "Checking for Rust installation..."
    
    # Check if Rust is already available
    if command_exists rustc && command_exists cargo; then
        local rust_version=$(rustc --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        print_success "Rust is already installed (version $rust_version)"
        return 0
    fi
    
    # Check if cargo env exists but not in PATH
    if [ -f "$HOME/.cargo/env" ]; then
        print_status "Found Rust installation, adding to PATH..."
        source "$HOME/.cargo/env"
        if command_exists rustc && command_exists cargo; then
            local rust_version=$(rustc --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
            print_success "Rust is now available (version $rust_version)"
            return 0
        fi
    fi
    
    print_status "Installing Rust (required for deepfilternet)..."
    
    local os=$(detect_os)
    case $os in
        "linux"|"macos")
            # Use rustup installer
            print_status "Downloading and installing Rust via rustup..."
            if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; then
                print_success "Rust installer completed"
                
                # Source the cargo environment
                if [ -f "$HOME/.cargo/env" ]; then
                    source "$HOME/.cargo/env"
                    
                    # Add to current shell PATH
                    export PATH="$HOME/.cargo/bin:$PATH"
                    
                    # Verify installation
                    if command_exists rustc && command_exists cargo; then
                        local rust_version=$(rustc --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
                        print_success "Rust installation successful (version $rust_version)"
                        return 0
                    else
                        print_warning "Rust installed but not immediately available. You may need to restart your terminal."
                        print_warning "Continuing with setup - deepfilternet installation may fail."
                        return 0
                    fi
                else
                    print_warning "Rust installation completed but cargo environment not found."
                    print_warning "Continuing with setup - deepfilternet installation may fail."
                    return 0
                fi
            else
                print_warning "Rust installation failed. Continuing with setup..."
                print_warning "deepfilternet installation may fail. You can install Rust manually later."
                return 0
            fi
            ;;
        "windows")
            print_warning "Windows detected. Please install Rust manually:"
            print_warning "1. Download from: https://rustup.rs/"
            print_warning "2. Run the installer and follow the instructions"
            print_warning "3. Restart your terminal"
            print_warning "Continuing with setup - deepfilternet installation may fail."
            return 0
            ;;
        *)
            print_warning "Unknown OS. Please install Rust manually:"
            print_warning "Visit: https://rustup.rs/"
            print_warning "Continuing with setup - deepfilternet installation may fail."
            return 0
            ;;
    esac
}

# Function to setup virtual environment
setup_venv() {
    local venv_name="venv"
    local python_exec="$1"  # Accept python executable as parameter
    
    if [ -z "$python_exec" ]; then
        print_error "No Python executable provided to setup_venv"
        exit 1
    fi
    
    local version=$($python_exec --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "Using Python executable: $python_exec (version $version)"
    
    if [ -d "$venv_name" ]; then
        print_success "Virtual environment already exists"
        # Check if existing venv uses correct Python version
        if [ -f "$venv_name/pyvenv.cfg" ]; then
            local venv_version=$(grep "version" "$venv_name/pyvenv.cfg" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
            if [[ "$venv_version" =~ ^3\.10\. ]]; then
                print_success "Existing virtual environment uses Python $venv_version"
            else
                print_warning "Existing virtual environment uses Python $venv_version, but we need 3.10.x"
                print_status "Recreating virtual environment with correct Python version..."
                rm -rf "$venv_name"
                $python_exec -m venv "$venv_name"
                print_success "Virtual environment recreated with Python 3.10"
            fi
        fi
    else
        print_status "Creating virtual environment with Python 3.10..."
        $python_exec -m venv "$venv_name"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment for this script
    print_status "Activating virtual environment for setup..."
    source "$venv_name/bin/activate"
    
    # Verify the Python version in the virtual environment
    local venv_python_version=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "Virtual environment Python version: $venv_python_version"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Virtual environment activated for setup"
}

# Function to install Python dependencies
install_python_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install core dependencies first (numpy, scipy) to avoid build issues
    print_status "Installing core dependencies (numpy, scipy)..."
    pip install numpy scipy
    
    # Install requirements
    if [ -f requirements.txt ]; then
        print_status "Installing remaining dependencies..."
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install spaCy model
    print_status "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
    print_success "spaCy model downloaded"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "storage"
        "storage/generated_audio"
        "models"
        "chroma_db"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "All directories created"
}

# Function to create activation helper script
create_activation_script() {
    print_status "Creating activation helper script..."
    
    cat > activate_titans.sh << 'EOF'
#!/bin/bash
# Titans API Environment Activation Script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create it with your API keys."
fi

echo "✅ Titans API environment activated!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Virtual environment: activated"
echo ""
echo "To start the API server, run:"
echo "  python app.py"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
EOF

    chmod +x activate_titans.sh
    print_success "Created activate_titans.sh helper script"
}

# Function to test the setup
test_setup() {
    print_status "Testing the setup..."
    
    # Test Python imports
    python3 -c "
import sys
sys.path.append('./libraries')

try:
    # Test core imports
    from libraries.knowledgebase.preprocess import process_documents_for_collection
    from libraries.knowledgebase.retrieval import query_collection
    from libraries.llm.inference import generate_styled_text
    from libraries.stt.transcription import listen_and_transcribe
    from libraries.tts.inference import generate_audio
    print('✓ All library imports successful')
    
    # Test Flask imports
    from flask import Flask
    from flask_cors import CORS
    print('✓ Flask imports successful')
    
    # Test database imports
    import psycopg2
    print('✓ PostgreSQL imports successful')
    
    # Test ML libraries
    import torch
    import numpy as np
    import spacy
    print('✓ ML library imports successful')
    
    # Test spaCy model
    nlp = spacy.load('en_core_web_sm')
    print('✓ spaCy model loaded successfully')
    
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Setup test passed!"
    else
        print_error "Setup test failed!"
        exit 1
    fi
}

# Function to display final instructions
display_final_instructions() {
    print_success "Setup completed successfully!"
    echo
    print_status "Next steps:"
    echo "1. Edit the .env file and add your API keys:"
    echo "   - OPENAI_API_KEY (required for embeddings and LLM inference)"
    echo "   - HUGGINGFACE_API_KEY (optional, for private/gated models)"
    echo
    echo "2. To activate the environment and start working:"
    echo "   ${GREEN}source ./activate_titans.sh${NC}"
    echo "   ${GREEN}# OR manually:${NC}"
    echo "   ${GREEN}source venv/bin/activate${NC}"
    echo
    echo "3. To start the API server:"
    echo "   ${GREEN}python app.py${NC}"
    echo
    echo "4. The API will be available at: ${BLUE}http://localhost:5000${NC}"
    echo
    print_status "Convenience commands:"
    echo "   ${GREEN}./activate_titans.sh${NC}  - Activate environment with helpful info"
    echo "   ${GREEN}source venv/bin/activate${NC}  - Just activate the virtual environment"
    echo "   ${GREEN}deactivate${NC}  - Deactivate the virtual environment"
    echo
    print_status "Optional: Download models for local inference:"
    echo "   - GGUF models: Place in ./models/ directory"
    echo "   - Use the /download-model endpoint to download Hugging Face models"
    echo
    print_status "Database connection details:"
    echo "   - Database: titans_db"
    echo "   - User: titans_user"
    echo "   - Password: titans_password"
    echo "   - Host: localhost"
    echo "   - Port: 5432"
    echo
    print_warning "Make sure PostgreSQL is running before starting the API!"
    echo
    print_success "🚀 Ready to launch! Run: ${GREEN}source ./activate_titans.sh${NC}"
}

# Main execution
main() {
    print_status "Starting Titans API setup..."
    echo
    
    # Check if Python 3.10.x is available, install if not
    print_status "Checking for Python 3.10.16..."
    local python_exec
    if python_exec=$(find_python_executable); then
        local version=$($python_exec --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        print_success "Found compatible Python: $python_exec (version $version)"
    else
        print_warning "Python 3.10.x not found on system."
        print_status "Installing Python 3.10.16 automatically..."
        
        if install_python_3_10; then
            # Try to find Python again after installation
            if python_exec=$(find_python_executable); then
                local version=$($python_exec --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
                print_success "Python 3.10 installed successfully: $python_exec (version $version)"
            else
                print_error "Failed to find Python 3.10 after installation. Please install Python 3.10.16 manually."
                print_error "Download from: https://www.python.org/downloads/release/python-31016/"
                exit 1
            fi
        else
            print_error "Failed to install Python 3.10 automatically."
            print_error "Please install Python 3.10.16 manually and run this script again."
            print_error "Download from: https://www.python.org/downloads/release/python-31016/"
            exit 1
        fi
    fi
    
    # Install system dependencies
    read -p "Install system dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_system_dependencies
    else
        print_warning "Skipping system dependencies installation"
    fi
    
    # Setup PostgreSQL
    read -p "Setup PostgreSQL database? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_postgresql
    else
        print_warning "Skipping PostgreSQL setup"
    fi
    
    # Create .env file
    create_env_file
    
    # Install Rust (required for deepfilternet)
    install_rust
    
    # Setup virtual environment (Python 3.10 is now guaranteed to be available)
    setup_venv "$python_exec"
    
    # Install Python dependencies
    install_python_dependencies
    
    # Create directories
    create_directories
    
    # Create activation helper script
    create_activation_script
    
    # Test setup
    test_setup
    
    # Display final instructions
    display_final_instructions
}

# Run main function
main "$@"
