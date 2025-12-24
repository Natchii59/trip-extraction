#!/usr/bin/env fish

# Complete installation script for Trip Parser Monorepo project
# Compatible with Fish shell

set RED (printf '\033[0;31m')
set GREEN (printf '\033[0;32m')
set YELLOW (printf '\033[1;33m')
set BLUE (printf '\033[0;34m')
set NC (printf '\033[0m') # No Color

printf "%s========================================%s\n" $BLUE $NC
printf "%s  Trip Parser Monorepo Installation  %s\n" $BLUE $NC
printf "%s========================================%s\n" $BLUE $NC
printf "\n"

# Functions to display messages
function print_info
    printf "%s[INFO]%s %s\n" $BLUE $NC "$argv"
end

function print_success
    printf "%s[✓]%s %s\n" $GREEN $NC "$argv"
end

function print_error
    printf "%s[✗]%s %s\n" $RED $NC "$argv"
end

function print_warning
    printf "%s[!]%s %s\n" $YELLOW $NC "$argv"
end

# Python verification
print_info "Checking Python..."
if not command -v python3 &> /dev/null
    print_error "Python 3 is not installed. Please install it first."
    exit 1
end

set PYTHON_VERSION (python3 --version | string match -r '\d+\.\d+')
print_success "Python $PYTHON_VERSION found"

# Node.js verification
print_info "Checking Node.js..."
if not command -v node &> /dev/null
    print_error "Node.js is not installed. Please install it first."
    exit 1
end

set NODE_VERSION (node --version)
print_success "Node.js $NODE_VERSION found"

# npm verification
print_info "Checking npm..."
if not command -v npm &> /dev/null
    print_error "npm is not installed. Please install it first."
    exit 1
end

set NPM_VERSION (npm --version)
print_success "npm v$NPM_VERSION found"

printf "\n"
print_info "Creating Python virtual environment..."

# Remove old virtual environment if it exists
if test -d .venv
    print_warning "Existing virtual environment detected. Removing..."
    rm -rf .venv
end

# Create new virtual environment
python3 -m venv .venv
if test $status -ne 0
    print_error "Failed to create virtual environment"
    exit 1
end
print_success "Virtual environment created"

printf "\n"
print_info "Activating virtual environment..."
source .venv/bin/activate.fish
print_success "Virtual environment activated"

printf "\n"
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

printf "\n"
print_info "Installing root Python dependencies..."
pip install -e ".[dev]"
if test $status -ne 0
    print_error "Failed to install Python dependencies"
    exit 1
end
print_success "Python dependencies installed"

printf "\n"
print_info "Installing Node.js dependencies..."
npm install
if test $status -ne 0
    print_error "Failed to install Node.js dependencies"
    exit 1
end
print_success "Node.js dependencies installed"

printf "\n"
print_info "Installing Python dependencies and monorepo packages..."
npm run install:all
if test $status -ne 0
    print_error "Failed to install monorepo packages"
    exit 1
end
print_success "All packages installed successfully"

printf "\n"
printf "%s========================================%s\n" $GREEN $NC
printf "%s  Installation completed successfully!  %s\n" $GREEN $NC
printf "%s========================================%s\n" $GREEN $NC
printf "\n"
printf "%s To activate the virtual environment:%s\n" $BLUE $NC
printf "  source .venv/bin/activate.fish\n"
printf "\n"
