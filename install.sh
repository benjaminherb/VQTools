SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_PATH}" || exit

SHARE_DIR="$HOME/.local/share/vqtools"
BIN_DIR="$HOME/.local/bin"

echo "Installing VQTools"

PYTHON_BIN=""
COMPAT_MODE=0
REBUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --compatibility)
            COMPAT_MODE=1
            shift
            ;;
        --rebuild)
            REBUILD=1
            shift
            ;;
        --) # end of options
            shift
            break
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$COMPAT_MODE" -eq 1 ]; then
    PYTHON_BIN="python3.10"
else
    PYTHON_BIN="python3.12"
fi

echo "Using python interpreter: $PYTHON_BIN"
if [ "$COMPAT_MODE" -eq 1 ]; then
    echo "Compatibility mode enabled: using a lower Python (if available)"
fi
if [ "$REBUILD" -eq 1 ]; then
    echo "Rebuild requested: existing virtual environment (if any) will be recreated"
fi

create_wrapper_script() {
    local tool_name="$1"
    local script_name="${tool_name%.py}"  # Remove .py
    # This works for both venv and conda envs created with -p (they both have a bin/python).
    printf '%s
' "#!/bin/bash" "" "WORKING_PATH=\"\$(pwd)\"" "SHARE_DIR=\"$HOME/.local/share/vqtools\"" "" "PYTHON_BIN=\"\$SHARE_DIR/vqenv/bin/python\"" "" "cd \"\$WORKING_PATH\"" "export PYTHONPATH=\"\$SHARE_DIR:\$PYTHONPATH\"" "" "\"\$PYTHON_BIN\" \"\$SHARE_DIR/tools/$tool_name\" \"\$@\"" > "$BIN_DIR/$script_name"
    chmod +x "$BIN_DIR/$script_name"
}

echo "# Creating directories"
mkdir -p "${SHARE_DIR}"
mkdir -p "${BIN_DIR}"

echo "# Copying tools, metrics, and requirements"
cp -r ./tools "${SHARE_DIR}/"
cp -r ./metrics "${SHARE_DIR}/"
cp ./requirements.txt "${SHARE_DIR}/"

cd "$SHARE_DIR"

# Check if venv exists
if [ "$REBUILD" -eq 1 ] && [ -d "vqenv" ]; then
    echo "Rebuild requested: removing existing venv 'vqenv'"
    rm -rf vqenv
fi

if [ ! -d "vqenv" ]; then
    echo "# Creating environment and installing requirements"
    if [ "$COMPAT_MODE" -eq 1 ]; then
        if command -v conda >/dev/null 2>&1; then
            echo "Creating conda environment at ./vqenv with python=3.10"
            conda create -y -p vqenv python=3.10
            conda env activate ./vqenv
            pip install -r requirements.txt
            pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu117
            pip install numpy opencv-python lpips pyiqa
        else
            echo "Compatibility mode requested but 'conda' was not found on PATH. Please install conda (Miniconda/Anaconda) or run without --compatibility."
            exit 1
        fi
    else
        "$PYTHON_BIN" -m venv vqenv || exit
        source vqenv/bin/activate
        pip install -r requirements.txt
    fi
else
    echo "# Virtual environment already exists, activating"
    if [ -f "vqenv/bin/activate" ]; then
        source vqenv/bin/activate
    fi
    pip install -r requirements.txt
fi


if ! pip show decord &> /dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Building decord..."
        cd "${SHARE_DIR}"
        git clone --recursive https://github.com/dmlc/decord 
        cd decord
        export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:$PKG_CONFIG_PATH"
        echo "Using MacOS specific fix"
        git fetch origin pull/353/head:mac
        git checkout mac
        sed -i '' '/\/\/set(CMAKE_CUDA_FLAGS/d' CMakeLists.txt
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_AUDIO=OFF -DCMAKE_CXX_STANDARD=11
        make
        cd ../python
        python setup.py install
        
        BUILD_LIB="${SHARE_DIR}/decord/build/libdecord.dylib"
        PKG_DIR=$(python -c "import decord; import os; print(os.path.dirname(decord.__file__))" 2>/dev/null)
        cp "$BUILD_LIB" "$PKG_DIR/"
        cp ${SHARE_DIR}/vqenv/decord/libdecord.dylib "${SHARE_DIR}/vqenv/lib/python3.12/site-packages/decord/"
    else
        echo "Installing decord via pip"
        pip install decord
    fi
fi

cd "$SHARE_DIR"

TOOLS=($(ls tools/*.py 2>/dev/null))

echo "# Creating wrapper scripts"
for tool_path in "${TOOLS[@]}"; do
    tool_name=$(basename "$tool_path")
    script_name="${tool_name%.py}"
    wrapper_path="${BIN_DIR}/${script_name}"
    
    echo "# Creating wrapper for ${script_name}"
    create_wrapper_script "$tool_name"
    chmod +x "$BIN_DIR/$script_name"
done

echo "# Installation complete!"
echo "Available tools:"
for tool_path in "${TOOLS[@]}"; do
    tool_name=$(basename "$tool_path")
    script_name="${tool_name%.py}"
    echo "  - ${script_name}"
done


