SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_PATH}" || exit

SHARE_DIR="$HOME/.local/share/vqtools"
BIN_DIR="$HOME/.local/bin"

echo "Installing VQTools"

create_wrapper_script() {
    local tool_name="$1"
    local script_name="${tool_name%.py}"  # Remove .py
    
    cat << EOF
#!/bin/bash

WORKING_PATH="\$(pwd)"
SHARE_DIR="$HOME/.local/share/vqtools"

cd "\$SHARE_DIR"
source vqenv/bin/activate
export PYTHONPATH="\$SHARE_DIR:\$PYTHONPATH"

cd "\$WORKING_PATH"
python "\$SHARE_DIR/tools/$tool_name" "\$@"
EOF
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
if [ ! -d "vqenv" ]; then
    echo "# Creating virtual environment and installing requirements"
    python3 -m venv vqenv
    source vqenv/bin/activate
    pip install -r requirements.txt
else
    echo "# Virtual environment already exists, activating"
    source vqenv/bin/activate
    pip install -r requirements.txt
fi

TOOLS=($(ls tools/*.py 2>/dev/null))

echo "# Creating wrapper scripts"
for tool_path in "${TOOLS[@]}"; do
    tool_name=$(basename "$tool_path")
    script_name="${tool_name%.py}"
    wrapper_path="${BIN_DIR}/${script_name}"
    
    echo "# Creating wrapper for ${script_name}"
    create_wrapper_script "$tool_name" > "$wrapper_path"
    chmod +x "$wrapper_path"
done

echo "# Installation complete!"
echo "Available tools:"
for tool_path in "${TOOLS[@]}"; do
    tool_name=$(basename "$tool_path")
    script_name="${tool_name%.py}"
    echo "  - ${script_name}"
done


