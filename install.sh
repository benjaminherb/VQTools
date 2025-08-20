SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_PATH}" || exit

SHARE_DIR="$HOME/.local/share/vqtools"
BIN_DIR="$HOME/.local/bin"

echo "Installing VQTools"

echo "# Making script executable"
chmod +x ./vqcheck

echo "# Creating directory ${SHARE_DIR}"
mkdir -p "${SHARE_DIR}"

echo "# Copying files"
cp ./vqcheck "${SHARE_DIR}"
cp ./vqcheck.py "${SHARE_DIR}"
cp ./requirements.txt "${SHARE_DIR}"

cd "$SHARE_DIR"

rm -r ./vqenv

echo "# Creating symlink ${BIN_DIR}/vqcheck"
ln -sf "${SHARE_DIR}/vqcheck" "${BIN_DIR}/vqcheck"

vq


