"""Make the ffmpeg from imageio-ffmpeg available as `ffmpeg` inside the virtual environment.

Used on Intel Macs (requirements-intel-mac.txt), where ffmpeg isn't installed with Homebrew.
Run it with the environment active: python setup_ffmpeg.py
"""

import sys
from pathlib import Path

if sys.prefix == sys.base_prefix:
    sys.exit("Activate the virtual environment first: source .venv/bin/activate")

try:
    import imageio_ffmpeg
except ImportError:
    sys.exit("imageio-ffmpeg is missing. Run: pip install -r requirements-intel-mac.txt")

source = Path(imageio_ffmpeg.get_ffmpeg_exe())
target = Path(sys.prefix) / "bin" / "ffmpeg"
if target.is_symlink() or target.exists():
    target.unlink()
target.symlink_to(source)
print(f"ffmpeg is ready: {target} -> {source}")
