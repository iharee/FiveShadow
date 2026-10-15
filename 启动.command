#!/bin/zsh
set -e
cd -- "${0:A:h}"
STUBBLE_UV_BIN="$(command -v uv || true)"
if [[ -z "$STUBBLE_UV_BIN" && -x /opt/homebrew/bin/uv ]]; then
  STUBBLE_UV_BIN=/opt/homebrew/bin/uv
fi
if [[ -z "$STUBBLE_UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  STUBBLE_UV_BIN="$HOME/.local/bin/uv"
fi
if [[ -z "$STUBBLE_UV_BIN" ]]; then
  print "未找到 uv。请安装 uv 后重新双击启动。"
  read "?按回车关闭…"
  exit 1
fi
exec "$STUBBLE_UV_BIN" run --locked stubble gui "$@"
