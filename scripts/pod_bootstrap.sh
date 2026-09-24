#!/usr/bin/env bash
# Bootstrap a fresh RunPod container. /workspace persists across restarts; the
# container filesystem (incl. $HOME) does not. Run on every pod start:
#
#   bash /workspace/driftenginelite/scripts/pod_bootstrap.sh
#
set -uo pipefail

apt-get update -qq && apt-get install -y -qq tmux git curl >/dev/null

mkdir -p /workspace/hf /workspace/.claude

# Env in .bashrc, once — the bare append duplicated these lines on every boot.
grep -q 'HF_HOME=/workspace/hf' ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'RC'
export HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"
RC
export HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"

# Claude state on the persistent volume. ~/.claude was already handled; the CLI
# also keeps state in ~/.claude.json, which was being reset every boot.
if [ ! -L ~/.claude ]; then
  [ -d ~/.claude ] && cp -a ~/.claude/. /workspace/.claude/ 2>/dev/null
  rm -rf ~/.claude
  ln -s /workspace/.claude ~/.claude
fi
if [ ! -L ~/.claude.json ]; then
  [ -f ~/.claude.json ] && mv ~/.claude.json /workspace/.claude.json
  [ -f /workspace/.claude.json ] || echo '{}' > /workspace/.claude.json
  ln -sf /workspace/.claude.json ~/.claude.json
fi

# Git: identity, ownership waiver, and a credential store that SURVIVES
# container resets -- work has been stranded unpushable on this pod repeatedly.
git config --global --add safe.directory /workspace/driftenginelite
git config --global credential.helper "store --file /workspace/.git-credentials"
git config --global user.name  >/dev/null 2>&1 || git config --global user.name  "isotonic962"
git config --global user.email >/dev/null 2>&1 || git config --global user.email "niklaserikgranberg@gmail.com"

# Claude install persisted on /workspace: download once ever, restore by
# symlink on every later boot ($HOME dies with the container; /workspace not).
mkdir -p ~/.local/bin ~/.local/share
if ! command -v claude >/dev/null 2>&1; then
  if [ -e /workspace/claude-install/share ]; then
    ln -sfn /workspace/claude-install/share ~/.local/share/claude
    cp -aP /workspace/claude-install/bin-claude ~/.local/bin/claude
  else
    curl -fsSL https://claude.ai/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
    # move the freshly installed tree onto the volume and link back to it
    if [ -d ~/.local/share/claude ] && [ -e ~/.local/bin/claude ]; then
      mkdir -p /workspace/claude-install
      cp -aP ~/.local/bin/claude /workspace/claude-install/bin-claude
      mv ~/.local/share/claude /workspace/claude-install/share
      ln -sfn /workspace/claude-install/share ~/.local/share/claude
    fi
  fi
fi
export PATH="$HOME/.local/bin:$PATH"

echo "── status ──────────────────────────────────────"
claude --version 2>/dev/null || echo "claude: INSTALL FAILED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "gpu: none visible"
if GIT_TERMINAL_PROMPT=0 git -C /workspace/driftenginelite push --dry-run origin HEAD >/dev/null 2>&1; then
  echo "git push: OK (token in /workspace/.git-credentials, survives restarts)"
else
  echo "git push: NOT SET UP -- one time only:"
  echo "  cd /workspace/driftenginelite && git push --dry-run origin main"
  echo "  (paste your GitHub token as the password; it persists from then on)"
fi
