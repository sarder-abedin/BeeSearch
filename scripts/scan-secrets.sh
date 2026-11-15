#!/usr/bin/env bash
# scripts/scan-secrets.sh — fail if a real credential is committed.
#
# WHY THIS EXISTS
# ───────────────
# Four live Langfuse API keys were committed to this public repository and sat
# in its git history. They were removed from the working tree, but removal does
# not un-publish them — they had to be revoked. This guard blocks the next one
# before it ever reaches a commit.
#
# Scans tracked files only, so it never trips on a developer's real .env
# (untracked and gitignored).
#
# Placeholders are allowed on purpose: the docs and .env.example need to show
# the SHAPE of a key. So every pattern below requires a real key's structure
# (a UUID, or a long random run) which "pk-lf-your-key-here", "sk-lf-...",
# "change_me" and "<your-token>" cannot satisfy.
#
# Usage:
#   ./scripts/scan-secrets.sh          # scan tracked files
#   ./scripts/scan-secrets.sh --self-test   # prove the patterns actually fire

set -uo pipefail

UUID='[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'

# name|regex — one per credential family.
PATTERNS=(
  "Langfuse key|(pk|sk)-lf-${UUID}"
  "OpenAI/Anthropic key|sk-(ant-)?[A-Za-z0-9_-]{32,}"
  "AWS access key id|AKIA[0-9A-Z]{16}"
  "Google API key|AIza[0-9A-Za-z_-]{35}"
  "GitHub token|(gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})"
  "Slack token|xox[baprs]-[A-Za-z0-9-]{12,}"
  "Private key block|-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----"
)

scan_tracked() {
  local found=0 name regex hits
  for entry in "${PATTERNS[@]}"; do
    name="${entry%%|*}"
    regex="${entry#*|}"
    # -I skips binaries; tracked files only, so an untracked .env is never read.
    # -e is required: the private-key pattern starts with "-----", which grep
    # would otherwise parse as options (the self-test caught exactly that).
    if hits=$(git grep -nIE -e "$regex" -- . 2>/dev/null); then
      echo "::error::Possible committed credential — $name"
      # Print file:line only. Never echo the match: CI logs are world-readable
      # on a public repo, and printing it would leak the secret a second time.
      echo "$hits" | cut -d: -f1,2 | sed 's/^/    /'
      found=1
    fi
  done
  return $found
}

self_test() {
  # A guard that cannot fail is not a guard. Verify each pattern matches a
  # realistic key and ignores the placeholder the docs use.
  local name regex fail=0
  # Samples are assembled from fragments at runtime so that no key-shaped
  # literal ever exists in this file. Two reasons: a literal would make this
  # scanner flag its own source the moment it became a tracked file, and a
  # sample copied from the real leak would re-commit a live credential.
  local D='-'
  declare -A SHOULD_MATCH=(
    ["Langfuse key"]="pk${D}lf${D}00000000${D}0000${D}4000${D}8000${D}000000000000"
    ["OpenAI/Anthropic key"]="sk${D}$(printf 'a%.0s' {1..40})"
    ["AWS access key id"]="AKIA$(printf 'A%.0s' {1..16})"
    ["Google API key"]="AIza$(printf 'B%.0s' {1..35})"
    ["GitHub token"]="ghp_$(printf 'c%.0s' {1..36})"
    ["Slack token"]="xox${D}placeholder" # replaced below; keeps no literal here
    ["Private key block"]="${D}${D}${D}${D}${D}BEGIN RSA PRIVATE KEY${D}${D}${D}${D}${D}"
  )
  SHOULD_MATCH["Slack token"]="xoxb${D}123456789012${D}abcdefghijkl"
  local PLACEHOLDERS=(
    "pk${D}lf${D}your${D}key${D}here" "sk${D}lf${D}your${D}key${D}here"
    "pk${D}lf${D}..." "sk${D}lf${D}..." 'change_me_to_a_32_char_random_string'
    '<your-token>' "sk${D}..." ''
  )

  for entry in "${PATTERNS[@]}"; do
    name="${entry%%|*}"; regex="${entry#*|}"
    if ! grep -qE -e "$regex" <<<"${SHOULD_MATCH[$name]}"; then
      echo "FAIL: '$name' did not match its own sample key"; fail=1
    fi
  done
  for p in "${PLACEHOLDERS[@]}"; do
    for entry in "${PATTERNS[@]}"; do
      regex="${entry#*|}"
      if grep -qE -e "$regex" <<<"$p"; then
        echo "FAIL: placeholder '$p' matched ${entry%%|*} — would block the docs"
        fail=1
      fi
    done
  done
  [[ $fail -eq 0 ]] && echo "self-test passed: every pattern fires on a real key and ignores placeholders"
  return $fail
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test; exit $?
fi

if scan_tracked; then
  echo "No committed credentials found in tracked files."
  exit 0
fi

cat <<'EOF'

A credential appears to be committed. Do NOT just delete the line and commit:
once pushed, it is public and must be treated as compromised.
  1. Revoke / rotate the key with the provider now.
  2. Move the value into .env (untracked, gitignored) and leave only a
     placeholder in .env.example.
EOF
exit 1
