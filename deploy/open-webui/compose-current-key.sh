#!/usr/bin/env bash
set +x
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compose_file="$script_dir/compose.yml"
provider_key="${OPENAI_API_KEY-}"
unset OPENAI_API_KEY

# This wrapper deliberately exposes only Compose operations that do not render or
# execute with the service environment. In particular, plain `config`, `run`, and
# `exec` could print the provider key.
case "${1-}" in
  up | down)
    ;;
  config)
    if test "$#" -ne 2 || test "${2-}" != '--quiet'; then
      echo 'Only `config --quiet` is allowed by this wrapper.' >&2
      exit 2
    fi
    ;;
  *)
    echo 'Allowed commands: up, down, config --quiet.' >&2
    exit 2
    ;;
esac

# Prefer an explicitly exported key so an external secret manager or key rotation
# can override retained container configuration.
if test -z "$provider_key"; then
  for source_container in open-webui; do
    if ! docker container inspect "$source_container" >/dev/null 2>&1; then
      continue
    fi

    mapfile -t container_keys < <(
      docker inspect "$source_container" \
        | jq -r '.[0].Config.Env[] | select(startswith("OPENAI_API_KEY=")) | sub("^OPENAI_API_KEY="; "")'
    )
    if test "${#container_keys[@]}" -ne 1 || test -z "${container_keys[0]}"; then
      echo "Expected exactly one non-empty OPENAI_API_KEY in $source_container" >&2
      exit 1
    fi
    provider_key="${container_keys[0]}"
    unset container_keys
    break
  done
fi

# A tightly permissioned file is the final fallback after retained containers are
# retired. Parse the one variable as data; never source executable shell content.
env_file="$script_dir/.env"
if test -z "$provider_key" && test -e "$env_file"; then
  if test -L "$env_file" || ! test -f "$env_file"; then
    echo '.env must be a regular, non-symlink file.' >&2
    exit 1
  fi
  if test "$(stat -c '%a' "$env_file")" != 600 || test "$(stat -c '%u' "$env_file")" != "$(id -u)"; then
    echo '.env must be owned by the invoking user with mode 0600.' >&2
    exit 1
  fi
  mapfile -t file_keys < <(awk 'index($0, "OPENAI_API_KEY=") == 1 { print substr($0, 16) }' "$env_file")
  if test "${#file_keys[@]}" -ne 1 || test -z "${file_keys[0]}"; then
    echo '.env must contain exactly one non-empty OPENAI_API_KEY assignment.' >&2
    exit 1
  fi
  provider_key="${file_keys[0]}"
  unset file_keys
fi

if test -z "$provider_key"; then
  echo 'No non-empty provider key was found in the environment, retained containers, or .env.' >&2
  exit 1
fi
if [[ "$provider_key" == 'replace-with-current'* ]]; then
  echo 'Refusing to use the placeholder provider key.' >&2
  exit 1
fi

export OPENAI_API_KEY="$provider_key"
unset provider_key source_container env_file
exec docker compose -f "$compose_file" "$@"
