#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_docker.sh <cpu|gpu> [quick|full] [extra args...]"
  echo "Examples:"
  echo "  ./run_docker.sh cpu"
  echo "  ./run_docker.sh cpu full"
  echo "  ./run_docker.sh gpu quick --layers 0,1 --seeds 42"
  exit 1
fi

TARGET="$1"
shift

MODE="quick"
if [[ $# -gt 0 ]]; then
  case "$1" in
    quick|full)
      MODE="$1"
      shift
      ;;
  esac
fi

EXTRA_ARGS=("$@")

case "$TARGET" in
  cpu)
    docker compose run --rm visionsae-cpu --mode "$MODE" "${EXTRA_ARGS[@]}"
    ;;
  gpu)
    docker compose --profile gpu run --rm visionsae-gpu --mode "$MODE" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Invalid target: $TARGET"
    echo "Expected cpu or gpu"
    exit 1
    ;;
esac
