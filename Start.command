#!/bin/bash
set -e
cd "$(dirname "$0")"

docker compose up -d --build
open "http://localhost:3000"
