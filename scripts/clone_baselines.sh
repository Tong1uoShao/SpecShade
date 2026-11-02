#!/bin/bash
# Shallow-clone the official baseline repositories into src/models/baseline/vendor/ (read-only).
cd "$(dirname "$0")/../src/models/baseline/vendor" || exit 1
for r in bsmhmmlf/Gaussian-Shading SunnierLee/GaussMarker showlab/RingID Mao718/MaXsive thomas11809/SFWMark; do
  n=$(basename "$r"); [ -d "$n" ] || git clone --depth 1 "https://github.com/$r.git" "$n"
done
ls
