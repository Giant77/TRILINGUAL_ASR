#!/usr/bin/env bash

total_seconds=0

while IFS= read -r -d '' file; do
  duration=$("/c/Users/y/AppData/Local/Microsoft/WinGet/Links/ffprobe.exe" \
    -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$file")

  duration=${duration%.*}  # remove decimals
  total_seconds=$((total_seconds + duration))

done < <(find "./Dataset/id/Librivox/audio_train/librivox-indonesia/train/indonesian" \
          -type f \( -iname "*.mp3" -o -iname "*.wav" -o -iname "*.m4a" -o -iname "*.flac" \) \
          -print0)

hours=$(awk "BEGIN {printf \"%.2f\", $total_seconds/3600}")
echo "Total hours: $hours"

# fleurs-id = 12.12 jam
# - dev: 1.11 ~9.16%
# - test: 2.27 ~18.73%
# - train: 8.74 ~72.11%

# fleurs-en = 9.81
# - dev: 0.99 ~10.09%
# - test: 1.68 ~17.13%
# - train: 7.14 ~72.78%

# fleurs-ar = 9.93
# - dev: 0.84 ~8.46%
# - test: 1.24 ~12.49
# - train: 7.85 ~79.05

