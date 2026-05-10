#!/usr/bin/env bash
set -euo pipefail

TEMPLATE="./config/cfg/phase2_daisy_template.yaml"
TMP_DIR="./config/generated"
mkdir -p "$TMP_DIR"

SEED="${1:-7447}"
JOBS="${2:-4}"

commands_file="${TMP_DIR}/commands.txt"
: > "$commands_file"

distances=(500 1000 1200)
rates=(1 2 3)
mcs_modes=("adaptive")
increment_thresholds=(0.5 0.9)
decrement_thresholds=(0.75 0.95)


decrement_threshold="0.85"
for distance in "${distances[@]}"; do
  for threshold in "${increment_thresholds[@]}"; do
    for rate in "${rates[@]}"; do

      run_name="phase2_crossflow_${distance}m_mcs_adaptive_${rate}Mbps_${threshold}increment_threshold"
      out_cfg="${TMP_DIR}/${run_name}.yaml"

      enable_mcs_snr="true"
      mcs_index="2"

      sed \
        -e "s/__RUN_NAME__/${run_name}/g" \
        -e "s/__DISTANCE__/${distance}/g" \
        -e "s/__RATE__/${rate}/g" \
        -e "s/__ENABLE_MCS_SNR__/${enable_mcs_snr}/g" \
        -e "s/__MCS_INDEX__/${mcs_index}/g" \
	-e "s/__increment_threshold__/${threshold}/g"\
	-e "s/__decrement_threshold__/${decrement_threshold}/g"\
        "$TEMPLATE" > "$out_cfg"

      echo "Running $run_name"
      echo "python main.py --no-input --config '$out_cfg' '$SEED'" >> "$commands_file"

    done
  done
done

increment_threshold="0.7"
for distance in "${distances[@]}"; do
  for threshold in "${decrement_thresholds[@]}"; do
    for rate in "${rates[@]}"; do

      run_name="phase2_crossflow_${distance}m_mcs_adaptive_${rate}Mbps_${threshold}decrement_threshold"
      out_cfg="${TMP_DIR}/${run_name}.yaml"

      enable_mcs_snr="true"
      mcs_index="2"

      sed \
        -e "s/__RUN_NAME__/${run_name}/g" \
        -e "s/__DISTANCE__/${distance}/g" \
        -e "s/__RATE__/${rate}/g" \
        -e "s/__ENABLE_MCS_SNR__/${enable_mcs_snr}/g" \
        -e "s/__MCS_INDEX__/${mcs_index}/g" \
	-e "s/__increment_threshold__/${increment_threshold}/g"\
	-e "s/__decrement_threshold__/${threshold}/g"\
        "$TEMPLATE" > "$out_cfg"

      echo "Running $run_name"
      echo "python main.py --no-input --config '$out_cfg' '$SEED'" >> "$commands_file"

    done
  done
done

echo "Running jobs with parallelism = $JOBS"
cat "$commands_file" | xargs -I CMD -P "$JOBS" bash -c CMD
