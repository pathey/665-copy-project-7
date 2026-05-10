#!/usr/bin/env bash
set -euo pipefail

TEMPLATE="./config/cfg/phase2_daisy_template.yaml"
TMP_DIR="./config/generated"
mkdir -p "$TMP_DIR"

SEED="${1:-7447}"

distances=(500 1000 1200)
rates=(1 2 3)
mcs_modes=(0 1 "adaptive")

for distance in "${distances[@]}"; do
  for mcs in "${mcs_modes[@]}"; do
    for rate in "${rates[@]}"; do

      run_name="phase2_crossflow_${distance}m_mcs_adaptive_${rate}Mbps"
      out_cfg="${TMP_DIR}/${run_name}.yaml"

      if [[ "$mcs" == "adaptive" ]]; then
        enable_mcs_snr="true"
        mcs_index="2"
      else
        enable_mcs_snr="false"
        mcs_index="$mcs"
        run_name="phase2_crossflow_${distance}m_mcs${mcs}_${rate}Mbps"
        out_cfg="${TMP_DIR}/${run_name}.yaml"
      fi

      sed \
        -e "s/__RUN_NAME__/${run_name}/g" \
        -e "s/__DISTANCE__/${distance}/g" \
        -e "s/__RATE__/${rate}/g" \
        -e "s/__ENABLE_MCS_SNR__/${enable_mcs_snr}/g" \
        -e "s/__MCS_INDEX__/${mcs_index}/g" \
        "$TEMPLATE" > "$out_cfg"

      echo "Running $run_name"
      python main.py --no-input --config "$out_cfg" "$SEED"

    done
  done
done
