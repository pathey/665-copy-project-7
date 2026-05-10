#!/usr/bin/env bash
set -euo pipefail

TEMPLATE="./config/cfg/phase2_daisy_template.yaml"
TMP_DIR="./config/generated"
mkdir -p "$TMP_DIR"

SEED="${1:-7447}"


	
	run_name="phase2_crossflow_500m_mcs0_1Mbps"
	out_cfg="${TMP_DIR}/${run_name}.yaml"
	
	increment_threshold=0.7
	decrement_threshold=0.85
	enable_mcs_snr="false"
        mcs_index=0
        run_name="phase2_crossflow_500m_mcs0_1Mbps"
        out_cfg="${TMP_DIR}/${run_name}.yaml"

      sed \
        -e "s/__RUN_NAME__/${run_name}/g" \
        -e "s/__DISTANCE__/500/g" \
        -e "s/__RATE__/1.0/g" \
        -e "s/__ENABLE_MCS_SNR__/${enable_mcs_snr}/g" \
        -e "s/__MCS_INDEX__/${mcs_index}/g" \
        -e "s/__increment_threshold__/${increment_threshold}/g" \
        -e "s/__decrement_threshold__/${decrement_threshold}/g" \

        "$TEMPLATE" > "$out_cfg"

      echo "Running $run_name"
      python main.py --no-input --config "$out_cfg" "$SEED"

