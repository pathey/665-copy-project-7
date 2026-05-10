#!/usr/bin/env bash
set -euo pipefail

TEMPLATE="./config/cfg/phase3_case_study_daisy_template.yaml"
TMP_DIR="./config/generated"
mkdir -p "$TMP_DIR"

SEED="${1:-7447}"
JOBS="${2:-8}"

commands_file="${TMP_DIR}/commands.txt"
: > "$commands_file"

distance="1000"
rates=(2 3 4)
mcs_modes=(0 1)
increment_thresholds=(0.7 0.9 1.1)
decrement_thresholds=(1.1 1.3 1.5)

enable_adaptive_mcs="false"

for mcs in "${mcs_modes[@]}"; do
  for rate in "${rates[@]}"; do
	run_name="phase3_daisy_${distance}m_mcs${mcs}_${rate}Mbps"
	out_cfg="${TMP_DIR}/${run_name}.yaml"

	sed \
	  -e "s/__RUN_NAME__/${run_name}/g" \
	  -e "s/__DISTANCE__/${distance}/g" \
	  -e "s/__RATE__/${rate}/g" \
	  -e "s/__ADAPTIVE_MCS__/${enable_adaptive_mcs}/g" \
	  -e "s/__MCS_INDEX__/${mcs}/g" \
	  -e "s/__increment_threshold__/0.7/g"\
	  -e "s/__decrement_threshold__/0.85/g"\
	  "$TEMPLATE" > "$out_cfg"

	echo "Running $run_name"
	echo "python main.py --no-input --config '$out_cfg' '$SEED'" >> "$commands_file"
  done
done

enable_adaptive_mcs="true"
for dec_threshold in "${decrement_thresholds[@]}"; do
  for inc_threshold in "${increment_thresholds[@]}"; do
    for rate in "${rates[@]}"; do

      run_name="phase3_daisy_${distance}m_mcs_adaptive_${rate}Mbps_${inc_threshold}inc_${dec_threshold}dec"
      out_cfg="${TMP_DIR}/${run_name}.yaml"
      
      mcs_index="2"

      sed \
        -e "s/__RUN_NAME__/${run_name}/g" \
        -e "s/__DISTANCE__/${distance}/g" \
        -e "s/__RATE__/${rate}/g" \
        -e "s/__ADAPTIVE_MCS__/${enable_adaptive_mcs}/g" \
        -e "s/__MCS_INDEX__/${mcs_index}/g" \
	-e "s/__increment_threshold__/${inc_threshold}/g"\
	-e "s/__decrement_threshold__/${dec_threshold}/g"\
        "$TEMPLATE" > "$out_cfg"

      echo "Running $run_name"
      echo "python main.py --no-input --config '$out_cfg' '$SEED'" >> "$commands_file"

    done
  done
done

echo "Running jobs with parallelism = $JOBS"
cat "$commands_file" | xargs -I CMD -P "$JOBS" bash -c CMD
