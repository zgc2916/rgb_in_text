#!/bin/bash
# Generate RGBT-VGNet visualizations for all benchmark datasets.
# Usage: bash script_visualize/all/run_all_rgbt_vgnet_tests.sh

echo "Starting RGBT-VGNet visualization..."
echo "Test scope:"
echo "  - Datasets: rgbtvg_flir, rgbtvg_m3fd, rgbtvg_mfad"
echo "  - Modalities: rgbt"
echo "  - Total: 3 combinations"
echo "========================================"

# Define datasets and modalities
DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgbt")

# Model base path
MODEL_BASE_PATH="../dataset_and_pretrain_model/result/MMVG"

# Counter
TOTAL_TESTS=$(( ${#DATASETS[@]} * ${#MODALITIES[@]} ))
CURRENT_TEST=0
SUCCESS_COUNT=0
FAILED_TESTS=()

# Start time
START_TIME=$(date +%s)

# Iterate through all combinations
for dataset in "${DATASETS[@]}"; do
    for modality in "${MODALITIES[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))

        # Extract short name from dataset name (rgbtvg_flir -> flir)
        DATASET_SHORT=$(echo $dataset | sed 's/rgbtvg_//')
        MODEL_CHECKPOINT="${MODEL_BASE_PATH}/MMVG_${modality}_${DATASET_SHORT}_best.pth"

        echo ""
        echo "Test $CURRENT_TEST/$TOTAL_TESTS: $dataset + $modality"
        echo "   Model: $MODEL_CHECKPOINT"
        echo "   Output: ./outputs/visualizations/RGBT_VGNet/$dataset/$modality"
        echo "----------------------------------------"

        # Check if model file exists
        if [ ! -f "$MODEL_CHECKPOINT" ]; then
            echo "Warning: Model file does not exist: $MODEL_CHECKPOINT"
            echo "   Skipping this test..."
            FAILED_TESTS+=("$dataset-$modality (model file not found)")
            continue
        fi

        # Run test
        if bash script_visualize/shell_scripts/visualize_rgbt_vgnet.sh "$dataset" "$modality" "$MODEL_CHECKPOINT"; then
            echo "Test successful: $dataset + $modality"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "Test failed: $dataset + $modality"
            FAILED_TESTS+=("$dataset-$modality")
        fi

        echo "----------------------------------------"

        # Brief pause to avoid GPU overload
        sleep 2
    done
done

# End time and statistics
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "RGBT-VGNet visualization completed!"
echo "========================================"
echo "Test statistics:"
echo "   Total tests: $TOTAL_TESTS"
echo "   Success: $SUCCESS_COUNT"
echo "   Failed: $((TOTAL_TESTS - SUCCESS_COUNT))"
echo "   Duration: ${MINUTES}m ${SECONDS}s"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for failed in "${FAILED_TESTS[@]}"; do
        echo "   - $failed"
    done
fi

if [ $SUCCESS_COUNT -eq $TOTAL_TESTS ]; then
    echo ""
    echo "All RGBT-VGNet visualizations completed successfully!"
    exit 0
else
    echo ""
    echo "Some RGBT-VGNet tests failed, please check the failed list above"
    exit 1
fi
