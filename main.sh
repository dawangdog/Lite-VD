GPU_LIST=$1
DATA=$2
IPC=$3
ARG4=$4
ARG5=$5
SELECTION_FILE=
ARTIFACT_FILE=
DISTILL_STATE_FILE=
FRAMES_WAS_SET=${FRAMES+x}
FRAMES=${FRAMES:-16}
VAE_MODEL=${VAE_MODEL:-2DVAE}
LATENT_ROOT=${LATENT_ROOT:-./latent_cache}
LATENT_DIR=${LATENT_ROOT}/${DATA}/${VAE_MODEL}/frames_${FRAMES}
LATENT_FILE=${LATENT_DIR}/latents.pt
MODEL=${MODEL:-VideoMAE}
EVAL_MODE=${EVAL_MODE:-SS}
EVAL_MODELS=${EVAL_MODELS:-}
METHOD=${METHOD:-ImportanceHOSVD}
RUN_MODE=${RUN_MODE:-distill}
NUM_EVAL=${NUM_EVAL:-1}
EPOCH_EVAL_TRAIN=${EPOCH_EVAL_TRAIN:-300}
EVAL_TEST_FREQ=${EVAL_TEST_FREQ:-100}
BATCH_TRAIN=${BATCH_TRAIN:-256}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-8}
VIDEO_TRANSFORMER_TUNE_MODE=${VIDEO_TRANSFORMER_TUNE_MODE:-full_finetune}
VIDEO_TRANSFORMER_LR_LINEAR_PROBE=${VIDEO_TRANSFORMER_LR_LINEAR_PROBE:-0.001}
VIDEO_TRANSFORMER_LR_FINETUNE=${VIDEO_TRANSFORMER_LR_FINETUNE:-0.00005}
VIDEO_TRANSFORMER_WEIGHT_DECAY=${VIDEO_TRANSFORMER_WEIGHT_DECAY:-0.05}
SKIP_EVAL_AFTER_DISTILL=${SKIP_EVAL_AFTER_DISTILL:-0}
LVDD_SELECT_MODE=${LVDD_SELECT_MODE:-DAPS}
SELECT_MODE_OVERRIDE=${SELECT_MODE_OVERRIDE:-}
COMPRESS_RATIO=${COMPRESS_RATIO:-0.75}
RANK_BOOST=${RANK_BOOST:-0.15}
HIGH_TOKEN_RATIO=${HIGH_TOKEN_RATIO:-0.06}
MEDIUM_TOKEN_RATIO=${MEDIUM_TOKEN_RATIO:-0.20}
BASELINE_ITERATION=${BASELINE_ITERATION:-}
BASELINE_EVAL_IT=${BASELINE_EVAL_IT:-}
BASELINE_INLOOP_EVAL=${BASELINE_INLOOP_EVAL:-0}
BASELINE_INIT_ONLY=${BASELINE_INIT_ONLY:-0}
INIT=${INIT:-noise}
PIXEL_ARTIFACT_DTYPE=${PIXEL_ARTIFACT_DTYPE:-float32}
VDSD_SPC=${VDSD_SPC:-}
VDSD_DPC=${VDSD_DPC:-}
VDSD_VPC=${VDSD_VPC:-}
VDSD_LR_DYNAMIC=${VDSD_LR_DYNAMIC:-}
VDSD_LR_HAL=${VDSD_LR_HAL:-}
VDSD_START_IT=${VDSD_START_IT:-0}
VDSD_BATCH_REAL=${VDSD_BATCH_REAL:-64}
PRELOAD=${PRELOAD:-}

if [ -z "${PRELOAD}" ]; then
case "${DATA}" in
    SSv2|Kinetics400|Kinetics400_long)
    PRELOAD=0
    ;;
    *)
    PRELOAD=1
    ;;
esac
fi

if [ -z "${BASELINE_ITERATION}" ]; then
case "${METHOD}" in
    DM|MTT)
    BASELINE_ITERATION=1000
    ;;
    DM+VDSD|MTT+VDSD)
    BASELINE_ITERATION=500
    ;;
    *)
    BASELINE_ITERATION=100
    ;;
esac
fi

if [ -z "${BASELINE_EVAL_IT}" ]; then
BASELINE_EVAL_IT=${BASELINE_ITERATION}
fi

VDSD_ITERATION=${VDSD_ITERATION:-${BASELINE_ITERATION}}
VDSD_EVAL_IT=${VDSD_EVAL_IT:-${BASELINE_EVAL_IT}}

case "${ARG4}" in
    *synthetic_data.pt)
    ARTIFACT_FILE=${ARG4}
    ;;
    *distill_state.pt)
    DISTILL_STATE_FILE=${ARG4}
    ;;
    "")
    ;;
    *)
    SELECTION_FILE=${ARG4}
    ;;
esac

if [ -n "${ARG5}" ]; then
case "${ARG5}" in
    *synthetic_data.pt)
    ARTIFACT_FILE=${ARG5}
    ;;
    *distill_state.pt)
    DISTILL_STATE_FILE=${ARG5}
    ;;
esac
fi

mkdir -p ${LATENT_DIR}

if [ -z "${GPU_LIST}" ]; then
GPU_LIST=0
fi

case "${METHOD}" in
    Random|Herding|Full|PixelRandom|PixelHerding|PixelFull|DM|MTT|DM+VDSD|MTT+VDSD)
    USE_LATENTS=0
    ;;
    *)
    USE_LATENTS=1
    ;;
esac

if [ "${RUN_MODE}" = "full_data" ]; then
USE_LATENTS=0
fi

if [ "${USE_LATENTS}" = "1" ]; then
if [ -f "${LATENT_FILE}" ]; then
LATENT_ARGS="--latent_file ${LATENT_FILE} --latent_cache_dir ${LATENT_DIR}"
echo "Reusing cached latents: ${LATENT_FILE}"
else
LATENT_ARGS="--latent_cache_dir ${LATENT_ROOT}"
echo "No cached latents found. The first encoding will be saved to: ${LATENT_FILE}"
fi
else
LATENT_ARGS=""
echo "Pixel/full-data baseline does not use latent cache."
fi

echo "Visible GPUs: ${GPU_LIST}"
echo "Run mode: ${RUN_MODE}"
echo "Method: ${METHOD}"
echo "Eval model: ${MODEL}"
if [ -n "${EVAL_MODELS}" ]; then
echo "Cross-architecture eval models: ${EVAL_MODELS}"
EVAL_MODELS_ARGS="--eval_models ${EVAL_MODELS}"
else
EVAL_MODELS_ARGS=""
fi
echo "Tune mode: ${VIDEO_TRANSFORMER_TUNE_MODE}"
echo "Eval epochs: ${EPOCH_EVAL_TRAIN}, test freq: ${EVAL_TEST_FREQ}, train batch: ${BATCH_TRAIN}, test batch: ${TEST_BATCH_SIZE}"
echo "Frames: ${FRAMES}, encode batch: ${ENCODE_BATCH_SIZE}"
echo "Compression: ratio=${COMPRESS_RATIO}, rank_boost=${RANK_BOOST}, high=${HIGH_TOKEN_RATIO}, medium=${MEDIUM_TOKEN_RATIO}"
echo "Baseline distillation: iteration=${BASELINE_ITERATION}, eval_it=${BASELINE_EVAL_IT}, in-loop eval=${BASELINE_INLOOP_EVAL}, init only=${BASELINE_INIT_ONLY}, init=${INIT}, pixel dtype=${PIXEL_ARTIFACT_DTYPE}"
echo "Dataset preload: ${PRELOAD} (large datasets should use PRELOAD=0)"

if [ "${RUN_MODE}" = "full_data" ]; then
FULL_DATA_ARGS="--full_data_baseline"
SELECT_MODE=full
echo "Full-data baseline is enabled: distillation and compression will be skipped."
elif [ "${METHOD}" = "Random" ] || [ "${METHOD}" = "Herding" ] || [ "${METHOD}" = "Full" ] || [ "${METHOD}" = "PixelRandom" ] || [ "${METHOD}" = "PixelHerding" ] || [ "${METHOD}" = "PixelFull" ] || [ "${METHOD}" = "DM" ] || [ "${METHOD}" = "MTT" ] || [ "${METHOD}" = "DM+VDSD" ] || [ "${METHOD}" = "MTT+VDSD" ]; then
FULL_DATA_ARGS=""
SELECT_MODE=full
echo "Pixel-space baseline mode is enabled."
elif [ "${METHOD}" = "LVDD_PCA" ] || [ "${METHOD}" = "LVDD_Tucker" ]; then
FULL_DATA_ARGS=""
SELECT_MODE=random
echo "LVDD-style latent baseline mode is enabled."
echo "LVDD select mode: ${LVDD_SELECT_MODE}"
else
FULL_DATA_ARGS=""
SELECT_MODE=summary_dpp
fi

if [ -n "${SELECT_MODE_OVERRIDE}" ]; then
SELECT_MODE=${SELECT_MODE_OVERRIDE}
echo "Selection mode override: ${SELECT_MODE}"
fi

if [ "${METHOD}" = "DM+VDSD" ] || [ "${METHOD}" = "MTT+VDSD" ]; then
if [ -z "${VDSD_VPC}" ]; then
if [ "${IPC}" = "1" ]; then
VDSD_VPC=1
elif [ "${IPC}" = "5" ]; then
VDSD_VPC=5
else
VDSD_VPC=${IPC}
fi
fi
if [ -z "${VDSD_SPC}" ]; then
VDSD_SPC=$((2 * VDSD_VPC))
fi
if [ -z "${VDSD_DPC}" ]; then
VDSD_DPC=$((2 * VDSD_VPC))
fi
if [ -z "${VDSD_LR_DYNAMIC}" ]; then
if [ "${VDSD_VPC}" = "5" ]; then
VDSD_LR_DYNAMIC=1e3
else
VDSD_LR_DYNAMIC=1e-4
fi
fi
if [ -z "${VDSD_LR_HAL}" ]; then
if [ "${VDSD_VPC}" = "5" ]; then
VDSD_LR_HAL=1e-6
else
VDSD_LR_HAL=1e-5
fi
fi

VDSD_STATIC_PREFIX=
if [ "${DATA}" = "miniUCF101" ] || [ "${DATA}" = "miniUCF101_long" ]; then
VDSD_STATIC_PREFIX=miniUCF
elif [ "${DATA}" = "HMDB51" ]; then
VDSD_STATIC_PREFIX=hmdb
fi
if [ -n "${VDSD_STATIC_PREFIX}" ]; then
VDSD_STATIC_PATH=${VDSD_STATIC_PATH:-../video_distillation-main/sh/s2d/${VDSD_STATIC_PREFIX}_spc${VDSD_SPC}.pt}
fi
if [ -n "${VDSD_STATIC_PATH}" ] && [ -f "${VDSD_STATIC_PATH}" ]; then
VDSD_STATIC_ARGS="--no_train_static --path_static ${VDSD_STATIC_PATH}"
else
VDSD_STATIC_ARGS=""
fi
VDSD_ARGS="--spc ${VDSD_SPC} --dpc ${VDSD_DPC} --vpc ${VDSD_VPC} --lr_dynamic ${VDSD_LR_DYNAMIC} --lr_hal ${VDSD_LR_HAL} --Iteration ${VDSD_ITERATION} --eval_it ${VDSD_EVAL_IT} --startIt ${VDSD_START_IT} --batch_real ${VDSD_BATCH_REAL} ${VDSD_STATIC_ARGS}"
echo "VDSD: spc=${VDSD_SPC}, dpc=${VDSD_DPC}, vpc=${VDSD_VPC}, lr_dynamic=${VDSD_LR_DYNAMIC}, lr_hal=${VDSD_LR_HAL}, iteration=${VDSD_ITERATION}"
if [ -n "${VDSD_STATIC_ARGS}" ]; then
echo "VDSD static memory: ${VDSD_STATIC_PATH}"
fi
else
VDSD_ARGS=""
fi

if [ "${BASELINE_INLOOP_EVAL}" = "0" ]; then
BASELINE_EVAL_ARGS="--skip_baseline_inloop_eval"
else
BASELINE_EVAL_ARGS=""
fi

if [ "${BASELINE_INIT_ONLY}" = "1" ]; then
BASELINE_INIT_ARGS="--baseline_init_only"
else
BASELINE_INIT_ARGS=""
fi

if [ -n "${SELECTION_FILE}" ]; then
SELECTION_ARGS="--selected_indices_file ${SELECTION_FILE}"
echo "Reusing fixed selected indices from: ${SELECTION_FILE}"
else
SELECTION_ARGS=""
fi

if [ -n "${ARTIFACT_FILE}" ]; then
ARTIFACT_ARGS="--artifact_file ${ARTIFACT_FILE}"
echo "Reusing full distilled artifact from: ${ARTIFACT_FILE}"
else
ARTIFACT_ARGS=""
fi

if [ -n "${DISTILL_STATE_FILE}" ]; then
DISTILL_STATE_ARGS="--distill_state_file ${DISTILL_STATE_FILE}"
echo "Reusing distillation state from: ${DISTILL_STATE_FILE}"
else
DISTILL_STATE_ARGS=""
fi

if [ "${SKIP_EVAL_AFTER_DISTILL}" = "1" ]; then
SKIP_EVAL_AFTER_DISTILL_ARGS="--skip_eval_after_distill"
echo "Visualization-only mode: downstream evaluation after distillation will be skipped."
else
SKIP_EVAL_AFTER_DISTILL_ARGS=""
fi

if [ "${PRELOAD}" = "1" ]; then
PRELOAD_ARGS="--preload"
else
PRELOAD_ARGS=""
fi

CUDA_VISIBLE_DEVICES=${GPU_LIST} python main_method.py \
--method ${METHOD} \
--vae_model ${VAE_MODEL} \
--select_mode ${SELECT_MODE} \
--dataset ${DATA} \
--data_path distill_utils/data \
--eval_mode ${EVAL_MODE} \
${EVAL_MODELS_ARGS} \
--ipc ${IPC} \
--num_eval ${NUM_EVAL} \
--epoch_eval_train ${EPOCH_EVAL_TRAIN} \
--eval_test_freq ${EVAL_TEST_FREQ} \
--Iteration ${BASELINE_ITERATION} \
--eval_it ${BASELINE_EVAL_IT} \
--lr_net 0.01 \
--batch_train ${BATCH_TRAIN} \
--test_batch_size ${TEST_BATCH_SIZE} \
--model ${MODEL} \
--num_workers ${NUM_WORKERS} \
--random_state 42 \
--init ${INIT} \
--pixel_artifact_dtype ${PIXEL_ARTIFACT_DTYPE} \
--compress_ratio ${COMPRESS_RATIO} \
--rank_boost ${RANK_BOOST} \
--encode_batch_size ${ENCODE_BATCH_SIZE} \
--frames ${FRAMES} \
--importance_temporal_weight 0.45 \
--importance_spatial_weight 0.15 \
--importance_local_weight 0.25 \
--importance_energy_weight 0.15 \
--high_token_ratio ${HIGH_TOKEN_RATIO} \
--medium_token_ratio ${MEDIUM_TOKEN_RATIO} \
--video_transformer_tune_mode ${VIDEO_TRANSFORMER_TUNE_MODE} \
--video_transformer_lr_linear_probe ${VIDEO_TRANSFORMER_LR_LINEAR_PROBE} \
--video_transformer_lr_finetune ${VIDEO_TRANSFORMER_LR_FINETUNE} \
--video_transformer_weight_decay ${VIDEO_TRANSFORMER_WEIGHT_DECAY} \
--lvdd_select_mode ${LVDD_SELECT_MODE} \
${FULL_DATA_ARGS} \
${LATENT_ARGS} \
${SELECTION_ARGS} \
${ARTIFACT_ARGS} \
${DISTILL_STATE_ARGS} \
${SKIP_EVAL_AFTER_DISTILL_ARGS} \
${BASELINE_EVAL_ARGS} \
${BASELINE_INIT_ARGS} \
${VDSD_ARGS} \
${PRELOAD_ARGS}
