#!/bin/bash
# =============================================================================
# 1015 — MAV-Flow, MEAN over twin critics everywhere, 5 maps x 3 seeds, 500k steps
# =============================================================================
# 15 jobs = 5 maps x 3 seeds:
#
#   SMACv1 Good   : 5m_vs_6m, 3s5z_vs_3s6z
#   SMACv2 Replay : terran_5_vs_5, zerg_5_vs_5, terran_10_vs_10
#   seeds         : 0, 1, 2
#
# WHAT THIS RUN TESTS
#   Every run 0716-1013 combined the two twin critics with min (clipped double-Q).
#   This arm uses mean everywhere, as MAC-Flow does (App. E: "mean(Q1,Q2) ... instead of
#   min(Q1,Q2), to avoid pessimism in an offline RL setting") and as Value Flows does
#   by default:
#       ret_agg=mean   BCFM bootstrap target y, and a' scoring
#       dcfm_agg=mean  DCFM target point and velocity
#       q_agg=mean     q_head teacher V (what the deployed policy ranks by)
#   The matched control is the 1011 run (same maps, seeds and code path). It differs
#   ONLY in these three settings, which were all 'min' there.
#   What to read: critic calibration vs MC return-to-go (1011 underestimated:
#   terran_5_vs_5 -0.65 to -0.71, 5m_vs_6m -2.8), q_head ranking, then win rate over
#   3 seeds. Failure signature to watch: bcfm/returns_mean inflating during training,
#   which is the optimism blow-up min was introduced to stop on 3m Poor (0716).
#
# USAGE
#   SMOKE=1 bash 1015_mean_launch_500k.sh   # ~5-10 min: one tiny job in the foreground
#                                           # (5m_vs_6m, 200 steps, 2 eval episodes).
#                                           # Run this first on a new machine.
#   bash 1015_mean_launch_500k.sh           # the full queue, in the background
#   Overrides (environment variables):
#     GPUS="0 1 2 3"    GPU ids to use, one worker each        (default: 0 1 2 3)
#     SEEDS="0 1 2"                                             (default: 0 1 2)
#     OG_MARL_DATA_DIR  vault root                              (default: ~/vaults/og_marl)
#     MIN_AVAIL_GB      host RAM a job needs free before it loads (default: 30)
#
# RUNTIME (measured with the same code path on 1011, idle A4000s, 1 job per GPU)
#     map               step_ms   wall/job (incl. data load + 5 evals x 50 eps)
#     5m_vs_6m           27.0      ~4.1 h
#     terran_5_vs_5      27.8      ~4.1 h
#     zerg_5_vs_5        27.7      ~4.1 h
#     3s5z_vs_3s6z       40.5      ~6.1 h
#     terran_10_vs_10    45.2      ~6.9 h
#   => ~76 GPU-hours; ~20.5 h wall clock on 4 GPUs. One job saturates an A4000, so
#   running 2+ jobs per GPU does not help.
#
# SCHEDULING
#   One work-stealing pool; each GPU worker claims the next job via atomic mkdir.
#   Seed-major, longest map first within a seed, so seed 0 of all maps lands first.
#
# HOST MEMORY
#   Building a dataset peaks at ~22-24 GB RSS (3s5z, terran_10_vs_10). Only one job
#   loads at a time (flock), and only once MemAvailable >= MIN_AVAIL_GB. On a machine
#   with less free RAM, lower MIN_AVAIL_GB rather than running loads in parallel.
#
# NAMING
#   save dir / W&B group : exp/1015_<map>_<quality>_mean_n4_500k/sd00{0,1,2}_*
#   log / status files   : logs/1015_<map>_<quality>_mean_n4_sd<N>_500k.{log,status}
#   W&B runs offline (project value-flows); workers `wandb sync` when the queue ends.
# =============================================================================

set -u

RUNDATE=1015
ARM=mean
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
SAVE_DIR="$ROOT_DIR/exp"

# kind|map|quality — longest first within a seed (see SCHEDULING).
MAPS=(
    "smacv2|terran_10_vs_10|Replay"
    "smac|3s5z_vs_3s6z|Good"
    "smacv2|terran_5_vs_5|Replay"
    "smacv2|zerg_5_vs_5|Replay"
    "smac|5m_vs_6m|Good"
)
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"

W_MODE=uniform
OFFLINE_STEPS=500000
EVAL_INTERVAL=100000
EVAL_EPISODES=50
SAVE_INTERVAL=100000
LOG_INTERVAL=250
CHURN_PROBE=256
SMAC_SOURCE=og_marl     # SMACv1 eval-env recorder; inert for smacv2

# Passed to main.py EXPLICITLY so this file records the experiment; the preflight
# asserts the agent defaults still agree.
N_CANDIDATES=4
N_Q_CANDIDATES=4
N_TARGET_CANDIDATES=4
RET_AGG=mean
DCFM_AGG=mean
Q_AGG=mean

MIN_AVAIL_GB="${MIN_AVAIL_GB:-30}"
LOAD_TIMEOUT_S=1800     # release the load lock after this even without the marker

cd "$ROOT_DIR" || exit 1
mkdir -p "$LOG_DIR" "$SAVE_DIR"

# LOAD-BEARING: XLA/LLVM need address space for compilation and per-thread stacks,
# and a capped `ulimit -v` makes pthread_create fail.
ulimit -v unlimited

# protobuf workaround so wandb (modern protobuf) and pysc2 (old protobuf) coexist.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OG_MARL_DATA_DIR="${OG_MARL_DATA_DIR:-$HOME/vaults/og_marl}"
# Do not grab 75% of the card up front; one job needs well under 2 GB.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PID_FILE="$LOG_DIR/${RUNDATE}_${ARM}_queue_pids"
SYNC_LOG="$LOG_DIR/${RUNDATE}_${ARM}_sync.log"
CLAIM_DIR="$LOG_DIR/${RUNDATE}_${ARM}_claims"
LOAD_LOCK="$LOG_DIR/${RUNDATE}_${ARM}_load.lock"
SNAPSHOT_DIR="$SAVE_DIR/${RUNDATE}_${ARM}_code_snapshot"

# --------------------------------------------------------------------------- #
#                                 JOB TABLE                                   #
# --------------------------------------------------------------------------- #
JOBS=()
for SEED in "${SEED_LIST[@]}"; do
    for M in "${MAPS[@]}"; do
        JOBS+=("${M}|${SEED}")
    done
done

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

env_name_for() {  # kind, map, quality
    if [ "$1" = smac ]; then printf 'smac_%s_%s' "$2" "$3"
    else printf 'smacv2_%s_%s' "$2" "$(lower "$3")"; fi
}
group_for() {     # map, quality
    printf '%s_%s_%s_%s_n%d_500k' "$RUNDATE" "$1" "$(lower "$2")" "$ARM" "$N_CANDIDATES"
}
run_tag_for() {   # map, quality, seed
    printf '%s_%s_%s_%s_n%d_sd%d_500k' "$RUNDATE" "$1" "$(lower "$2")" "$ARM" "$N_CANDIDATES" "$3"
}

avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }

agent_flags() {
    printf '%s\n' \
        --agent=agents/mav_flows.py \
        --agent.w_mode="$W_MODE" \
        --agent.num_candidates="$N_CANDIDATES" \
        --agent.num_q_candidates="$N_Q_CANDIDATES" \
        --agent.num_target_candidates="$N_TARGET_CANDIDATES" \
        --agent.ret_agg="$RET_AGG" \
        --agent.dcfm_agg="$DCFM_AGG" \
        --agent.q_agg="$Q_AGG"
}

# --------------------------------------------------------------------------- #
#                       W&B SYNC (once, when a worker finishes)               #
# --------------------------------------------------------------------------- #
sync_all_runs() {
    local d
    for d in "$SAVE_DIR"/${RUNDATE}_*_${ARM}_n*_500k/sd*/wandb/offline-run-*; do
        [ -d "$d" ] || continue
        printf '[%s] syncing %s\n' "$(date --iso-8601=seconds)" "$d" >> "$SYNC_LOG"
        wandb sync "$d" >> "$SYNC_LOG" 2>&1 \
            || printf '[%s] sync FAILED: %s (re-run `wandb sync` by hand)\n' \
                 "$(date --iso-8601=seconds)" "$d" >> "$SYNC_LOG"
    done
}

# --------------------------------------------------------------------------- #
#                              WORKER (one GPU)                               #
# --------------------------------------------------------------------------- #
CURRENT_JOB_PID=""

terminate_queue() {
    if [ -n "$CURRENT_JOB_PID" ] && kill -0 "$CURRENT_JOB_PID" 2>/dev/null; then
        kill "$CURRENT_JOB_PID"
        wait "$CURRENT_JOB_PID" 2>/dev/null || true
    fi
    exit 143
}

run_job() {  # gpu, kind, map, quality, seed
    local gpu="$1" kind="$2" map="$3" quality="$4" seed="$5"
    local env_name group run_tag log_file status_file job_pid exit_code
    local t0 t1 elapsed started_iso waited extra=() aflags=()

    env_name=$(env_name_for "$kind" "$map" "$quality")
    group=$(group_for "$map" "$quality")
    run_tag=$(run_tag_for "$map" "$quality" "$seed")
    log_file="$LOG_DIR/${run_tag}.log"
    status_file="$LOG_DIR/${run_tag}.status"
    [ "$kind" = smac ] && extra=(--smac_env_source="$SMAC_SOURCE")
    mapfile -t aflags < <(agent_flags)

    # ---- memory gate: one loader at a time, and only with headroom ----
    printf 'WAITING gpu=%s map=%s seed=%s for the load slot (avail=%s GB)\n' \
        "$gpu" "$map" "$seed" "$(avail_gb)" > "$status_file"
    exec 9>"$LOAD_LOCK"
    flock 9
    while [ "$(avail_gb)" -lt "$MIN_AVAIL_GB" ]; do
        printf 'WAITING gpu=%s map=%s seed=%s for memory: avail=%s GB < %s GB (%s)\n' \
            "$gpu" "$map" "$seed" "$(avail_gb)" "$MIN_AVAIL_GB" \
            "$(date --iso-8601=seconds)" > "$status_file"
        sleep 60
    done

    t0=$(date +%s)
    started_iso=$(date --iso-8601=seconds)
    printf 'RUNNING gpu=%s map=%s seed=%s env=%s started=%s\n' \
        "$gpu" "$map" "$seed" "$env_name" "$started_iso" > "$status_file"

    # `9>&-`: keep the lock fd out of the 4-7 h job, so the load slot belongs to
    # this worker alone (released by the flock -u below).
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 python main.py \
        --env_name="$env_name" \
        "${aflags[@]}" \
        "${extra[@]}" \
        --seed="$seed" \
        --offline_steps="$OFFLINE_STEPS" \
        --log_interval="$LOG_INTERVAL" \
        --eval_interval="$EVAL_INTERVAL" \
        --eval_episodes="$EVAL_EPISODES" \
        --save_interval="$SAVE_INTERVAL" \
        --churn_probe_size="$CHURN_PROBE" \
        --save_dir="$SAVE_DIR" \
        --enable_wandb=1 \
        --wandb_mode=offline \
        --wandb_run_group="$group" \
        > "$log_file" 2>&1 9>&- &
    job_pid=$!
    CURRENT_JOB_PID=$job_pid

    # Hold the load slot until the dataset is built (main.py prints the parity
    # line right after), the job dies, or the timeout passes.
    waited=0
    while kill -0 "$job_pid" 2>/dev/null && [ "$waited" -lt "$LOAD_TIMEOUT_S" ] \
          && ! grep -q '\[OK\] SMAC obs parity' "$log_file" 2>/dev/null; do
        sleep 10; waited=$((waited + 10))
    done
    flock -u 9
    exec 9>&-

    if wait "$job_pid"; then
        t1=$(date +%s); elapsed=$((t1 - t0))
        printf 'COMPLETED gpu=%s map=%s seed=%s started=%s finished=%s elapsed=%s (%dh%02dm)\n' \
            "$gpu" "$map" "$seed" "$started_iso" "$(date --iso-8601=seconds)" \
            "$elapsed" $((elapsed/3600)) $(((elapsed%3600)/60)) > "$status_file"
    else
        exit_code=$?
        t1=$(date +%s); elapsed=$((t1 - t0))
        printf 'FAILED gpu=%s map=%s seed=%s exit_code=%s started=%s finished=%s elapsed=%s (%dh%02dm)\n' \
            "$gpu" "$map" "$seed" "$exit_code" "$started_iso" "$(date --iso-8601=seconds)" \
            "$elapsed" $((elapsed/3600)) $(((elapsed%3600)/60)) > "$status_file"
    fi
    CURRENT_JOB_PID=""
}

if [ "${1:-}" = "--worker" ]; then
    trap terminate_queue TERM INT HUP
    gpu="$2"
    for idx in "${!JOBS[@]}"; do
        mkdir "$CLAIM_DIR/$idx" 2>/dev/null || continue
        IFS='|' read -r kind map quality seed <<< "${JOBS[$idx]}"
        run_job "$gpu" "$kind" "$map" "$quality" "$seed"
    done
    sync_all_runs
    exit 0
fi

# --------------------------------------------------------------------------- #
#                                  PREFLIGHT                                  #
# --------------------------------------------------------------------------- #
set -e

python - <<'PYCHK' || { echo "ERROR: missing packages. Run: pip install -r requirements_mav_lock.txt"; exit 1; }
import importlib
missing = [m for m in ('jax', 'flax', 'og_marl', 'flashbax', 'smac', 'smacv2', 'pysc2', 'wandb')
           if importlib.util.find_spec(m) is None]
assert not missing, f'not importable: {missing}'
print('[OK] python packages importable.')
PYCHK

SC2_DIR="${SC2PATH:-$HOME/StarCraftII}"
[ -d "$SC2_DIR/Maps/SMAC_Maps" ] \
    || { echo "ERROR: no $SC2_DIR/Maps/SMAC_Maps. Install StarCraft II and the SMAC maps (README.md)."; exit 1; }
echo "[OK] StarCraft II at $SC2_DIR (set SC2PATH to override)."

if [ "${SMOKE:-0}" = 1 ]; then
    MAPS=("smac|5m_vs_6m|Good")
fi
for M in "${MAPS[@]}"; do
    IFS='|' read -r kind map quality <<< "$M"
    dir=$([ "$kind" = smac ] && echo smac_v1 || echo smac_v2)
    VAULT="$OG_MARL_DATA_DIR/$dir/${map}.vlt/$quality"
    [ -d "$VAULT" ] || { echo "ERROR: missing vault $VAULT (see README.md to download)"; exit 1; }
done
echo "[OK] all ${#MAPS[@]} vaults present under $OG_MARL_DATA_DIR."

for GPU in "${GPU_LIST[@]}"; do
    USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) \
        || { echo "ERROR: GPU $GPU not visible to nvidia-smi."; exit 1; }
    if [ "$USED" -gt 1024 ]; then
        echo "WARNING: GPU $GPU already holds ${USED} MiB — another job may be on it."
    fi
done
echo "[OK] GPUs ${GPU_LIST[*]} visible. Host MemAvailable: $(avail_gb) GB."

GPU_CHECK=$(CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -c \
    "import jax; print('cuda' if any('cuda' in str(d).lower() for d in jax.devices()) else 'cpu')")
[ "$GPU_CHECK" = "cuda" ] || { echo "ERROR: jax is on CPU, not GPU. Aborting."; exit 1; }
echo "[OK] jax is using GPU."

PF_N_CAND="$N_CANDIDATES" PF_N_Q="$N_Q_CANDIDATES" PF_N_T="$N_TARGET_CANDIDATES" \
PF_RET="$RET_AGG" PF_DCFM="$DCFM_AGG" PF_Q="$Q_AGG" python - <<'PYCHK'
import os
from agents.mav_flows import MAVFlowAgent, get_config

c = get_config()
assert c['w_normalize'] is False, 'w_normalize must be False (see the 0831/0903 notes)'
assert hasattr(MAVFlowAgent.sample_actions_rejection, '__wrapped__'), \
    'sample_actions_rejection is not jitted -- eval will be ~500x slower (see 0903)'
want = {
    'num_candidates':        int(os.environ['PF_N_CAND']),
    'num_q_candidates':      int(os.environ['PF_N_Q']),
    'num_target_candidates': int(os.environ['PF_N_T']),
    'ret_agg':               os.environ['PF_RET'],
    'dcfm_agg':              os.environ['PF_DCFM'],
    'q_agg':                 os.environ['PF_Q'],
}
bad = {k: (c[k], v) for k, v in want.items() if c[k] != v}
assert not bad, f'agent defaults no longer match this launch script: {bad} (have, want)'
print(f"[OK] agent: ret/dcfm/q_agg={c['ret_agg']}/{c['dcfm_agg']}/{c['q_agg']}, "
      f"n_cand/n_q/n_t={c['num_candidates']}/{c['num_q_candidates']}/"
      f"{c['num_target_candidates']}, sampler jitted")
PYCHK

if ! grep -q 'api\.wandb\.ai' "${HOME}/.netrc" 2>/dev/null; then
    echo "WARNING: no api.wandb.ai entry in ~/.netrc. Runs still log offline under"
    echo "         exp/*/wandb/, but the end-of-queue sync will fail until you run: wandb login"
else
    echo "[OK] wandb credentials present."
fi

# --------------------------------------------------------------------------- #
#                         SMOKE TEST (SMOKE=1, foreground)                    #
# --------------------------------------------------------------------------- #
if [ "${SMOKE:-0}" = 1 ]; then
    mapfile -t AFLAGS < <(agent_flags)
    SMOKE_LOG="$LOG_DIR/${RUNDATE}_${ARM}_smoke.log"
    echo ""
    echo "Smoke test: 5m_vs_6m Good, 200 steps, evals at steps 1 and 200 (2 episodes each)."
    echo "Log: $SMOKE_LOG"
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python main.py \
        --env_name=smac_5m_vs_6m_Good \
        "${AFLAGS[@]}" \
        --smac_env_source="$SMAC_SOURCE" \
        --seed=0 \
        --offline_steps=200 \
        --log_interval=50 \
        --eval_interval=200 \
        --eval_episodes=2 \
        --save_interval=200 \
        --churn_probe_size="$CHURN_PROBE" \
        --save_dir="$SAVE_DIR/${RUNDATE}_${ARM}_smoke" \
        --enable_wandb=0 \
        > "$SMOKE_LOG" 2>&1 \
        || { echo "SMOKE FAILED — last lines of $SMOKE_LOG:"; tail -20 "$SMOKE_LOG"; exit 1; }
    echo "[OK] smoke test passed: trained, evaluated in SC2 and saved a checkpoint."
    echo "     The full queue is: bash $0"
    exit 0
fi

# Record exactly what ran, including any uncommitted edits.
mkdir -p "$SNAPSHOT_DIR"
git rev-parse HEAD > "$SNAPSHOT_DIR/git_head.txt"
git diff HEAD > "$SNAPSHOT_DIR/git_diff_vs_head.patch"
cp agents/mav_flows.py main.py "$0" "$SNAPSHOT_DIR/"
echo "[OK] code snapshot in $SNAPSHOT_DIR (HEAD $(cut -c1-7 "$SNAPSHOT_DIR/git_head.txt") + diff)."

rm -rf "$CLAIM_DIR"; mkdir -p "$CLAIM_DIR"
: > "$LOAD_LOCK"

# --------------------------------------------------------------------------- #
#                                   LAUNCH                                    #
# --------------------------------------------------------------------------- #
echo ""
echo "${#JOBS[@]} jobs over ${#GPU_LIST[@]} GPU workers (work-stealing, one loader at a time):"
for idx in "${!JOBS[@]}"; do
    IFS='|' read -r k_ m_ q_ sd_ <<< "${JOBS[$idx]}"
    printf '  %2d. %-16s %-6s seed=%s\n' "$((idx+1))" "$m_" "$q_" "$sd_"
done
echo ""
: > "$PID_FILE"
for GPU in "${GPU_LIST[@]}"; do
    nohup bash "$0" --worker "$GPU" > /dev/null 2>&1 &
    printf '%s\n' "$!" >> "$PID_FILE"
    echo "  [GPU $GPU] worker PID=$!"
done

cat <<EOF

~76 GPU-hours total; ~20.5 h wall clock on 4 GPUs. Loads are staggered (one at a
time, MemAvailable >= ${MIN_AVAIL_GB} GB), so the last worker starts training
~15-20 min after the first.

Queue state:  grep -h . logs/${RUNDATE}_*_${ARM}_*_500k.status | sort
Monitor:      tail -f logs/${RUNDATE}_5m_vs_6m_good_${ARM}_n${N_CANDIDATES}_sd0_500k.log
Kill all:     kill \$(cat $PID_FILE)
A FAILED job keeps its claim. To redo one: rm -rf $CLAIM_DIR/<idx> && bash $0 --worker <gpu>
EOF
