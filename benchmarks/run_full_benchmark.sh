#!/usr/bin/env bash
# Full benchmark: run from the benchmarks/ folder. Every suite is resumable, re-runnning
# this script continues where it stopped. Tables are written as csv,json; add
# --export-formats all (or csv,json,tex) to any command to also get LaTeX tables
set -eo pipefail

source ../.venv/bin/activate
set -u # after activation, which may read unset variables

# One seed for everything, scenarios/workdays are directly comparable
SEED=42

# insertion ablation
python3 run_ablation_suite.py \
  --outdir ablation_suite \
  --seed "$SEED" \
  --agent-counts 1 2 3 \
  --request-counts 10 15 20 25 30 40 50 60 \
  --types R C M \
  --timings U P \
  --nr-repeats 3 \
  --corpus both \
  --nr-probes 3 \
  --nr-measure-repeats 5 \
  --export-formats all

# single scenario with plots
python3 run_suite.py \
  --outdir single_plots \
  --seed "$SEED" \
  --sizes L \
  --types M \
  --timings P \
  --soc-test stress \
  --nr-repeats 1 \
  --solver-name gurobi \
  --solver-config-type gurobi \
  --plots

# single scenario suite
python3 run_suite.py \
  --outdir single_suite \
  --seed "$SEED" \
  --sizes S M L \
  --types R C M \
  --timings U P \
  --nr-repeats 3 \
  --soc-test normal \
  --objective closed \
  --solver-name gurobi \
  --solver-config-type gurobi

# multiple workdays, open and closed objective
for OBJECTIVE in open closed; do
  python3 run_workday_suite.py \
    --outdir "workdays_${OBJECTIVE}" \
    --seed "$SEED" \
    --sizes L \
    --types M \
    --timings U P \
    --base-rates 6 10 \
    --nr-surges 1 \
    --nr-repeats 3 \
    --objective "$OBJECTIVE" \
    --solver-name gurobi \
    --solver-config-type gurobi
done

# workdays comparison with plots
python3 compare_workday_objectives.py \
  --outdir workdays_comparison \
  --closed workdays_closed \
  --open workdays_open \
  --all-workdays
