
source ../.venv/bin/activate

python3 run_ablation_suite.py \
  --outdir ablation_suite \
  --request-counts 10 30 60 \
  --nr-repeats 1    # insertion ablation

python3 run_suite.py \
  --outdir single_plots \
  --sizes L \
  --types M \
  --timings P \
  --soc-test stress \
  --nr-repeats 1 \
  --solver-name gurobi \
  --solver-config-type gurobi \
  --plots           # single scenario with plots

python3 run_suite.py \
  --outdir single_suite \
  --soc-test normal \
  --solver-name gurobi \
  --solver-config-type gurobi   # single scenario suite

python3 run_workday_suite.py \
  --outdir workdays_open \
  --nr-surges 3 \
  --solver-name gurobi \
  --solver-config-type gurobi \
  --objective open        # multiple workdays, open objective

python3 run_workday_suite.py \
  --outdir workdays_closed \
  --nr-surges 3 \
  --solver-name gurobi \
  --solver-config-type gurobi \
  --objective closed      # multiple workdays, closed objective

python3 compare_workday_objectives.py \
  --outdir workdays_comparison \
  --closed workdays_closed \
  --open workdays_open \
  --all-workdays      # workdays comparison with plots
