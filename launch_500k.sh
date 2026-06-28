#!/bin/bash
# MAV-Flow 500k offline training run on SMAC 3m Good.
# Wraps in caffeinate to prevent macOS idle sleep.

caffeinate -i python main.py --env_name=smac_3m_good --agent=agents/mav_flows.py --offline_steps=500000 --online_steps=0 --eval_interval=0 --save_interval=100000 --log_interval=1000 --enable_wandb=1 --wandb_mode=online --wandb_run_group=mav_flow_3m_good_500k --seed=0
