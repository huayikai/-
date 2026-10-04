# Run each command in its own terminal from the server repository root.

CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_bm --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z use_tensorboard=True seed=81 name=audit_bm_td_3s5z_vs_3s6z_s81 audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=False

CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z use_tensorboard=True seed=81 name=audit_router_aux_3s5z_vs_3s6z_s81 audit_fix_td_lambda=False audit_isolate_dvd_aux_hidden=True

CUDA_VISIBLE_DEVICES=2 python3 src/main_audit.py --config=dvd_audit_bm --env-config=sc2 with env_args.map_name=6h_vs_8z use_tensorboard=True seed=79 name=audit_bm_td_6h_vs_8z_s79 audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=False

CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z use_tensorboard=True seed=79 name=audit_router_both_6h_vs_8z_s79 audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=True
