# First sync the updated src directory to /home/zhangbei/pymarl2/src.
# Run each command separately in its own terminal from ~/pymarl2.
# Do not run this whole file sequentially as a four-GPU launcher.

CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z use_tensorboard=True seed=79 name=audit_router_mix025_detachhidden_5m_3s5z_vs_3s6z_s79 audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=True audit_detach_mixed_hidden=True audit_log_agent_gradients=True counterfactual_mix_loss_weight=0.25 t_max=5050000 local_results_path=ablation_results_10_6

CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z use_tensorboard=True seed=81 name=audit_router_mix025_detachhidden_5m_3s5z_vs_3s6z_s81 audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=True audit_detach_mixed_hidden=True audit_log_agent_gradients=True counterfactual_mix_loss_weight=0.25 t_max=5050000 local_results_path=ablation_results_10_6

CUDA_VISIBLE_DEVICES=2 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z use_tensorboard=True seed=79 name=audit_router_mix025_detachhidden_5m_6h_vs_8z_s79 audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=True audit_detach_mixed_hidden=True audit_log_agent_gradients=True counterfactual_mix_loss_weight=0.25 t_max=5050000 local_results_path=ablation_results_10_6

CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z use_tensorboard=True seed=80 name=audit_router_mix025_detachhidden_5m_6h_vs_8z_s80 audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=True audit_detach_mixed_hidden=True audit_log_agent_gradients=True counterfactual_mix_loss_weight=0.25 t_max=5050000 local_results_path=ablation_results_10_6
