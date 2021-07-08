#!/bash/bin/
cd /home/thesisuser1/Documents/results/
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 100 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 40 --baseline rollout --run_name 'cvrp100_attention_batch_rollout' --resume /home/thesisuser1/Documents/results/outputs/cvrp_100/cvrp100_attention_batch_rollout_20210515T125901/epoch-61.pt > cvrp100_attention_batch_rollout_2.txt 

