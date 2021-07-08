#!/bash/bin
#Train CVRP20
#/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_attention_batch_rollout'
#NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp20_attention_batch_critic'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp20_attention_batch_exponential'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_attention_instance_rollout'
# #NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp20_attention_instance_critic'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp20_attention_instance_exponential'

# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_pointer_batch_rollout'
# #NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp20_pointer_batch_critic'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp20_pointer_batch_exponential'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_pointer_instance_rollout'
# #NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp20_pointer_instance_critic'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp20_pointer_instance_exponential'

#Train CVRP50
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp50_attention_batch_rollout'
#NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp50_attention_batch_critic'
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp50_attention_batch_exponential'
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp50_attention_instance_rollout'
#NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp50_attention_instance_critic'
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp50_attention_instance_exponential'

# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp50_pointer_batch_rollout'
# #NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp50_pointer_batch_critic'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp50_pointer_batch_exponential'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp50_pointer_instance_rollout'
# #NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp50_pointer_instance_critic'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp50_pointer_instance_exponential'

#Train CVRP100
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp100_attention_batch_rollout'
#NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp100_attention_batch_critic'
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp100_attention_batch_exponential'
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp100_attention_instance_rollout'
#NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp100_attention_instance_critic'
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp100_attention_instance_exponential'

# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp100_pointer_batch_rollout'
# #NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp100_pointer_batch_critic'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp100_pointer_batch_exponential'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp100_pointer_instance_rollout'
# #NOT SUPPORTED FOR VRP (TSP ONLY)/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline critic --run_name 'cvrp100_pointer_instance_critic'
# /bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 256 --epoch_size 128000 --model pointer --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline exponential --run_name 'cvrp100_pointer_instance_exponential'


