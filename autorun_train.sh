#!/bash/bin

cd /home/thesisuser1/Documents/results/

#Train CVRP20
#/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 512 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_attention_batch_rollout' > cvrp20_attention_batch_rollout.txt 
#/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 20 --batch_size 512 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_attention_instance_rollout' > cvrp20_attention_instance_rollout.txt

#Train CVRP50
#/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 512 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_attention_batch_rollout' > cvrp50_attention_batch_rollout.txt 
#/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 50 --batch_size 512 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_attention_instance_rollout' > cvrp50_attention_instance_rollout.txt 

#Train CVRP100
/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 100 --batch_size 256 --epoch_size 128000 --model attention --normalization batch --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp100_attention_batch_rollout' > cvrp100_attention_batch_rollout.txt 
#/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/run.py --problem cvrp --graph_size 100 --batch_size 512 --epoch_size 128000 --model attention --normalization instance --lr_model 1e-4 --lr_critic 1e-4 --n_epochs 100 --baseline rollout --run_name 'cvrp20_attention_instance_rollout' > cvrp100_attention_instance_rollout.txt 


