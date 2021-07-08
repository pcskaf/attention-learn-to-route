#!/bash/bin

#echo /home/thesisuser1/Documents/attention-learn-to-route/outputs/cvrp_20/cvrp20_attention_batch_rollout_20210429T142100/*
for i in {0..99..1}
do
	#echo /home/thesisuser1/Documents/attention-learn-to-route/outputs/cvrp_20/cvrp20_attention_batch_rollout_20210429T142100/epoch-$i.pt
	#/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/eval.py /home/thesisuser1/Documents/attention-learn-to-route/data/vrp/vrp20_validation_seed4321.pkl --model /home/thesisuser1/Documents/attention-learn-to-route/outputs/cvrp_20/cvrp20_attention_batch_rollout_20210429T142100/epoch-$i.pt --decode_strategy greedy
	/bin/python3.8 /home/thesisuser1/Documents/attention-learn-to-route/eval.py /home/thesisuser1/Documents/attention-learn-to-route/data/vrp/vrp20_validation_seed4321.pkl --model /home/thesisuser1/Documents/attention-learn-to-route/outputs/cvrp_20/cvrp20_attention_batch_rollout_20210429T142100/epoch-$i.pt --decode_strategy bs --width 10
done
