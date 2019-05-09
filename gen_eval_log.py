from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import json

out_dir = "summary/mnist/"

for s in range(5):
	ea = event_accumulator.EventAccumulator(out_dir+str(s)+"/eval/")
	ea.Reload()
	s_acc = pd.DataFrame(ea.Scalars('accuracy'))['value'].max()
	min_diff = 1.0
	hyper = None
	with open(out_dir+"results.txt", "r") as f:
		for line in f:
			i_hyper, i_acc = line.split('},')
			i_acc = float(i_acc)
			i_hyper += '}'
			cur_diff = abs(s_acc - i_acc)
			if cur_diff < min_diff:
				hyper = i_hyper
				min_diff = cur_diff
	hyper = hyper.replace('\'', '"')
	hyper = json.loads(hyper)
	print(hyper)
