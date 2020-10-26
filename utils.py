from main import Dataset, Labels
import math
import os

def print_matrix(m):
	s = [[str(m[l][ll]) for ll in Labels] for l in Labels]
	lens = [max(map(len, col)) for col in zip(*s)]
	fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
	table = [fmt.format(*row) for row in s]
	print('\n'.join(table))

def print_metrics(m):
	mlen = max([len(l.name) for l in Labels])
	for k, v in m.items():
		print('\t{}\t{}'.format(k.name+':'+' '*(mlen-len(k.name)), v))

def evaluate(model, ds):
	# confusion matrix
	cm = {l: {ll: 0 for ll in Labels} for l in Labels}
	for ind in ds.index:
		x = ds['REVIEW_TEXT'][ind].split(' ') #array
		#t = ds['LABEL'][ind] #string
		t = Labels.label1 if(ds['LABEL'][ind] == "__label1__") else Labels.label2

		y = model.predict(x)
		cm[y][t] += 1

	print("Confusion Matrix")
	print_matrix(cm)

	# precision, recall and f1
	sum_row  = {l: sum(cm[l].values()) for l in Labels}
	sum_col = {l: 0 for l in Labels}
	for l in Labels:
		for ll in Labels:
			sum_col[l] += cm[ll][l]

	p = {l: cm[l][l] / (sum_row[l] if sum_row[l] else 1) for l in Labels}
	r = {l: cm[l][l] / (sum_col[l] if sum_col[l] else 1) for l in Labels}
	f1 = {l: 2*p[l]*r[l]/(p[l]+r[l]) if (p[l]+r[l]) else 0 for l in Labels}

	print("\nPrecision")
	print_metrics(p)
	print("\nRecall")
	print_metrics(r)
	print("\nF1")
	print_metrics(f1)

	acc = sum([cm[l][l] for l in Labels]) / len(ds)
	print("\nAccuracy:\t{}".format(acc))
	print("Precision:\t{}".format(sum(p.values())/len(p.values())))
	print("Recall:\t\t{}".format(sum(r.values())/len(r.values())))
	print("F1:\t\t\t{}".format(sum(f1.values())/len(f1.values())))
