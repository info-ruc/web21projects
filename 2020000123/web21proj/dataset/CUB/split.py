import csv

for mode in ['train', 'val', 'test']:
	in_path = mode + '_ori.csv'
	out_path = mode + '.csv'
	f_in = open(in_path, 'r')
	f_out = open(out_path, 'w', newline='')
	reader = csv.reader(f_in)
	writer = csv.writer(f_out)
	
	for i, row in enumerate(reader):
		if i == 0:
			out_row = row[:2]
			writer.writerow(out_row)
		else:
			out_row = []
			out_row.append(row[1] + '/' + row[0][:-3] + 'png')
			out_row.append(row[1])
			writer.writerow(out_row)
