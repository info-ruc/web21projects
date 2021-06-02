import os
import pickle

class Metric(object):
	"""Base metric class.
    Subclasses must override score() and can optionally override various
    other methods.
    """
	def __init__(self):
		print("Generating the entropy dictionary.")
		# f = open('C:/Users/Gsw/Desktop/entropy.txt')
		with open('/home/yutong_bai/doc_profile/DocPro/data/entropy.dict', 'rb') as fr:
			self.entropy = pickle.load(fr)
			
	def rerank(self, scores, lines):
		idxs = list(zip(*sorted(enumerate(scores), key=lambda x:x[1], reverse=True)))[0]
		final_lines = []
		for idx in idxs:
			final_lines.append(lines[idx])
		return final_lines

	def score(self, clicks):
		raise NotImplementedError()

	def evaluate(self, filehandle):
		"""
		evaluate queries in the filehadle, each line is the same as the training dataset.
		"""
		last_queryid = 0
		f = open(self.__class__.__name__+'.txt', 'w')
		re = open('results.txt','a')
		mscore1, mscore2 = 0.0, 0.0
		nquery = 0.0
		clicks = []
		scores = []
		for line in filehandle:
			user, sessionid, querytime, query, url, title, sat, _,score = line.strip().split('\t')
			queryid = user + sessionid + querytime + query

			if queryid != last_queryid: # 表示一个query结束了
				if len(clicks) == 50:
					score1 = self.score(clicks)
					score2 = self.score(self.rerank(scores, clicks))
					if score1 != -1:
						nquery += 1
						mscore1 += score1
						mscore2 += score2
					f.write(last_queryid+'\t'+str(self.entropy[query])+'\t'+
						'\t'+str(score1)+'\t'+str(score2)+'\n')
				clicks = []
				scores = []
				last_queryid = queryid
			clicks.append(sat)
			scores.append(float(score))
		if len(clicks) != 0 and len(clicks) != 1:
			score1 = self.score(clicks)
			score2 = self.score(self.rerank(scores, clicks))
			if score1 != -1:
				nquery += 1
				mscore1 += score1
				mscore2 += score2
			f.write(last_queryid+'\t'+str(self.entropy[query])+'\t'+
				'\t'+str(score1)+'\t'+str(score2)+'\n')
		f.close()
		print("The "+self.__class__.__name__+" of original ranking is {}.".format(mscore1/nquery))
		print("The "+self.__class__.__name__+" of new ranking is {}.".format(mscore2/nquery))
		re.write("The "+self.__class__.__name__+" of original ranking is {}.\n".format(mscore1/nquery))
		re.write("The "+self.__class__.__name__+" of new ranking is {}.\n".format(mscore2/nquery))



	def write_score(self, scores, lines, filehandle):
		assert(len(scores[0])==len(lines))
		for i in range(len(scores[0])):
			filehandle.write(lines[i].rstrip('\n')+'\t'+str(scores[0][i][0])+'\n')

class AP(Metric):
	# 平均正确率(Average Precision)：对不同召回率点上的正确率进行平均。
	def  __init__(self, cutoff='1'):
		super(AP, self).__init__()
		self.cutoff = cutoff

	def score(self, clicks):
		num_rel = 0
		total_prec = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
				num_rel += 1
				total_prec += num_rel / (i + 1.0)
		return (total_prec / num_rel) if num_rel > 0 else -1


class MRR(Metric):
	def __init__(self, cutoff='1'):
		super(MRR, self).__init__()
		self.cutoff = cutoff

	def score(self, clicks):
		num_rel = 0
		total_prec = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff:
				num_rel = 1
				total_prec = 1.0 / (i+1)
				break
		return total_prec if num_rel > 0 else -1


class Precise(Metric):
	def __init__(self, cutoff='1', k=1):
		super(Precise, self).__init__()
		self.cutoff = cutoff
		self.k = k

	def score(self, clicks):
		prec_in = 0.0
		prec_out = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff:
				if i+1 <= self.k:
					prec_in = 1
					break
				else:
					prec_out = 1
		if prec_in > 0:
			return 1
		else:
			return 0 if prec_out > 0 else -1

class AvePosition(Metric):
	def __init__(self, cutoff='1'):
		super(AvePosition, self).__init__()
		self.cutoff = cutoff

	def score(self, clicks):
		position = 0.0
		nclick = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff:
				position += i+1
				nclick += 1
		return (position/nclick) if nclick > 0 else -1


class Pgain(Metric):
	def __init__(self, cutoff='1'):
		super(Pgain, self).__init__()
		self.cutoff = cutoff

	def score(self, clicks):
		num_rel = 0
		total_prec = 0.0
		for i in range(len(clicks)):
			if clicks[i] == self.cutoff:
				num_rel += 1
				total_prec += num_rel / (i + 1.0)
		return (total_prec / num_rel) if num_rel > 0 else -1

	def evaluate(self, filehandle):
		last_queryid = 0
		last_query = ''
		last_sc = 0
		f = open(self.__class__.__name__+'.txt', 'w')
		nup = 0.0
		ndown = 0.0
		nquery = 0.0
		clicks = []
		scores = []
		for line in filehandle:
			# query, queryid, sessionid, date, url, sat, urlrank, score = line.rstrip().split('\t')
			query, queryid, sessionid, date, url, sat, urlrank, sc , score= line.rstrip().split('\t')
			if queryid != last_queryid:
				if len(clicks) != 0:
					score1 = self.score(clicks)
					score2 = self.score(self.rerank(scores, clicks))
					if score1 > score2:
						ndown += 1
						f.write(last_queryid+'\t'+str(self.entropy[last_query.lower()])+'\t'+last_sc+'\t1'+'\n')
					elif score1 < score2:
						nup += 1
						f.write(last_queryid+'\t'+str(self.entropy[last_query.lower()])+'\t'+last_sc+'\t0'+'\n')
					clicks = []
					scores = []
				last_queryid = queryid
				last_query = query
				last_sc = sc
			clicks.append(sat)
			scores.append(float(score))
		print("The number of better rankings is {}.".format(nup))
		print("The number of worse rankings is {}.".format(ndown))
		print("The Pgain is {}.".format((nup-ndown)/(nup+ndown)))
