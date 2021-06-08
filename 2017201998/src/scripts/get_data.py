import os
import re
import json
import pickle
import math

def get_title(line,command):
    rep = ['\\textrm','\\textsf','\\texttt','\\rmfamily','\\sffamily','\\ttfamily','\\textup','\\textit','\\textsl','\\textsc','\\upshape','\\itshape','\\slshape','\\scshape','\\textmd','\\textbf','\\textnormal','\\text','\\bfseries','\\mdseries','{enumerate}','\\item','\\bf','\\b ','\\it','\\rm','\\tt','\\sl','\\sc','\\sf','\\large','\\LARGE','\\Large','\\huge','\\Huge','\\HUGE','\\small','\\tiny','\\emph','\\em','\\hl','\\hbox','\\fbox','\\mbox','\\normalfont','\\normalsize','\\titlefont','\\selectfont','\\mathrm','\\mathit','\\mathbf','\\mathsf','\\mathtt','\\mathcal','\\mathbb','\\mathfrak','\\mathscr','\\gls','\\glspl','\\Gls','\\Glspl','\\par','\\cal','quote','\\underline','\\xspace','\\medskip','\\bigskip','\\smallskip','\\unskip','\\qquad','\\quad','\\ensuremath','\\raggedright','\\raggedleft','\\centering','\\begin','\\end','{center}','{flushleft}','{flushright}','\\newline','\\noindent','\\indent','\\footnotesize','\\left','\\right','\\mid','\\hat','\\tilde','\\widetilde','\\bar','\\dots','\\acp','\\acl','\\ac','\\vec','\\boldmath','\\allowbreak','\\boldsymbol','\\var','\\rightarrow','\\leftarrow','\\sim','\\singlespacing']
    rep2 = ['[',']','}','{','`','$']
    dro = ['AAAI Press','My Publication','Proceedings','IJCAI--PRICAI','Conference','ACM']

    t = line.strip()
    #print("bef ",t)
    #t = t.lower()
    if t.find('\\thanks') != -1:
        t = t[:t.find('\\thanks')]
    if t.find('\\footnote') != -1:
        t = t[:t.find('\\footnote')]
                    
    t = re.sub(r'\\vspace{(.*?)}', '',t)
    t = re.sub(r'\\vspace\*{(.*?)}', '',t)
    t = re.sub(r'\\hspace{(.*?)}', '',t)
    t = re.sub(r'\\hspace\*{(.*?)}', '',t)
    t = re.sub(r'\\fontsize{(.*?)}{(.*?)}', '',t)
    t = re.sub(r'\\color{(.*?)}', '',t)
    t = re.sub(r'\\textcolor{(.*?)}', '',t)
    t = re.sub(r'\\ref{(.*?)}', '',t)
    t = re.sub(r'\\citealp{(.*?)}', '',t)
    t = re.sub(r'\\citealt{(.*?)}', '',t)
    t = re.sub(r'\\citep{(.*?)}', '',t)
    t = re.sub(r'\\citet{(.*?)}', '',t)
    t = re.sub(r'\\cite{(.*?)}', '',t)
    t = re.sub(r'\\tnoteref{(.*?)}', '',t)
    t = re.sub(r'\\tnotetext{(.*?)}', '',t)
    t = re.sub(r'\\looseness=-(\d+)', '',t)
    t = re.sub(r'\\baselineskip=(\d+)pt', '',t)
    
    for r in rep:
        t = t.replace(r,'')
    t = t.replace('~',' ')

    for d in dro:
        if d in t:
            t = ''
            break
    #print("aft ",t)
    if '\\' in t:
        t = replace_command(t,command)
    for r in rep2:
        t = t.replace(r,'')
    t = t.replace('\\',' ')
    return t.strip()

def get_abs(line,command):
    rep = ['\\textrm','\\textsf','\\texttt','\\rmfamily','\\sffamily','\\ttfamily','\\textup','\\textit','\\textsl','\\textsc','\\upshape','\\itshape','\\slshape','\\scshape','\\textmd','\\textbf','\\textnormal','\\text','\\bfseries','\\mdseries','{enumerate}','\\item','\\bf','\\b ','\\it','\\rm','\\tt','\\sl','\\sc','\\sf','\\large','\\LARGE','\\Large','\\huge','\\Huge','\\HUGE','\\small','\\tiny','\\emph','\\em','\\hl','\\hbox','\\fbox','\\mbox','\\normalfont','\\normalsize','\\titlefont','\\selectfont','\\mathrm','\\mathit','\\mathbf','\\mathsf','\\mathtt','\\mathcal','\\mathbb','\\mathfrak','\\mathscr','\\gls','\\glspl','\\Gls','\\Glspl','\\par','\\cal','quote','\\underline','\\xspace','\\medskip','\\bigskip','\\smallskip','\\unskip','\\qquad','\\quad','\\ensuremath','\\raggedright','\\raggedleft','\\centering','\\begin','\\end','{center}','{flushleft}','{flushright}','\\newline','\\noindent','\\indent','\\footnotesize','\\left','\\right','\\mid','\\hat','\\tilde','\\widetilde','\\bar','\\dots','\\acp','\\acl','\\ac','\\vec','\\boldmath','\\allowbreak','\\boldsymbol','\\var','\\rightarrow','\\leftarrow','\\sim','\\singlespacing']
    rep2 = ['[',']','}','{','`','$']
    #print('bef ', line)
    l = line.strip()

    if l.find('%') != -1 and l.find('\\%') == -1:
        l = l[:l.find('%')]
    
    l = re.sub(r'\\vspace{(.*?)}', '',l)
    l = re.sub(r'\\vspace\*{(.*?)}', '',l)
    l = re.sub(r'\\hspace{(.*?)}', '',l)
    l = re.sub(r'\\hspace\*{(.*?)}', '',l)
    l = re.sub(r'\\hskip (\d+)pt', '',l)
    l = re.sub(r'\\vskip (\d+)pt', '',l)
    l = re.sub(r'\\fontsize{(.*?)}{(.*?)}', '',l)
    l = re.sub(r'\\color{(.*?)}', '',l)
    l = re.sub(r'\\textcolor{(.*?)}', '',l)
    l = re.sub(r'\\ref{(.*?)}', '',l)
    l = re.sub(r'\\citealp{(.*?)}', '',l)
    l = re.sub(r'\\citealt{(.*?)}', '',l)
    l = re.sub(r'\\citep{(.*?)}', '',l)
    l = re.sub(r'\\citet{(.*?)}', '',l)
    l = re.sub(r'\\cite{(.*?)}', '',l)
    l = re.sub(r'\\footnote{(.*?)}', '',l)
    l = re.sub(r'\\blfootnote{(.*?)}', '',l)
    l = re.sub(r'\\tnoteref{(.*?)}', '',l)
    l = re.sub(r'\\tnotetext{(.*?)}', '',l)
    l = re.sub(r'\\thank{(.*?)}', '',l)
    l = re.sub(r'\\url{(.*?)}', 'url',l)
    l = re.sub(r'\\href{(.*?)}', 'url',l)
    l = re.sub(r'\\label{(.*?)}', '',l)
    l = re.sub(r'\\looseness=-(\d+)', '',l)
    l = re.sub(r'\\baselineskip=(\d+)pt', '',l)
    

    for r in rep:
        l = l.replace(r,'')
    l = l.replace('~',' ')
    if '\\' in l:
        l = replace_command(l,command)
    for r in rep2:
        l = l.replace(r,'')
    l = l.replace('\\',' ')
    #print('aft ', l)
    return l.strip()

def get_abs_file(abs_file,command):
    abstract = []
    f = open(abs_file, 'r', encoding = 'utf-8')
    for line in f.readlines():
        l = line.strip()
        if line.startswith('%'):
            continue
        if '\\keywords' in line or '\\Keywords' in line or 'keywords:' in line or 'Keywords:' in line or '{Keywords' in line or '{keywords' in line or 'Keywords}' in line or 'keywords}' in line:
            continue
        l = re.sub(r'abstract', '', l, flags=re.I)
        l = re.sub(r'\\begin', '', l, flags=re.I)
        l = re.sub(r'\\end', '', l, flags=re.I)
        l = re.sub(r'\\(.*?)section\*', '', l, flags=re.I)
        l = re.sub(r'\\(.*?)section', '', l, flags=re.I)
        l = get_abs(l,command)
        if l != '':
            abstract.append(l)
    return ' '.join(abstract)

def get_command(line):
    match = re.findall(r'\\newcommand(\*|){(.*)}(\[\d\]|){(.*)}', line)
    if len(match) != 0:
        k = match[0][1].strip()
        v = match[0][3].strip()
        return k,v
    match = re.findall(r'\\newcommand(.*?)(\[\d\]|){(.*)}', line)
    if len(match) != 0:
        k = match[0][0].strip()
        v = match[0][2].strip()
        return k,v
    match = re.findall(r'\\def(.*?){(.*)}', line)
    if len(match) != 0:
        k = match[0][0].strip()
        v = match[0][1].strip()
        return k,v
    return None,None

def replace_command(line,command):
    rep = ['\\textrm','\\textsf','\\texttt','\\rmfamily','\\sffamily','\\ttfamily','\\textup','\\textit','\\textsl','\\textsc','\\upshape','\\itshape','\\slshape','\\scshape','\\textmd','\\textbf','\\textnormal','\\text','\\bfseries','\\mdseries','\\bf','\\b ','\\it','\\rm','\\tt','\\sl','\\sc','\\sf','\\large','\\LARGE','\\Large','\\huge','\\Huge','\\HUGE','\\small','\\tiny','\\emph','\\em','\\hl','\\hbox','\\fbox','\\mbox','\\normalfont','\\normalsize','\\titlefont','\\selectfont','\\mathrm','\\mathit','\\mathbf','\\mathsf','\\mathtt','\\mathcal','\\mathbb','\\mathfrak','\\mathscr','\\gls','\\glspl','\\Gls','\\Glspl','\\par','\\cal','quote','\\underline','\\xspace','\\medskip','\\bigskip','\\smallskip','\\unskip','\\qquad','\\quad','\\ensuremath','\\raggedright','\\raggedleft','\\centering','\\begin','\\end','{center}','{flushleft}','{flushright}','\\newline','\\noindent','\\indent','\\footnotesize','\\left','\\right','\\mid','\\hat','\\tilde','\\widetilde','\\bar','\\dots','\\acp','\\acl','\\ac','\\vec','\\boldmath','\\allowbreak','\\boldsymbol','{enumerate}','\\item','\\var','\\rightarrow','\\leftarrow','\\sim','\\singlespacing']

    t = line.strip()
    #print("bef ",t)
    #t = t.lower()
    command_ = sorted(command.items(), key=lambda item:item[0], reverse=True)
    for c in command_:
        k,v = c
        t = t.replace(k,v)
    if t.find('\\thanks') != -1:
        t = t[:t.find('\\thanks')]
    if t.find('\\footnote') != -1:
        t = t[:t.find('\\footnote')]
                    
    t = re.sub(r'\\vspace{(.*?)}', '',t)
    t = re.sub(r'\\vspace\*{(.*?)}', '',t)
    t = re.sub(r'\\hspace{(.*?)}', '',t)
    t = re.sub(r'\\hspace\*{(.*?)}', '',t)
    t = re.sub(r'\\hskip (\d+)pt', '',t)
    t = re.sub(r'\\vskip (\d+)pt', '',t)
    t = re.sub(r'\\fontsize{(.*?)}{(.*?)}', '',t)
    t = re.sub(r'\\textcolor{(.*?)}', '',t)
    t = re.sub(r'\\color{(.*?)}', '',t)
    t = re.sub(r'\\ref{(.*?)}', '',t)
    t = re.sub(r'\\citealp{(.*?)}', '',t)
    t = re.sub(r'\\citealt{(.*?)}', '',t)
    t = re.sub(r'\\citep{(.*?)}', '',t)
    t = re.sub(r'\\citet{(.*?)}', '',t)
    t = re.sub(r'\\cite{(.*?)}', '',t)
    t = re.sub(r'\\footnote{(.*?)}', '',t)
    t = re.sub(r'\\blfootnote{(.*?)}', '',t)
    t = re.sub(r'\\tnoteref{(.*?)}', '',t)
    t = re.sub(r'\\tnotetext{(.*?)}', '',t)
    t = re.sub(r'\\thank{(.*?)}', '',t)
    t = re.sub(r'\\url{(.*?)}', 'url',t)
    t = re.sub(r'\\href{(.*?)}', 'url',t)
    t = re.sub(r'\\label{(.*?)}', '',t)
    t = re.sub(r'\\looseness=-(\d+)', '',t)
    t = re.sub(r'\\baselineskip=(\d+)pt', '',t)
    
    for r in rep:
        t = t.replace(r,'')
    t = t.replace('~',' ')

    return t.strip()

def filltering(datas):
    drop = 0
    datas_ = {}
    for k,v in datas.items():
        #print(k,v)
        t_num = len(v['title'])
        a_num = len(v['abstract'])
        if t_num == 0 or a_num == 0:
            drop += 1
            #print(k,v)
            continue
        elif t_num == 1:
            datas_[k] = {'title': v['title'][0]}
        else:
            #print(k,v)
            titles = []
            lens = []
            for title in v['title']:
                words = title.split(" ")
                if len(words) <= 20:
                    #titles.append(words)
                    titles.append(title)
                    lens.append(len(words))
                #print(len(words))
            max_index = lens.index(max(lens))
            datas_[k] = {'title': titles[max_index]}
            
        if a_num == 1:
            datas_[k].update({'abstract': v['abstract'][0]})
            #datas_[k] = {'abstract': v['abstract'][0]}
        else:
            lens = []
            for abstract in v['abstract']:
                lens.append(len(abstract))
            max_index = lens.index(max(lens))
            datas_[k].update({'abstract': v['abstract'][max_index]})
            #datas_[k] = {'abstract': v['abstract'][max_index]}
            #print('k   ', k)
            #print('abss ', v['abstract'])
            #print('abs  ', datas_[k])
        
    print('Drop ',drop)
    return datas_


def load_data(data_path):
    datas = {}

    for paper_name in os.listdir(data_path):
        #print(paper_name)
        paper_path = os.path.join(data_path, paper_name)
        papers = []
        for root, dirs, files in os.walk(paper_path):
            for name in files:
                paper = os.path.join(root, name)
                if paper.endswith('.tex'):
                    papers.append(paper)
        #print(papers)
        data = {}
        data['title'] = []
        data['abstract'] = []

        #get command
        command = {}
        for paper in papers:
            if 'sample' in paper or 'template' in paper or 'Template'in paper or 'IEEEtran' in paper or 'CvprRebuttal' in paper:
                continue
            f = open(paper, 'r', encoding = 'utf-8')
            for line in f.readlines():
                line = line.strip()
                if line.startswith('%'):
                    continue
                if line.find('%') != -1 and line.find('\\%') == -1:
                    line = line[:line.find('%')]
                line = line.strip()

                k,v = get_command(line)
                if k != None and k!= '' and '\\' in k:
                    command[k] = v
        
        for paper in papers:
            abst = 0
            tit = 0
            if 'sample' in paper or 'template' in paper or 'Template'in paper or 'IEEEtran' in paper or 'CvprRebuttal' in paper:
                continue
            f = open(paper, 'r', encoding = 'utf-8')
            for line in f.readlines():
                line = line.strip()
                if line.startswith('%'):
                    continue
                if line.find('%') != -1 and line.find('\\%') == -1:
                    line = line[:line.find('%')]
                line = line.strip()
                
                #get title
                t = ''
                match = re.findall(r'\\(acm|icml|aistats|mlsys|)title{(.*)', line)
                if len(match) != 0:
                    t = match[0][1].strip()
                else:
                    match = re.findall(r'\\(acm|icml|aistats|mlsys|)title\[(.*?)\]{(.*)', line)
                    if len(match) != 0:
                        t = match[0][2].strip()
                if t != '':
                    t_ = get_title(t,command)
                    if line.endswith('}') == False and '}.' not in line and '\\thanks' not in line:
                        title = []
                        title.append(t_)
                        tit = 1
                    else:
                        data['title'].append(t_)
                elif tit == 1:
                    t_ = get_title(line,command)
                    if t_ != '':
                        title.append(t_)
                    if '}' in line or t_ == '':
                        tit = 0
                        data['title'].append(' '.join(title))
                        title = []

                #get abstract
                if '\\keywords' in line or '\\Keywords' in line or 'keywords:' in line or 'Keywords:' in line or '{Keywords' in line or '{keywords' in line or 'Keywords}' in line or 'keywords}' in line:
                    continue
                if '\\input{' in line or '\\input{' in line:
                    match = re.findall(r'\\input\{(.*?)\}', line)
                    match2 = re.findall(r'abs', match[0], re.I)
                    if len(match2) != 0 and 'tabs' not in line and 'eabs' not in line and 'abso' not in line and 'tiks' not in line:
                        abs_file = paper_path + '/' + match[0]
                        if '.tex' not in abs_file:
                            abs_file = abs_file + '.tex'
                        a = get_abs_file(abs_file,command)
                        data['abstract'].append(a)

                elif 'begin{abstract}' in line or 'begin{Abstract}' in line or 'begin{ABSTRACT}' in line:
                    abst = 1
                    abstract = []
                elif 'end{abstract}' in line or 'end{Abstract}' in line or 'end{ABSTRACT}' in line:
                    abst = 0
                    if len(abstract) != 0:
                        data['abstract'].append(' '.join(abstract))
                elif abst == 2 and line == '':
                    abst = 0
                    data['abstract'].append(' '.join(abstract))
                elif 'abstract{' in line or 'Abstract{' in line or 'ABSTRACT{' in line or 'abstract}' in line or 'Abstract}' in line or 'ABSTRACT}' in line:
                    abst = 2
                    abstract = []
                if abst > 0:
                    if '\\begin' in line or '\\end' in line or 'section' in line or 'abstract{' in line or 'Abstract{' in line or 'ABSTRACT{' in line or 'abstract}' in line or 'Abstract}' in line or 'ABSTRACT}' in line:
                        l = re.sub(r'abstract', '', line, flags=re.I)
                    else:
                        l = line
                    l = re.sub(r'\\(.*?)section\*', '', l, flags=re.I)
                    l = re.sub(r'\\(.*?)section', '', l, flags=re.I)
                    l = re.sub(r'\\begin', '', l, flags=re.I)
                    l = re.sub(r'\\end', '', l, flags=re.I)
                    l = get_abs(l,command)
                    if l != '':
                        abstract.append(l)

            datas[paper_name] = data
            #print(datas)

    datas_ = filltering(datas)
    #print(datas_)
    
    d = json.dumps(datas_)
    f = open('ta_data.json', 'w') 
    f.write(d)
    f.close()
    '''
    fout = open('abstarct', 'w', encoding='utf-8')
    for k,v in datas_.items():
        print(k," ",v,"\n",end='',file=fout)
    fout.close()
    '''

load_data('ori_data/')
