import re,sys
def norm(path):
    t=open(path,encoding='utf-8',errors='replace').read()
    t=t.split('*** START OF',1)[1].split('\n',1)[1].split('*** END OF',1)[0]
    m=re.search(r'\n\s*(FOOTNOTES|NOTES)[:.]?\s*\n',t)
    if m and m.start()>len(t)*0.6:t=t[:m.start()]
    t=re.sub(r'\[\d+\]','',t)
    out=[]
    for line in t.split('\n'):
        s=line.strip()
        m=re.match(r"^([A-Z][A-Z' ]{1,24}[A-Z])\.\s+(.*)$",s)
        if m:
            out.append(m.group(1).title().upper()+':');s=m.group(2).strip()
            if s:out.append(s)
            continue
        out.append(s)
    t='\n'.join(out)
    t=re.sub(r'\n{3,}','\n\n',t).encode('ascii','ignore').decode()
    i=t.find('ACT')
    return t[i:] if 0<=i<len(t)*0.3 else t
for p in sys.argv[1:]:
    open(p.replace('.txt','.norm.txt'),'w').write(norm(p))
