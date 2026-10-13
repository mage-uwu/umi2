/*
 * umi_chat.c -- umi-1M-chat: a multi-turn retrieval chatbot on the nanotss MLM.
 *
 * There is no autoregressive generation. Every reply is a cached answer
 * written by DeepSeek (gen_data.py, gen_convs.py). The bidirectional
 * masked-LM reads the conversation state and the user's message and fills a
 * three-byte answer-ID slot, a typed decision about what to retrieve:
 *
 *   [ctx1 a b t][ctx2 a b t] GS <message> SEP [MASK][MASK][MASK] END PAD...
 *
 * ctx1/ctx2 are the IDs of umi's last two replies ("---" when there are
 * none). IDs are hierarchical: single-turn intents are (domain, intent, 0);
 * turns of a DeepSeek conversation tree are (conv_hi, conv_lo, turn), so a
 * follow-up shares its first two bytes with the reply it follows and the
 * model can copy them, then choose the next turn from the message.
 *
 * Retrieval is two-stage. A large hashed char n-gram cache (TF-IDF over every
 * answer's phrasings) proposes candidates, plus every turn of the
 * conversation trees in context. The MLM then makes the typed decision: each
 * candidate's ID is scored byte by byte through the masked slot (reveal a to
 * score b, reveal a,b to score t), and log p + beta * cosine + gamma *
 * [candidate directly follows umi's last reply in its cached tree] picks it.
 * classify() is the MLM-only beam over all IDs, used for training metrics.
 * Below --threshold confidence the fallback answer is retrieved.
 *
 * User names: a name in the user's text is replaced by the tag
 * <regex>user_name</regex> before the model sees it (training data uses the
 * same form), and the same tag inside a cached answer is spawned back in from
 * the remembered name. In BLT mode (default) the byte encoder also reads a
 * large hashed 3..8-byte n-gram cache (--ngram-bins), excluded from the ~1M
 * budget as upstream does.
 *
 * umi-smart-internet: with --web DIR --web-load F a second model, trained the same
 * way on DeepSeek's widget cache (gen_web.py, prepare_web.py), picks an HTML
 * widget, container or dashboard for each turn (or "plain"), and --html OUT
 * compiles the conversation into a page: umi's own visual interface.
 *
 * Build:  cc -O3 -march=native -fopenmp umi_chat.c -lm -o umi_chat
 * Train:  ./umi_chat train --data data --seconds 7200 --threads 8 --save umi.bin
 * Eval:   ./umi_chat eval  --data data --load umi.bin
 * Chat:   ./umi_chat chat  --data data --load umi.bin [-v] [--ask "a|b|c"]
 *         [--web data_web --web-load web.bin --html page.html]
 */
#define NANOTSS_NO_MAIN
#include "../nanotss_blt_mlm.c"
#include <regex.h>
#include <ctype.h>

static const char ALPHA[]="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/";
enum{NA=240,NB=128,NT=64,SEQ=96,CTX=6,MAXMSG=SEQ-CTX-6,GS=0x1D,SEP=0x1F,END=0x1E,PAD=0,NONE='-',NV=4,KA=4,KB=3};
static const char TAG[]="<regex>user_name</regex>";
/* Byte for each ID part, from one table: the 64 printable symbols, then 0x80..0xFF, then the remaining
 * low bytes that are not layout markers (251 in all). Earlier, smaller ID spaces (a < 192, b and t < 64)
 * map to the same bytes, so older checkpoints keep working. */
static unsigned char IDB[256];
static int idbyte(int x){
    if(!IDB[0]){int k=0;for(int i=0;i<64;i++)IDB[k++]=ALPHA[i];for(int c=0x80;c<0x100;c++)IDB[k++]=c;
        for(int c=1;c<0x80;c++)if(!strchr(ALPHA,c)&&c!=GS&&c!=SEP&&c!=END&&c!=NONE)IDB[k++]=c;}
    return IDB[x];
}
static int abyte(int a){return idbyte(a);}
static int bbyte(int b){return idbyte(b);}

typedef struct{int id,c1,c2;char*msg;}Ex;               /* ids packed as a*4096+b*64+t, -1 = none */
typedef struct{int id,parent,nv,last;char*name,*ans[NV],*ano[NV];}Ans;
typedef struct{int*off,*ans;float*w,*idf;int n;}NgIdx;
/* One retrieval cache: the answers, their ID index and the n-gram index. umi keeps two, the chat
 * cache and the umi-smart-internet widget cache, and C points at the one in use. */
typedef struct{Ans*answers;int n_answers,*ans_at;unsigned char has_a[NA],has_ab[NA][NB];int intro_id,fallback_id,plain_id,strict_tree;NgIdx ng;}Cache;
/* strict_tree (set for the widget cache, which has a "plain" class): a non-root conversation turn
 * may only follow its own parent, because a widget refinement changes *that* widget */
static Cache cache_chat={.intro_id=-1,.fallback_id=-1,.plain_id=-1},cache_web={.intro_id=-1,.fallback_id=-1,.plain_id=-1},*C=&cache_chat;
#define answers (C->answers)
#define n_answers (C->n_answers)
#define ans_at (C->ans_at)
#define has_a (C->has_a)
#define has_ab (C->has_ab)
#define intro_id (C->intro_id)
#define fallback_id (C->fallback_id)
#define plain_id (C->plain_id)
#define strict_tree (C->strict_tree)
#define ng (C->ng)
#define PACK(a,b,t) ((a)*NB*NT+(b)*NT+(t))
#define ID_A(x) ((x)/(NB*NT))
#define ID_B(x) ((x)/NT%NB)
#define ID_T(x) ((x)%NT)

static int parse_id(const char*s){int a,b,t;if(*s=='-'||sscanf(s,"%d.%d.%d",&a,&b,&t)!=3||a<0||a>=NA||b<0||b>=NB||t<0||t>=NT)return -1;return PACK(a,b,t);}
static Ex*load_ex(const char*dir,const char*name,int*n){
    char p[1024];snprintf(p,sizeof p,"%s/%s",dir,name);FILE*fp=fopen(p,"r");if(!fp){fprintf(stderr,"cannot open %s\n",p);exit(1);}
    int cap=4096;Ex*e=malloc(cap*sizeof(Ex));*n=0;char line[4096];
    while(fgets(line,sizeof line,fp)){
        line[strcspn(line,"\r\n")]=0;char*f[4];int k=0;f[k++]=line;
        for(char*q=line;*q&&k<4;q++)if(*q=='\t'){*q=0;f[k++]=q+1;}
        if(k<4||!*f[3])continue;int id=parse_id(f[0]);if(id<0)continue;
        if(*n==cap)e=realloc(e,(cap*=2)*sizeof(Ex));e[*n]=(Ex){id,parse_id(f[1]),parse_id(f[2]),strdup(f[3])};(*n)++;
    }fclose(fp);return e;
}
static void load_answers(const char*dir){
    char p[1024];snprintf(p,sizeof p,"%s/answers.tsv",dir);FILE*fp=fopen(p,"r");if(!fp){fprintf(stderr,"cannot open %s\n",p);exit(1);}
    ans_at=malloc((size_t)NA*NB*NT*sizeof(int));for(int i=0;i<NA*NB*NT;i++)ans_at[i]=-1;
    int cap=4096;answers=calloc(cap,sizeof(Ans));char*line=malloc(1<<16);
    while(fgets(line,1<<16,fp)){
        line[strcspn(line,"\r\n")]=0;char*f[5];int k=0;f[k++]=line;
        for(char*q=line;*q&&k<5;q++)if(*q=='\t'){*q=0;f[k++]=q+1;}
        if(k<4)continue;int id=parse_id(f[0]);if(id<0)continue;
        if(ans_at[id]>=0){Ans*r=answers+ans_at[id];if(r->nv<NV){r->ans[r->nv]=strdup(f[2]);r->ano[r->nv++]=strdup(f[3]);}continue;}
        if(n_answers==cap)answers=realloc(answers,(cap*=2)*sizeof(Ans));
        Ans*r=answers+n_answers;*r=(Ans){.id=id,.parent=k>4?parse_id(f[4]):-1,.nv=1,.last=-1,.name=strdup(f[1])};r->ans[0]=strdup(f[2]);r->ano[0]=strdup(f[3]);ans_at[id]=n_answers++;
        has_a[ID_A(id)]=1;has_ab[ID_A(id)][ID_B(id)]=1;
        if(!strcmp(f[1],"umi_core/user_introduces_name"))intro_id=id;
        if(!strcmp(f[1],"umi_core/fallback_unknown"))fallback_id=id;
        if(!strncmp(f[1],"plain/",6))plain_id=id; /* widget cache: "no widget" */
    }fclose(fp);free(line);
    strict_tree=plain_id>=0;
}

/* ---- model wrapper: BLT (Faithful) or direct byte core (Model) ---- */
typedef struct{Faithful*f;Model*m;int threads;}Net;
static int net_mask(const Net*n){return (n->f?n->f->global->c.vocab:n->m->c.vocab)-1;}
static void net_logits(Net*n,int th,const int*tok,real*lg256){
    int ns,y[SEQ];for(int i=0;i<SEQ;i++)y[i]=-1;
    if(n->f){faithful_eval(n->f,th,tok,y,SEQ,&ns,lg256);return;}
    const int V=n->m->c.vocab;real*tmp=malloc((size_t)SEQ*V*sizeof(real));eval_seq(n->m,tok,y,SEQ,n->m->ws[th],&ns,tmp);
    for(int i=0;i<SEQ;i++)memcpy(lg256+(size_t)i*256,tmp+(size_t)i*V,256*sizeof(real));free(tmp);
}
static void put_id(int*tok,int id){
    if(id<0){tok[0]=tok[1]=tok[2]=NONE;return;}
    tok[0]=abyte(ID_A(id));tok[1]=bbyte(ID_B(id));tok[2]=bbyte(ID_T(id));
}
/* Lay out one example; returns the position of the first answer slot. reveal = number of
 * leading answer bytes shown (teacher forcing for the beam); aux>0 adds BERT noise on the message. */
static int encode(const char*msg,int c1,int c2,int id,int reveal,int mask,double aux,uint64_t*rng,int*tok,int*lab){
    int len=(int)strlen(msg);if(len>MAXMSG)len=MAXMSG;
    for(int i=0;i<SEQ;i++){tok[i]=PAD;lab[i]=-1;}
    put_id(tok,c1);put_id(tok+3,c2);tok[CTX]=GS;int o=CTX+1;
    for(int i=0;i<len;i++){
        tok[o+i]=(unsigned char)msg[i];
        if(aux>0&&urand(rng)<aux){lab[o+i]=tok[o+i];real u=urand(rng);if(u<(real).8)tok[o+i]=mask;else if(u<(real).9)tok[o+i]=(int)(urand(rng)*256)&255;}
    }
    int s=o+len;tok[s]=SEP;s++;
    int want[3]={-1,-1,-1};if(id>=0){want[0]=abyte(ID_A(id));want[1]=bbyte(ID_B(id));want[2]=bbyte(ID_T(id));}
    for(int j=0;j<3;j++){if(j<reveal){tok[s+j]=want[j];}else{tok[s+j]=mask;lab[s+j]=want[j];}}
    tok[s+3]=END;return s;
}
static double lsm(const real*l,int v){double mx=-1e300,z=0;for(int i=0;i<256;i++)mx=l[i]>mx?l[i]:mx;for(int i=0;i<256;i++)z+=exp(l[i]-mx);return l[v]-mx-log(z);}
/* top-k indices of p[0..n) with p>0 */
static int topk(const double*p,int n,int k,int*out){int m=0;for(;m<k;m++){int b=-1;for(int i=0;i<n;i++){int used=0;for(int j=0;j<m;j++)used|=out[j]==i;if(!used&&p[i]>0&&(b<0||p[i]>p[b]))b=i;}if(b<0)break;out[m]=b;}return m;}

/* Retrieval decision: returns the packed answer ID; *conf is its joint probability. */
static int classify(Net*n,int th,const char*msg,int c1,int c2,double*conf){
    int tok[SEQ],lab[SEQ],mask=net_mask(n);real*lg=malloc((size_t)SEQ*256*sizeof(real));
    int s=encode(msg,c1,c2,-1,0,mask,0,NULL,tok,lab);net_logits(n,th,tok,lg);
    double pa[NA],z=0;for(int a=0;a<NA;a++){pa[a]=has_a[a]?exp(lsm(lg+(size_t)s*256,abyte(a))):0;z+=pa[a];}
    for(int a=0;a<NA;a++)pa[a]/=z;
    int ta[KA],na=topk(pa,NA,KA,ta),best=-1;double bestp=-1;
    for(int i=0;i<na;i++){
        int a=ta[i];tok[s]=abyte(a);tok[s+1]=tok[s+2]=mask;net_logits(n,th,tok,lg);
        double pb[NB],zb=0;for(int b=0;b<NB;b++){pb[b]=has_ab[a][b]?exp(lsm(lg+(size_t)(s+1)*256,bbyte(b))):0;zb+=pb[b];}
        int tb[KB],nb=topk(pb,NB,KB,tb);
        for(int j=0;j<nb;j++){
            int b=tb[j];double pab=pa[a]*pb[b]/zb;if(pab<=bestp)continue;
            int nt=0,only=-1;for(int t=0;t<NT;t++)if(ans_at[PACK(a,b,t)]>=0){nt++;only=t;}
            if(nt==1){bestp=pab;best=PACK(a,b,only);continue;}
            tok[s+1]=bbyte(b);net_logits(n,th,tok,lg);
            double pt[NT],zt=0;for(int t=0;t<NT;t++){pt[t]=ans_at[PACK(a,b,t)]>=0?exp(lsm(lg+(size_t)(s+2)*256,bbyte(t))):0;zt+=pt[t];}
            for(int t=0;t<NT;t++)if(pt[t]>0&&pab*pt[t]/zt>bestp){bestp=pab*pt[t]/zt;best=PACK(a,b,t);}
        }
    }
    free(lg);*conf=bestp;return best;
}

/* ---- the n-gram cache: hashed char 3..5-gram TF-IDF index over every answer's phrasings ----
 * It proposes candidates; the MLM's masked answer slot makes the final, typed decision. */
enum{NGB=1<<20,NGK=16};
static int grams(const char*msg,uint32_t*out){
    char buf[512];int L=snprintf(buf,sizeof buf," %s ",msg);if(L>(int)sizeof buf-1)L=sizeof buf-1;int m=0;
    for(int n=3;n<=5;n++)for(int i=0;i+n<=L;i++){uint32_t h=2166136261u^n;for(int j=0;j<n;j++)h=(h^(unsigned char)buf[i+j])*16777619u;out[m++]=h&(NGB-1);}
    /* dedupe with a small open-addressing set on the stack (linear time, thread-safe) */
    int cap=64;while(cap<2*m)cap<<=1;uint32_t set[cap];memset(set,0xff,cap*sizeof(uint32_t));int u=0;
    for(int i=0;i<m;i++){uint32_t v=out[i],h=(v*2654435761u)&(cap-1);
        while(set[h]!=0xffffffffu&&set[h]!=v)h=(h+1)&(cap-1);
        if(set[h]==0xffffffffu){set[h]=v;out[u++]=v;}}
    return u;
}
/* Counting sort by bucket. Messages are visited in answer order, so each bucket's list comes out
 * already sorted by answer and equal answers are adjacent: no comparison sort is needed. */
static void ngram_build(Ex*e,int ne){
    uint32_t g[1600];int*cnt=calloc(NGB+1,sizeof(int)),*byans=calloc(n_answers+1,sizeof(int)),*ord=malloc(ne*sizeof(int)),no=0;
    for(int i=0;i<ne;i++)if(ans_at[e[i].id]>=0)byans[ans_at[e[i].id]+1]++;
    for(int a=0;a<n_answers;a++)byans[a+1]+=byans[a];
    for(int i=0;i<ne;i++)if(ans_at[e[i].id]>=0){ord[byans[ans_at[e[i].id]]++]=i;no++;}
    free(byans);
    for(int q=0;q<no;q++){int i=ord[q];int m=grams(e[i].msg,g);for(int j=0;j<m;j++)cnt[g[j]+1]++;}
    for(int b=0;b<NGB;b++)cnt[b+1]+=cnt[b];
    size_t np=cnt[NGB];int*list=malloc((np+1)*sizeof(int)),*fill=malloc(NGB*sizeof(int));memcpy(fill,cnt,NGB*sizeof(int));
    for(int q=0;q<no;q++){int i=ord[q],a=ans_at[e[i].id];int m=grams(e[i].msg,g);for(int j=0;j<m;j++)list[fill[g[j]]++]=a;}
    free(fill);free(ord);
    ng.off=calloc(NGB+1,sizeof(int));ng.idf=calloc(NGB,sizeof(float));ng.ans=malloc((np+1)*sizeof(int));ng.w=malloc((np+1)*sizeof(float));
    double*norm=calloc(n_answers,sizeof(double));size_t k=0;
    for(int b=0;b<NGB;b++){
        int s0=cnt[b],s1=cnt[b+1];ng.off[b]=(int)k;if(s1==s0)continue;
        size_t k0=k;
        for(int i=s0;i<s1;){int j=i;while(j<s1&&list[j]==list[i])j++;ng.ans[k]=list[i];ng.w[k]=(float)(j-i);k++;i=j;}
        ng.idf[b]=logf((float)n_answers/(float)(k-k0));
        for(size_t q=k0;q<k;q++){ng.w[q]=(1+logf(ng.w[q]))*ng.idf[b];norm[ng.ans[q]]+=(double)ng.w[q]*ng.w[q];}
    }
    ng.off[NGB]=(int)k;
    for(size_t q=0;q<k;q++)ng.w[q]/=(float)sqrt(norm[ng.ans[q]]+1e-9);
    free(list);free(cnt);free(norm);ng.n=n_answers;
}
/* top-k answers by cosine similarity; returns count, fills answer indices and scores.
 * If all is non-NULL it receives the cosine of every answer (caller frees). */
static int ngram_top(const char*msg,int k,int*out,float*sc,float**all){
    uint32_t g[1600];int m=grams(msg,g);float*s=calloc(ng.n,sizeof(float));double qn=0;
    for(int j=0;j<m;j++){int b=g[j];if(ng.off[b+1]-ng.off[b]>ng.n/8)continue; /* stop-grams */
        float q=ng.idf[b];qn+=q*q;for(int p=ng.off[b];p<ng.off[b+1];p++)s[ng.ans[p]]+=q*ng.w[p];}
    qn=sqrt(qn+1e-9);int c=0;
    for(int a=0;a<ng.n;a++){float v=s[a]/=qn;if(c<k||v>sc[c-1]){int j=c<k?c++:k-1;while(j>0&&sc[j-1]<v){sc[j]=sc[j-1];out[j]=out[j-1];j--;}sc[j]=v;out[j]=a;}}
    if(all)*all=s;else free(s);return c;
}
/* MLM log-probability of each candidate ID: p(a) p(b|a) p(t|a,b), each slot renormalised over IDs that exist */
static void mlm_score(Net*n,int th,const char*msg,int c1,int c2,const int*ids,int nc,double*lp){
    int tok[SEQ],lab[SEQ],mask=net_mask(n);real*lg=malloc((size_t)SEQ*256*sizeof(real));
    int s=encode(msg,c1,c2,-1,0,mask,0,NULL,tok,lab);net_logits(n,th,tok,lg);
    double la[NA],z=0;for(int a=0;a<NA;a++)z+=has_a[a]?exp(lsm(lg+(size_t)s*256,abyte(a))):0;
    for(int a=0;a<NA;a++)la[a]=has_a[a]?lsm(lg+(size_t)s*256,abyte(a))-log(z):-1e9;
    double top=-1e9;for(int i=0;i<nc;i++){lp[i]=la[ID_A(ids[i])];top=fmax(top,lp[i]);}
    for(int i=0;i<nc;i++){
        int a=ID_A(ids[i]);int seen=0;for(int j=0;j<i;j++)seen|=ID_A(ids[j])==a;if(seen)continue;
        if(la[a]<top-6){for(int j=i;j<nc;j++)if(ID_A(ids[j])==a)lp[j]=-1e9;continue;} /* pruned: p(a) < 1/400 of the best */
        tok[s]=abyte(a);tok[s+1]=tok[s+2]=mask;net_logits(n,th,tok,lg);
        double zb=0;for(int b=0;b<NB;b++)if(has_ab[a][b])zb+=exp(lsm(lg+(size_t)(s+1)*256,bbyte(b)));
        for(int j=i;j<nc;j++)if(ID_A(ids[j])==a)lp[j]+=lsm(lg+(size_t)(s+1)*256,bbyte(ID_B(ids[j])))-log(zb);
        for(int j=i;j<nc;j++){
            if(ID_A(ids[j])!=a)continue;int b=ID_B(ids[j]),dup=0;for(int k=i;k<j;k++)dup|=ID_A(ids[k])==a&&ID_B(ids[k])==b;if(dup)continue;
            int nt=0;for(int t=0;t<NT;t++)nt+=ans_at[PACK(a,b,t)]>=0;if(nt<2)continue;
            tok[s+1]=bbyte(b);net_logits(n,th,tok,lg);tok[s+1]=mask;
            double zt=0;for(int t=0;t<NT;t++)if(ans_at[PACK(a,b,t)]>=0)zt+=exp(lsm(lg+(size_t)(s+2)*256,bbyte(t)));
            for(int k=j;k<nc;k++)if(ID_A(ids[k])==a&&ID_B(ids[k])==b)lp[k]+=lsm(lg+(size_t)(s+2)*256,bbyte(ID_T(ids[k])))-log(zt);
        }
    }
    free(lg);
}
static double hy_beta=8,hy_gamma=4,root_bonus=1; /* root_bonus: prefer widgets that open a refinable conversation */ /* gamma: prior for a turn that directly follows the last reply */static int eval_n=0; /* eval: examples per half, 0 = all */ /* weight of the n-gram cosine against the MLM log-probability */
/* Hybrid retrieval: candidates = n-gram top-K plus every turn of the conversations in context;
 * decision = MLM log-prob + beta * cosine. beta<0 means n-gram only. Returns packed ID. */
static int decide(Net*n,int th,const char*msg,int c1,int c2,double beta,double gamma,double*conf,double*pplain){
    int ids[NGK+2*NT+1],na[NGK];float sc[NGK],*cos_all;int k=ngram_top(msg,NGK,na,sc,&cos_all),nc=0;double cs[NGK+2*NT+1];
    for(int i=0;i<k;i++){Ans*a=answers+na[i];
        if(strict_tree&&a->parent>=0&&a->parent!=c1&&a->parent!=c2)continue; /* someone else's refinement */
        ids[nc]=a->id;cs[nc++]=sc[i];}
    /* "plain" (no widget) matches no particular n-grams, so it is always a candidate */
    if(plain_id>=0){int dup=0;for(int i=0;i<nc;i++)dup|=ids[i]==plain_id;if(!dup){ids[nc]=plain_id;cs[nc++]=cos_all[ans_at[plain_id]];}}
    for(int h=0;h<2;h++){int c=h?c2:c1;if(c<0)continue;int a=ID_A(c),b=ID_B(c);if(a<64)continue;
        for(int t=0;t<NT;t++){int id=PACK(a,b,t);if(ans_at[id]<0||answers[ans_at[id]].parent!=c)continue;int dup=0; /* only turns that follow it */for(int i=0;i<nc;i++)dup|=ids[i]==id;if(!dup){ids[nc]=id;cs[nc++]=cos_all[ans_at[id]];}}}
    free(cos_all);
    if(pplain)*pplain=1;
    if(!nc){*conf=0;return -1;}
    if(beta<0){*conf=cs[0];return ids[0];}
    /* Two hypotheses. A turn that follows one of umi's last replies is scored by the MLM with that
     * context; any other candidate is a topic switch, so it is scored as a fresh message, without it. */
    double lp[NGK+2*NT+1],lp0[NGK+2*NT+1];mlm_score(n,th,msg,c1,c2,ids,nc,lp);
    if(c1>=0)mlm_score(n,th,msg,-1,-1,ids,nc,lp0);
    int best=0;double mx=-1e300,z=0;
    for(int i=0;i<nc;i++){int p=answers[ans_at[ids[i]]].parent,fol=c1>=0&&(p==c1||(c2>=0&&p==c2));
        if(strict_tree&&p<0&&ID_A(ids[i])>=64)lp[i]+=root_bonus;
        if(c1>=0&&!fol)lp[i]=lp0[i];
        lp[i]+=beta*cs[i]+(c1>=0&&p==c1?gamma:0);if(lp[i]>mx){mx=lp[i];best=i;}}
    for(int i=0;i<nc;i++)z+=exp(lp[i]-mx);*conf=1/z;
    if(pplain){*pplain=0;for(int i=0;i<nc;i++)if(ids[i]==plain_id)*pplain=exp(lp[i]-mx)/z;}
    return ids[best];
}
static void evaluate_hybrid(Net*n,Ex*e,int ne,double beta,double gamma,double*acc0,double*acc1,int*n0,int*n1){
    int h0=0,h1=0,c0=0,c1=0;
#pragma omp parallel for num_threads(n->threads) reduction(+:h0,h1,c0,c1) schedule(dynamic,4)
    for(int i=0;i<ne;i++){
#ifdef _OPENMP
        int th=omp_get_thread_num();
#else
        int th=0;
#endif
        double c;int r=decide(n,th,e[i].msg,e[i].c1,e[i].c2,beta,gamma,&c,NULL);
        if(e[i].c1<0){c0++;h0+=r==e[i].id;}else{c1++;h1+=r==e[i].id;}
    }*acc0=c0?(double)h0/c0:0;*acc1=c1?(double)h1/c1:0;if(n0)*n0=c0;if(n1)*n1=c1;
}

/* accuracy on examples without context (single turn) and with context (follow-ups) */
static void evaluate(Net*n,Ex*e,int ne,int limit,double*acc0,double*acc1,int*n0,int*n1){
    int step=limit>0&&limit<ne?ne/limit:1,h0=0,h1=0,c0=0,c1=0;
#pragma omp parallel for num_threads(n->threads) reduction(+:h0,h1,c0,c1) schedule(dynamic,4)
    for(int i=0;i<ne;i+=step){
#ifdef _OPENMP
        int th=omp_get_thread_num();
#else
        int th=0;
#endif
        double c;int r=classify(n,th,e[i].msg,e[i].c1,e[i].c2,&c);
        if(e[i].c1<0){c0++;h0+=r==e[i].id;}else{c1++;h1+=r==e[i].id;}
    }*acc0=c0?(double)h0/c0:0;*acc1=c1?(double)h1/c1:0;if(n0)*n0=c0;if(n1)*n1=c1;
}

static Net net_load(const char*path,int threads){
    Net n={0};n.threads=threads;n.f=faithful_load(path,threads,SEQ);if(!n.f)n.m=load(path,threads,SEQ);
    if(!n.f&&!n.m){fprintf(stderr,"cannot load %s\n",path);exit(1);}
    if(!(n.f?n.f->global->c.bidir:n.m->c.bidir)){fprintf(stderr,"%s is not an MLM checkpoint\n",path);exit(1);}
    return n;
}

static Net net_load(const char*path,int threads);
static int cmd_train(const char*dir,const char*mode,double seconds,double entropy_seconds,int B,int threads,double lr,double aux,double sw,int seed,const char*out,int width,int bins,const char*init){
    int ntr,nte;Ex*tr=load_ex(dir,"train.tsv",&ntr),*te=load_ex(dir,"test.tsv",&nte);load_answers(dir);
    int withctx=0;for(int i=0;i<ntr;i++)withctx+=tr[i].c1>=0;
    printf("train %d (%d with context)  test %d  answers %d\n",ntr,withctx,nte,n_answers);
    Cfg gc={.vocab=257,.d=width,.n_layer=4,.T=4,.D=7,.n=16,.p=16,.K=3,.group=width,.walks=2,.temp=1,.fff_gate=1,.fw_mask=1,.bidir=1};
    FConfig fc={.width=width/2,.heads=4,.bins=bins,.maxpatch=8,.enc_layers=2,.dec_layers=4,.local_depth=4,.local_trees=2,.local_state=8,.neural=1,.mixer=2,.target=4};
    Net n={0};n.threads=threads;int blt=!strcmp(mode,"blt");double overhead=0;
    g_rng^=((uint64_t)seed+1)*0x9E3779B97F4A7C15ULL;
    if(init){ /* fine-tune: start from a trained checkpoint (its frozen entropy patcher included) */
        n=net_load(init,threads);blt=n.f!=NULL;use_muon=blt;
        printf("fine-tuning from %s\n",init);
    }else if(blt){
        use_muon=1;n.f=faithful_new(gc,fc,threads,SEQ,1);
        /* the frozen causal entropy patcher learns from the training messages and answers as one byte stream */
        size_t cap=0;for(int i=0;i<ntr;i++)cap+=strlen(tr[i].msg)+1;for(int i=0;i<n_answers;i++)cap+=strlen(answers[i].ans[0])+1;
        unsigned char*raw=malloc(cap);size_t at=0;uint64_t r=seed*7+3;
        int*perm=malloc(ntr*sizeof(int));for(int i=0;i<ntr;i++)perm[i]=i;for(int i=ntr-1;i>0;i--){int j=(int)(urand(&r)*(i+1))%(i+1);int t=perm[i];perm[i]=perm[j];perm[j]=t;}
        for(int i=0;i<ntr;i++){size_t l=strlen(tr[perm[i]].msg);memcpy(raw+at,tr[perm[i]].msg,l);at+=l;raw[at++]='\n';if(i%8==0&&i/8<n_answers){Ans*a=answers+i/8;l=strlen(a->ans[0]);memcpy(raw+at,a->ans[0],l);at+=l;raw[at++]='\n';}}
        overhead=train_entropy_model(n.f->entropy,raw,(long)at,B,SEQ,entropy_seconds,seed);
        double avg=calibrate_neural(n.f,raw,(long)at,SEQ,4);printf("entropy patcher: avg patch %.3f bytes\n",avg);free(raw);free(perm);
    }else{use_muon=0;n.m=model_new(&gc,threads,SEQ);params_init(&gc,&n.m->P);model_requantize(n.m);}
    (void)mode;
    const Cfg*mc=blt?&n.f->global->c:&n.m->c;
    size_t params=blt?core_parameters(&n.f->global->c)+core_parameters(&n.f->encoder->c)+core_parameters(&n.f->decoder->c)+n.f->dense+(embed_rows(&n.f->encoder->c)+1)*n.f->c.width:core_parameters(mc)+(embed_rows(mc)+1)*mc->d;
    printf("mode %s  core params %zu  n-gram cache rows %zu (%zu values)  frozen entropy %zu\n",mode,params,blt?n.f->hashrows:0,blt?n.f->hashsize:0,blt?core_parameters(&n.f->entropy->c)+(size_t)257*n.f->entropy->c.d:0);fflush(stdout);
    int mask=net_mask(&n),*bt=malloc((size_t)B*SEQ*sizeof(int)),*by=malloc((size_t)B*SEQ*sizeof(int));
    uint64_t rng=((uint64_t)seed+11)*0xD1B54A32D192ED03ULL;
    Hyper hp={.beta1=.9,.beta2=.98,.eps=1e-8,.weight_decay=.01,.clip=1};double budget=seconds-overhead,t0=wall(),nextlog=30,nextsave=1800,loss=0,ema=0;int step=0;
    while(wall()-t0<budget){
        double el=wall()-t0;hp.step=++step;hp.lr=lr*fmin(1,(el+.1)/fmin(10,budget*.05))*(.05+.95*.5*(1+cos(3.141592653589793*fmin(1,el/budget))));
        for(int k=0;k<B;k++){
            Ex*x=tr+(int)(urand(&rng)*ntr)%ntr;int c1=x->c1,c2=x->c2;
            /* topic switch: a context-free message may arrive at any point of a conversation */
            if(c1<0&&urand(&rng)<sw){c1=answers[(int)(urand(&rng)*n_answers)%n_answers].id;c2=urand(&rng)<.5?-1:answers[(int)(urand(&rng)*n_answers)%n_answers].id;}
            encode(x->msg,c1,c2,x->id,(int)(urand(&rng)*3)%3,mask,aux,&rng,bt+k*SEQ,by+k*SEQ);
        }
        loss=blt?faithful_step(n.f,bt,by,B,SEQ,&hp,NULL):tss_train_step(&n.m->c,&n.m->P,&n.m->Q,&n.m->G,n.m->hit,&n.m->M,&n.m->V,n.m->Gt,n.m->hits,n.m->ws,threads,bt,by,B,SEQ,&hp,n.m->k);
        ema=step==1?loss:.98*ema+.02*loss;
        if(wall()-t0>=nextlog){double a0,a1;evaluate(&n,te,nte,600,&a0,&a1,NULL,NULL);printf("step %6d  t %6.0fs  loss %.4f  test@600 single-turn %.3f follow-up %.3f\n",step,wall()-t0,ema,a0,a1);fflush(stdout);nextlog+=(budget<900?60:300);}
        if(out&&wall()-t0>=nextsave){if(blt?faithful_save(n.f,out):save(n.m,out))fprintf(stderr,"checkpoint save failed\n");else printf("checkpoint saved at %.0fs\n",wall()-t0);nextsave+=1800;}
    }
    double a0,a1,t0a,t1a;int n0,n1;evaluate(&n,te,nte,nte>20000?20000:0,&a0,&a1,&n0,&n1);evaluate(&n,tr,ntr,3000,&t0a,&t1a,NULL,NULL);
    printf("final: steps %d  test single-turn %.4f (n=%d)  follow-up %.4f (n=%d)  | train@3000 %.4f / %.4f\n",step,a0,n0,a1,n1,t0a,t1a);
    if(out&&(blt?faithful_save(n.f,out):save(n.m,out))){fprintf(stderr,"save failed\n");return 1;}
    return 0;
}

/* ---- names: regex canonicalisation ---- */
static regex_t re_strong,re_weak;static char user_name[64];
static void regex_init(void){
    /* strong: the next word is always a name; weak: only if the model then says "user introduces name" */
    if(regcomp(&re_strong,"(my name is|my name's|name's|call me|you can call me|i go by)[[:space:]]+([a-z][a-z'-]*)",REG_EXTENDED|REG_ICASE)||
       regcomp(&re_weak,"(^|[^a-z])(i am|i'm|im|this is|it's|its)[[:space:]]+([a-z][a-z'-]*)",REG_EXTENDED|REG_ICASE)){fprintf(stderr,"regex\n");exit(1);}
}
/* Replace capture group g of re in s with TAG; copies the captured word into cap. */
static int sub_name(regex_t*re,int g,const char*s,char*out,size_t on,char*cap,size_t cn){
    regmatch_t m[4];if(regexec(re,s,4,m,0)||m[g].rm_so<0)return 0;
    size_t wl=m[g].rm_eo-m[g].rm_so;if(s[m[g].rm_eo]=='>'||!strncasecmp(s+m[g].rm_so,"regex",5))return 0;
    snprintf(cap,cn,"%.*s",(int)(wl<cn-1?wl:cn-1),s+m[g].rm_so);
    snprintf(out,on,"%.*s%s%s",(int)m[g].rm_so,s,TAG,s+m[g].rm_eo);return 1;
}
/* Lower-case, squeeze spaces, and replace whole-word occurrences of the remembered name with TAG. */
static void canon(const char*in,char*out,size_t on){
    size_t o=0,nl=strlen(user_name),tl=strlen(TAG);int sp=1;
    for(size_t i=0;in[i]&&o+8<on;){
        if(nl&&!strncasecmp(in+i,user_name,nl)&&(i==0||!isalnum((unsigned char)in[i-1]))&&!isalnum((unsigned char)in[i+nl])){if(o+tl+8>=on)break;memcpy(out+o,TAG,tl);o+=tl;i+=nl;sp=0;continue;}
        unsigned char c=in[i++];if(isspace(c)){if(!sp)out[o++]=' ';sp=1;continue;}out[o++]=tolower(c);sp=0;
    }
    while(o&&out[o-1]==' ')o--;out[o]=0;
}
/* Pick a variant (never the one used last time) and spawn the name into its <regex> tags. */
static void render(Ans*a,char*out,size_t on){
    int v=(int)(rand01()*a->nv)%a->nv;if(a->nv>1&&v==a->last)v=(v+1)%a->nv;a->last=v;
    const char*src=*user_name?a->ans[v]:a->ano[v];size_t o=0,tl=strlen(TAG);
    for(const char*p=src;*p&&o+1<on;){
        if(!strncmp(p,TAG,tl)){size_t l=strlen(user_name);if(o+l<on){memcpy(out+o,user_name,l);o+=l;}p+=tl;continue;}
        out[o++]=*p==0x1C?'\n':*p;p++; /* web fragments store newlines as 0x1C */
    }out[o]=0;
}
static int hist[2]={-1,-1}; /* IDs of umi's last two replies */
/* ---- umi-smart-internet: compile the conversation into an HTML page with widgets ---- */
static Net*webnet;static const char*html_out,*web_dir;static double web_thresh=.995,web_gamma=4; /* see the widget gate study in weights/umi-smart-internet/README.md */
typedef struct{char*user,*umi,*widget;}Turn;static Turn*turns;static int n_turns;
static void html_esc(FILE*f,const char*s){for(;*s;s++){if(*s=='<')fputs("&lt;",f);else if(*s=='>')fputs("&gt;",f);else if(*s=='&')fputs("&amp;",f);else if(*s=='"')fputs("&quot;",f);else fputc(*s,f);}}
static void write_page(void){
    FILE*f=fopen(html_out,"w");if(!f){perror(html_out);return;}
    fputs("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>umi</title><style>\n",f);
    char p[1024];snprintf(p,sizeof p,"%s/design.css",web_dir);FILE*css=fopen(p,"r");if(css){int c;while((c=fgetc(css))!=EOF)fputc(c,f);fclose(css);}
    fputs("\n.umi-turn{margin:0 0 1rem}.umi-widget{margin:.5rem 0 0}</style></head><body><main class=\"umi-page\">\n",f);
    for(int i=0;i<n_turns;i++){
        fputs("<div class=\"umi-turn\"><div class=\"msg user\">",f);html_esc(f,turns[i].user);fputs("</div>\n",f);
        const char*w=turns[i].widget,*slot=w?strstr(w,"{{answer}}"):NULL;
        if(slot){fwrite(w,1,slot-w,f);html_esc(f,turns[i].umi);fputs(slot+10,f);}  /* wrapper widget presents the reply */
        else{if(*turns[i].umi){fputs("<div class=\"msg umi\">",f);html_esc(f,turns[i].umi);fputs("</div>\n",f);}if(w){fputs("<div class=\"umi-widget\">",f);fputs(w,f);fputs("</div>\n",f);}}
        fputs("</div>\n",f);
    }
    fputs("</main></body></html>\n",f);fclose(f);
}
/* A weak match ("i'm X", "it's X") can only be a name when X ends the message or nearly does
 * ("i'm sam", "hi i'm sam btw") and is not a common word ("i'm stressed about...", "it's for a job"). */
static int weak_ok(const char*text){
    static const char*stop[]={"a","an","the","not","so","just","really","very","too","also","still","good","fine","ok","okay",
        "great","tired","sad","happy","stressed","worried","bored","busy","sick","hungry","lost","confused","sorry","glad","sure",
        "back","here","home","done","ready","new","going","trying","feeling","about","for","from","in","on","at","with","to",
        "like","kinda","pretty","literally","always","never","gonna","getting","doing","having","looking","thinking","learning",
        "working","writing","making","cooking","planning","stuck","late","early","fine","alright","cool","nice","bad","awful",NULL};
    regmatch_t m[4];if(regexec(&re_weak,text,4,m,0))return 0;
    char w[64];int wl=m[3].rm_eo-m[3].rm_so;if(wl<=0||wl>=64)return 0;
    for(int i=0;i<wl;i++)w[i]=tolower((unsigned char)text[m[3].rm_so+i]);w[wl]=0;
    for(int i=0;stop[i];i++)if(!strcmp(w,stop[i]))return 0;
    int words=0,in=0;for(const char*q=text+m[3].rm_eo;*q;q++){int a=isalnum((unsigned char)*q);words+=a&&!in;in=a;}
    return words<=2;
}
/* One chat turn: canonicalise names, retrieve, remember the name and the reply.
 * Writes umi's rendered reply to msg and the canonical text the model saw to seen;
 * returns the packed answer ID. Shared by the CLI and the WebAssembly build. */
static int respond(Net*n,const char*line,double thresh,char*msg,size_t mn,char*text,double*conf,int*fell){
    char alt[1024],cap[64]="",cap2[64]="";double c,c2;
    canon(line,text,1024);
    int strong=sub_name(&re_strong,2,text,alt,sizeof alt,cap,sizeof cap);if(strong)strcpy(text,alt);
    int r=decide(n,0,text,hist[0],hist[1],hy_beta,hy_gamma,&c,NULL);
    if(!strong&&weak_ok(text)&&sub_name(&re_weak,3,text,alt,sizeof alt,cap2,sizeof cap2)){
        int r2=decide(n,0,alt,hist[0],hist[1],hy_beta,hy_gamma,&c2,NULL);
        if(r2==intro_id&&c2>c){r=r2;c=c2;strcpy(cap,cap2);strcpy(text,alt);}
    }
    if(r==intro_id&&*cap){
        snprintf(user_name,sizeof user_name,"%s",cap);user_name[0]=toupper((unsigned char)user_name[0]);
        for(char*p=user_name+1;*p;p++)*p=tolower((unsigned char)*p);
    }
    *fell=r<0||c<thresh;if(*fell&&fallback_id>=0)r=fallback_id;
    render(answers+ans_at[r],msg,mn);*conf=c;
    hist[1]=hist[0];hist[0]=r;return r;
}
/* One conversational turn with both models.
 * The widget model reads the canonical message first. A confident *wrapper* widget ("put that in a
 * card", "as flashcards") restyles umi's previous reply: the chat model is not asked and the
 * conversation context stays where it is. Otherwise the chat model replies, and a confident widget
 * is attached (a wrapper then presents the new reply). Shared by the CLI and the WebAssembly build. */
typedef struct{int r,fell,restyle,wrap,refine;double c,wc;char text[1024],msg[2048];char*widget;const char*wname;}TurnOut;
static char last_reply[2048];static int whist[2]={-1,-1}; /* IDs of the last two widgets shown */static double wrap_thresh=.5;static regex_t re_deictic;static int deictic_ok;
/* "put THAT in a card", "make IT a flashcard": the message points back at umi's last reply */
static int points_back(const char*text){
    if(!deictic_ok){deictic_ok=!regcomp(&re_deictic,"(^|[^a-z])(that|this|it|those|these|your (last )?(answer|reply))([^a-z]|$)",REG_EXTENDED|REG_ICASE|REG_NOSUB)?1:-1;}
    return deictic_ok==1&&!regexec(&re_deictic,text,0,NULL,0);
}
static int is_wrapper(const Ans*a){return strstr(a->ans[0],"{{answer}}")!=NULL;}
static void turn(Net*n,const char*line,double thresh,TurnOut*o){
    memset(o,0,sizeof*o);o->r=-1;
    int w=-1;double wc=0;
    if(webnet){
        char alt[1024],cap[64];canon(line,o->text,sizeof o->text);
        if(sub_name(&re_strong,2,o->text,alt,sizeof alt,cap,sizeof cap))strcpy(o->text,alt);
        double pp;C=&cache_web;w=decide(webnet,0,o->text,whist[0],whist[1],hy_beta,web_gamma,&wc,&pp);
        wc=1-pp; /* gate on "show a widget at all" (all non-plain mass), then show the best one */
        if(w>=0&&w!=plain_id){Ans*wa=answers+ans_at[w];
            int restyle=is_wrapper(wa)&&*last_reply&&points_back(o->text)&&wc>=wrap_thresh;
            if(!restyle&&wc<web_thresh)w=-1;}
        else w=-1;
        if(w>=0){Ans*wa=answers+ans_at[w];o->wrap=is_wrapper(wa);o->wname=wa->name;o->wc=wc;
            size_t cap2=strlen(wa->ans[0])+strlen(wa->ano[0])+4096;o->widget=malloc(cap2);render(wa,o->widget,cap2);}
        /* a refinement of the widget on screen is the whole answer: the chat model is not asked */
        if(w>=0&&!o->wrap){int p=answers[ans_at[w]].parent;o->refine=p>=0&&(p==whist[0]||p==whist[1]);}
        if(w>=0&&!o->wrap){whist[1]=whist[0];whist[0]=w;} /* a content widget becomes the context for refinements */
        C=&cache_chat;
        if(w>=0&&o->wrap&&*last_reply&&points_back(o->text)){o->restyle=1;o->r=hist[0];o->c=wc;snprintf(o->msg,sizeof o->msg,"%s",last_reply);return;}
        if(o->refine){o->r=hist[0];o->c=wc;o->msg[0]=0;return;}
    }
    o->r=respond(n,line,thresh,o->msg,sizeof o->msg,o->text,&o->c,&o->fell);
    snprintf(last_reply,sizeof last_reply,"%s",o->msg);
}
static void reply(Net*n,const char*line,double thresh,int verbose){
    TurnOut o;turn(n,line,thresh,&o);
    if(verbose){
        if(o.restyle)printf("  [restyle previous reply as %s | p=%.3f]\n",o.wname,o.wc);
        else if(o.refine)printf("  [refine the widget on screen: %s | p=%.3f]\n",o.wname,o.wc);
        else{Ans*a=answers+ans_at[o.r];printf("  [%d.%d.%d %s | p=%.3f%s | seen as \"%s\"]\n",ID_A(o.r),ID_B(o.r),ID_T(o.r),a->name,o.c,o.fell?" -> fallback":"",o.text);}
        if(o.widget&&!o.restyle&&!o.refine)printf("  [widget %s | p=%.3f]\n",o.wname,o.wc);
    }
    printf("umi> %s\n",o.refine?"[updated widget]":o.msg);fflush(stdout);
    if(webnet){turns=realloc(turns,(n_turns+1)*sizeof(Turn));turns[n_turns++]=(Turn){strdup(line),strdup(o.msg),o.widget};if(html_out)write_page();}
    else free(o.widget);
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: umi_chat train|eval|chat --data DIR [--load F|--save F] ...\n");return 2;}
    const char*cmd=argv[1],*dir="data",*in=NULL,*out=NULL,*mode="blt",*ask=NULL,*wload=NULL;
    double seconds=600,es=60,lr=.003,aux=.08,thresh=.05,sw=.3;int B=32,threads=4,seed=0,verbose=0,width=128,bins=16384;
    for(int i=2;i<argc;i++){
        const char*a=argv[i];if(!strcmp(a,"-v")){verbose=1;continue;}
        const char*v=i+1<argc?argv[++i]:NULL;if(!v){fprintf(stderr,"missing value for %s\n",a);return 2;}
        if(!strcmp(a,"--data"))dir=v;else if(!strcmp(a,"--load"))in=v;else if(!strcmp(a,"--save"))out=v;else if(!strcmp(a,"--mode"))mode=v;
        else if(!strcmp(a,"--seconds"))seconds=atof(v);else if(!strcmp(a,"--entropy-seconds"))es=atof(v);else if(!strcmp(a,"--lr"))lr=atof(v);
        else if(!strcmp(a,"--aux"))aux=atof(v);else if(!strcmp(a,"--threshold"))thresh=atof(v);else if(!strcmp(a,"--batch"))B=atoi(v);
        else if(!strcmp(a,"--switch"))sw=atof(v);else if(!strcmp(a,"--beta"))hy_beta=atof(v);else if(!strcmp(a,"--gamma"))hy_gamma=atof(v);else if(!strcmp(a,"--root-bonus"))root_bonus=atof(v);else if(!strcmp(a,"--eval-n"))eval_n=atoi(v);
        else if(!strcmp(a,"--web"))web_dir=v;else if(!strcmp(a,"--web-load"))wload=v;else if(!strcmp(a,"--html"))html_out=v;else if(!strcmp(a,"--web-threshold"))web_thresh=atof(v);else if(!strcmp(a,"--web-gamma"))web_gamma=atof(v);
        else if(!strcmp(a,"--threads"))threads=atoi(v);else if(!strcmp(a,"--seed"))seed=atoi(v);else if(!strcmp(a,"--width"))width=atoi(v);
        else if(!strcmp(a,"--ngram-bins"))bins=atoi(v);else if(!strcmp(a,"--ask"))ask=v;
        else{fprintf(stderr,"unknown option %s\n",a);return 2;}
    }
    if(!strcmp(cmd,"train"))return cmd_train(dir,mode,seconds,es,B,threads,lr,aux,sw,seed,out,width,bins,in);
    if(!in){fprintf(stderr,"--load required\n");return 2;}
    load_answers(dir);Net n=net_load(in,threads);
    {int ntr;Ex*tr=load_ex(dir,"train.tsv",&ntr);ngram_build(tr,ntr);}
    if(!strcmp(cmd,"eval")){
        /* pick the n-gram weight on the even half of the test set, report everything on the odd half */
        int nte,n0,n1;Ex*all=load_ex(dir,"test.tsv",&nte);Ex*tune=malloc(nte*sizeof(Ex)),*te=malloc(nte*sizeof(Ex));int nu=0,nr=0;
        for(int i=0;i<nte;i++){if(i%2)te[nr++]=all[i];else tune[nu++]=all[i];}
        if(eval_n>0){int st=nu/eval_n>1?nu/eval_n:1,k=0;for(int i=0;i<nu;i+=st)tune[k++]=tune[i];nu=k;st=nr/eval_n>1?nr/eval_n:1;k=0;for(int i=0;i<nr;i+=st)te[k++]=te[i];nr=k;}
        /* topic switch: the single-turn questions again, asked right after a random cached conversation turn */
        int nconv=0,*conv=malloc(n_answers*sizeof(int));for(int i=0;i<n_answers;i++)if(ID_A(answers[i].id)>=64)conv[nconv++]=i;
        Ex*swu=malloc(nu*sizeof(Ex)),*swr=malloc(nr*sizeof(Ex));int nsu=0,nsr=0;uint64_t rs=12345;
        for(int h=0;h<2&&nconv;h++){Ex*src=h?te:tune,*dst=h?swr:swu;int ns=h?nr:nu,*cnt=h?&nsr:&nsu;
            for(int i=0;i<ns;i++)if(src[i].c1<0){Ans*a=answers+conv[(int)(urand(&rs)*nconv)%nconv];dst[(*cnt)++]=(Ex){src[i].id,a->id,a->parent,src[i].msg};}}
        double a0,a1,as,dummy,best=-1,bb=hy_beta,bg=hy_gamma;int ns0,ns1;
        const double gb[]={8,16,32,64},gg[]={-4,0,2,4,8};
        for(int i=0;i<4;i++)for(int j=0;j<5;j++){
            evaluate_hybrid(&n,tune,nu,gb[i],gg[j],&a0,&a1,&n0,&n1);evaluate_hybrid(&n,swu,nsu,gb[i],gg[j],&dummy,&as,&ns0,&ns1);
            double m=(a0+a1+as)/3; /* each situation counts equally */
            printf("tune beta %4.1f gamma %4.1f: single-turn %.4f follow-up %.4f topic-switch %.4f\n",gb[i],gg[j],a0,a1,as);if(m>best){best=m;bb=gb[i];bg=gg[j];}}
        printf("held-out half (n=%d):\n",nr);
        evaluate(&n,te,nr,0,&a0,&a1,&n0,&n1);printf("  MLM beam only        single-turn %.4f (n=%d)  follow-up %.4f (n=%d)\n",a0,n0,a1,n1);
        evaluate_hybrid(&n,te,nr,-1,0,&a0,&a1,&n0,&n1);printf("  n-gram cache only            single-turn %.4f  follow-up %.4f\n",a0,a1);
        evaluate_hybrid(&n,te,nr,bb,bg,&a0,&a1,&n0,&n1);evaluate_hybrid(&n,swr,nsr,bb,bg,&dummy,&as,&ns0,&ns1);
        printf("  hybrid (beta %.0f, gamma %.0f)  single-turn %.4f  follow-up %.4f  topic-switch %.4f (n=%d)\n",bb,bg,a0,a1,as,ns1);
        printf("use: chat --beta %.0f --gamma %.0f\n",bb,bg);
        return 0;
    }
    if(!strcmp(cmd,"chat")){
        regex_init();
        if(web_dir&&wload){ /* second model: the umi-smart-internet widget cache */
            C=&cache_web;load_answers(web_dir);int nw;Ex*tw=load_ex(web_dir,"train.tsv",&nw);ngram_build(tw,nw);
            static Net wn;wn=net_load(wload,threads);webnet=&wn;C=&cache_chat;
            if(!html_out)html_out="umi.html";
        }
        if(ask){char buf[8192];snprintf(buf,sizeof buf,"%s",ask);for(char*p=strtok(buf,"|");p;p=strtok(NULL,"|")){printf("you> %s\n",p);reply(&n,p,thresh,verbose);}return 0;}
        char line[1024];printf("umi-1M-chat (Ctrl-D to quit)\n");
        while(printf("you> "),fflush(stdout),fgets(line,sizeof line,stdin)){line[strcspn(line,"\r\n")]=0;if(*line)reply(&n,line,thresh,verbose);}
        return 0;
    }
    fprintf(stderr,"unknown command %s\n",cmd);return 2;
}
