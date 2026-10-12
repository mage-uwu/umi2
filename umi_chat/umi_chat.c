/*
 * umi_chat.c -- umi-1M-chat: a retrieval chatbot on the nanotss MLM.
 *
 * There is no autoregressive generation. Every reply is a cached answer
 * (written by DeepSeek, see gen_data.py). The bidirectional masked-LM reads
 * the user's message and fills a two-byte answer-ID slot:
 *
 *     <message> SEP [MASK] [MASK] END PAD...
 *                   domain intent
 *
 * Decoding is a small mask-predict search: score the domain byte with both
 * slots masked, reveal each of the top-K domains, score the intent byte, and
 * retrieve the answer with the best joint log-probability. A low joint
 * probability retrieves the fallback answer instead.
 *
 * User names are kept consistent with regexes: a name in the user's text is
 * replaced by the tag <regex>user_name</regex> before the model sees it
 * (training data uses the same canonical form), and the same tag inside a
 * cached answer is spawned back in from the remembered name.
 *
 * In BLT mode (default) the byte encoder also reads a large hashed 3..8-byte
 * n-gram cache (--ngram-bins); those rows are excluded from the ~1M budget.
 *
 * Build:  cc -O3 -march=native -fopenmp umi_chat.c -lm -o umi_chat
 * Train:  ./umi_chat train --data data --seconds 3600 --threads 8 --save umi.bin
 * Eval:   ./umi_chat eval  --data data --load umi.bin
 * Chat:   ./umi_chat chat  --data data --load umi.bin [--ask "text"] [-v]
 */
#define NANOTSS_NO_MAIN
#include "../nanotss_blt_mlm.c"
#include <regex.h>
#include <ctype.h>

/* 64 ID symbols; must match prepare.py. */
static const char ALPHA[]="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/";
static int sym(int c){const char*p=c?strchr(ALPHA,c):NULL;return p?(int)(p-ALPHA):-1;}
enum{SEQ=80,MAXMSG=SEQ-4,SEP=0x1F,END=0x1E,PAD=0,TOPK=4,NV=4};
static const char TAG[]="<regex>user_name</regex>";

typedef struct{int a,b;char*msg;}Ex;
typedef struct{int a,b,nv,last;char*name,*ans[NV],*ano[NV];}Ans;
static Ans answers[64*64];static int n_answers,ans_at[64][64];
static int intro_a=-1,intro_b=-1,fallback_a=-1,fallback_b=-1;

static char*dupn(const char*s,size_t n){char*r=malloc(n+1);memcpy(r,s,n);r[n]=0;return r;}
static Ex*load_ex(const char*dir,const char*name,int*n){
    char p[1024];snprintf(p,sizeof p,"%s/%s",dir,name);FILE*fp=fopen(p,"r");if(!fp){fprintf(stderr,"cannot open %s\n",p);exit(1);}
    int cap=1024;Ex*e=malloc(cap*sizeof(Ex));*n=0;char line[4096];
    while(fgets(line,sizeof line,fp)){
        size_t l=strcspn(line,"\r\n");line[l]=0;if(l<4||line[2]!='\t')continue;
        int a=sym(line[0]),b=sym(line[1]);if(a<0||b<0)continue;
        if(*n==cap)e=realloc(e,(cap*=2)*sizeof(Ex));e[*n].a=a;e[*n].b=b;e[*n].msg=dupn(line+3,strlen(line+3));(*n)++;
    }fclose(fp);return e;
}
static void load_answers(const char*dir){
    char p[1024];snprintf(p,sizeof p,"%s/answers.tsv",dir);FILE*fp=fopen(p,"r");if(!fp){fprintf(stderr,"cannot open %s\n",p);exit(1);}
    memset(ans_at,-1,sizeof ans_at);char line[8192];
    while(fgets(line,sizeof line,fp)&&n_answers<64*64){
        line[strcspn(line,"\r\n")]=0;char*f[4];int k=0;f[k++]=line;
        for(char*q=line;*q&&k<4;q++)if(*q=='\t'){*q=0;f[k++]=q+1;}
        if(k<4||strlen(f[0])!=2)continue;int a=sym(f[0][0]),b=sym(f[0][1]);if(a<0||b<0)continue;
        if(ans_at[a][b]>=0){Ans*r=answers+ans_at[a][b];if(r->nv<NV){r->ans[r->nv]=strdup(f[2]);r->ano[r->nv++]=strdup(f[3]);}continue;}
        Ans*r=answers+n_answers;r->a=a;r->b=b;r->nv=1;r->last=-1;r->name=strdup(f[1]);r->ans[0]=strdup(f[2]);r->ano[0]=strdup(f[3]);ans_at[a][b]=n_answers++;
        if(!strcmp(f[1],"umi_core/user_introduces_name")){intro_a=a;intro_b=b;}
        if(!strcmp(f[1],"umi_core/fallback_unknown")){fallback_a=a;fallback_b=b;}
    }fclose(fp);
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

/* Lay out one example; returns the domain-slot position. aux>0 adds BERT corruption on message bytes. */
static int encode(const char*msg,int a,int b,int reveal_a,int mask,double aux,uint64_t*rng,int*tok,int*lab){
    int len=(int)strlen(msg);if(len>MAXMSG)len=MAXMSG;
    for(int i=0;i<SEQ;i++){tok[i]=PAD;lab[i]=-1;}
    for(int i=0;i<len;i++){
        tok[i]=(unsigned char)msg[i];
        if(aux>0&&urand(rng)<aux){lab[i]=tok[i];real u=urand(rng);if(u<(real).8)tok[i]=mask;else if(u<(real).9)tok[i]=(int)(urand(rng)*256)&255;}
    }
    tok[len]=SEP;tok[len+1]=mask;tok[len+2]=mask;tok[len+3]=END;
    if(a>=0){lab[len+1]=ALPHA[a];lab[len+2]=ALPHA[b];}
    if(reveal_a){tok[len+1]=ALPHA[a];lab[len+1]=-1;}
    return len+1;
}
static double logsoftmax_at(const real*l,int v){double mx=-1e300,z=0;for(int i=0;i<256;i++)mx=l[i]>mx?l[i]:mx;for(int i=0;i<256;i++)z+=exp(l[i]-mx);return l[v]-mx-log(z);}

/* Retrieval decision. Returns the answer index; *conf is the joint probability. */
static int classify(Net*n,int th,const char*msg,double*conf,int*dom_top){
    int tok[SEQ],lab[SEQ],mask=net_mask(n);real*lg=malloc((size_t)SEQ*256*sizeof(real));
    int s=encode(msg,-1,-1,0,mask,0,NULL,tok,lab);net_logits(n,th,tok,lg);
    /* domain distribution, renormalised over domains that exist */
    double pa[64],z=0;for(int a=0;a<64;a++){int any=0;for(int b=0;b<64;b++)any|=ans_at[a][b]>=0;pa[a]=any?exp(logsoftmax_at(lg+(size_t)s*256,ALPHA[a])):0;z+=pa[a];}
    for(int a=0;a<64;a++)pa[a]/=z;
    int top[TOPK];for(int k=0;k<TOPK;k++){top[k]=-1;for(int a=0;a<64;a++){int used=0;for(int j=0;j<k;j++)used|=top[j]==a;if(!used&&pa[a]>0&&(top[k]<0||pa[a]>pa[top[k]]))top[k]=a;}}
    if(dom_top)*dom_top=top[0];
    int best=-1;double bestp=-1;
    for(int k=0;k<TOPK&&top[k]>=0;k++){
        int a=top[k];tok[s]=ALPHA[a];net_logits(n,th,tok,lg);
        double pb[64],zb=0;for(int b=0;b<64;b++){pb[b]=ans_at[a][b]>=0?exp(logsoftmax_at(lg+(size_t)(s+1)*256,ALPHA[b])):0;zb+=pb[b];}
        for(int b=0;b<64;b++)if(pb[b]>0&&pa[a]*pb[b]/zb>bestp){bestp=pa[a]*pb[b]/zb;best=ans_at[a][b];}
    }
    free(lg);*conf=bestp;return best;
}

static void evaluate(Net*n,Ex*e,int ne,int limit,double*acc,double*dacc){
    if(limit>0&&limit<ne)ne=limit;int hit=0,dhit=0;
#pragma omp parallel for num_threads(n->threads) reduction(+:hit,dhit) schedule(dynamic,8)
    for(int i=0;i<ne;i++){
#ifdef _OPENMP
        int th=omp_get_thread_num();
#else
        int th=0;
#endif
        double c;int d;int r=classify(n,th,e[i].msg,&c,&d);hit+=r>=0&&answers[r].a==e[i].a&&answers[r].b==e[i].b;dhit+=d==e[i].a;
    }*acc=(double)hit/ne;*dacc=(double)dhit/ne;
}

static Net net_load(const char*path,int threads){
    Net n={0};n.threads=threads;n.f=faithful_load(path,threads,SEQ);if(!n.f)n.m=load(path,threads,SEQ);
    if(!n.f&&!n.m){fprintf(stderr,"cannot load %s\n",path);exit(1);}
    if(!(n.f?n.f->global->c.bidir:n.m->c.bidir)){fprintf(stderr,"%s is not an MLM checkpoint\n",path);exit(1);}
    return n;
}

static int cmd_train(const char*dir,const char*mode,double seconds,double entropy_seconds,int B,int threads,double lr,double aux,int seed,const char*out,int width,int bins){
    int ntr,nte;Ex*tr=load_ex(dir,"train.tsv",&ntr),*te=load_ex(dir,"test.tsv",&nte);load_answers(dir);
    printf("train %d  test %d  intents %d\n",ntr,nte,n_answers);
    Cfg gc={.vocab=257,.d=width,.n_layer=4,.T=4,.D=7,.n=16,.p=16,.K=3,.group=width,.walks=2,.temp=1,.fff_gate=1,.fw_mask=1,.bidir=1};
    FConfig fc={.width=width/2,.heads=4,.bins=bins,.maxpatch=8,.enc_layers=2,.dec_layers=4,.local_depth=4,.local_trees=2,.local_state=8,.neural=1,.mixer=2,.target=4};
    Net n={0};n.threads=threads;int blt=!strcmp(mode,"blt");double overhead=0;
    g_rng^=((uint64_t)seed+1)*0x9E3779B97F4A7C15ULL;
    if(blt){
        use_muon=1;n.f=faithful_new(gc,fc,threads,SEQ,1);
        /* the frozen causal entropy patcher learns from the training messages as one byte stream */
        size_t cap=0;for(int i=0;i<ntr;i++)cap+=strlen(tr[i].msg)+1;unsigned char*raw=malloc(cap);size_t at=0;
        uint64_t r=seed*7+3;int*perm=malloc(ntr*sizeof(int));for(int i=0;i<ntr;i++)perm[i]=i;for(int i=ntr-1;i>0;i--){int j=(int)(urand(&r)*(i+1))%(i+1);int t=perm[i];perm[i]=perm[j];perm[j]=t;}
        for(int i=0;i<ntr;i++){size_t l=strlen(tr[perm[i]].msg);memcpy(raw+at,tr[perm[i]].msg,l);at+=l;raw[at++]='\n';}
        overhead=train_entropy_model(n.f->entropy,raw,(long)at,B,SEQ,entropy_seconds,seed);
        double avg=calibrate_neural(n.f,raw,(long)at,SEQ,4);printf("entropy patcher: avg patch %.3f bytes\n",avg);free(raw);free(perm);
    }else{use_muon=0;n.m=model_new(&gc,threads,SEQ);params_init(&gc,&n.m->P);model_requantize(n.m);}
    const Cfg*mc=blt?&n.f->global->c:&n.m->c;
    size_t params=blt?core_parameters(&n.f->global->c)+core_parameters(&n.f->encoder->c)+core_parameters(&n.f->decoder->c)+n.f->dense+(embed_rows(&n.f->encoder->c)+1)*n.f->c.width:core_parameters(mc)+(embed_rows(mc)+1)*mc->d;
    printf("mode %s  core params %zu  n-gram cache rows %zu (%zu values)  frozen entropy %zu\n",mode,params,blt?n.f->hashrows:0,blt?n.f->hashsize:0,blt?core_parameters(&n.f->entropy->c)+(size_t)257*n.f->entropy->c.d:0);fflush(stdout);
    int mask=net_mask(&n),*bt=malloc((size_t)B*SEQ*sizeof(int)),*by=malloc((size_t)B*SEQ*sizeof(int));
    uint64_t rng=((uint64_t)seed+11)*0xD1B54A32D192ED03ULL;
    Hyper hp={.beta1=.9,.beta2=.98,.eps=1e-8,.weight_decay=.01,.clip=1};double budget=seconds-overhead,t0=wall(),nextlog=30,loss=0,ema=0;int step=0;
    while(wall()-t0<budget){
        double el=wall()-t0;hp.step=++step;hp.lr=lr*fmin(1,(el+.1)/fmin(10,budget*.05))*(.05+.95*.5*(1+cos(3.141592653589793*fmin(1,el/budget))));
        for(int k=0;k<B;k++){Ex*x=tr+(int)(urand(&rng)*ntr)%ntr;encode(x->msg,x->a,x->b,urand(&rng)<.5,mask,aux,&rng,bt+k*SEQ,by+k*SEQ);}
        loss=blt?faithful_step(n.f,bt,by,B,SEQ,&hp,NULL):tss_train_step(&n.m->c,&n.m->P,&n.m->Q,&n.m->G,n.m->hit,&n.m->M,&n.m->V,n.m->Gt,n.m->hits,n.m->ws,threads,bt,by,B,SEQ,&hp,n.m->k);
        ema=step==1?loss:.98*ema+.02*loss;
        if(wall()-t0>=nextlog){double acc,dacc;evaluate(&n,te,nte,400,&acc,&dacc);printf("step %6d  t %6.0fs  loss %.4f  test@400 intent %.3f domain %.3f\n",step,wall()-t0,ema,acc,dacc);fflush(stdout);nextlog+=60;}
    }
    double acc,dacc,tacc,tdacc;evaluate(&n,te,nte,0,&acc,&dacc);evaluate(&n,tr,ntr,2000,&tacc,&tdacc);
    printf("final: steps %d  test intent acc %.4f  domain acc %.4f  (train@2000 %.4f)\n",step,acc,dacc,tacc);
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
/* Lower-case, squeeze spaces, and replace whole-word occurrences of the remembered name. */
static void canon(const char*in,char*out,size_t on){
    size_t o=0;int sp=1;char nm[64];size_t nl=strlen(user_name);for(size_t i=0;i<nl;i++)nm[i]=tolower((unsigned char)user_name[i]);
    for(size_t i=0;in[i]&&o+8<on;){
        if(nl&&!strncasecmp(in+i,nm,nl)&&(i==0||!isalnum((unsigned char)in[i-1]))&&!isalnum((unsigned char)in[i+nl])){size_t tl=strlen(TAG);if(o+tl+8>=on)break;memcpy(out+o,TAG,tl);o+=tl;i+=nl;sp=0;continue;}
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
        out[o++]=*p++;
    }out[o]=0;
}
static void reply(Net*n,const char*line,double thresh,int verbose){
    char text[1024],alt[1024],cap[64]="",cap2[64]="",msg[1024];double c,c2;
    canon(line,text,sizeof text);
    int strong=sub_name(&re_strong,2,text,alt,sizeof alt,cap,sizeof cap);if(strong)strcpy(text,alt);
    int r=classify(n,0,text,&c,NULL);
    if(!strong&&sub_name(&re_weak,3,text,alt,sizeof alt,cap2,sizeof cap2)){
        int r2=classify(n,0,alt,&c2,NULL);
        if(r2>=0&&answers[r2].a==intro_a&&answers[r2].b==intro_b&&c2>c){r=r2;c=c2;strcpy(cap,cap2);strcpy(text,alt);}
    }
    if(r>=0&&answers[r].a==intro_a&&answers[r].b==intro_b&&*cap){
        snprintf(user_name,sizeof user_name,"%s",cap);user_name[0]=toupper((unsigned char)user_name[0]);
        for(char*p=user_name+1;*p;p++)*p=tolower((unsigned char)*p);
    }
    int fell=r<0||c<thresh;if(fell&&fallback_a>=0)r=ans_at[fallback_a][fallback_b];
    render(answers+r,msg,sizeof msg);
    if(verbose)printf("  [%s | p=%.3f%s | seen as \"%s\"]\n",answers[r].name,c,fell?" -> fallback":"",text);
    printf("umi> %s\n",msg);fflush(stdout);
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: umi_chat train|eval|chat --data DIR [--load F|--save F] ...\n");return 2;}
    const char*cmd=argv[1],*dir="data",*in=NULL,*out=NULL,*mode="blt",*ask=NULL;
    double seconds=600,es=60,lr=.003,aux=.08,thresh=.15;int B=32,threads=4,seed=0,verbose=0,width=128,bins=16384;
    for(int i=2;i<argc;i++){
        const char*a=argv[i];if(!strcmp(a,"-v")){verbose=1;continue;}
        const char*v=i+1<argc?argv[++i]:NULL;if(!v){fprintf(stderr,"missing value for %s\n",a);return 2;}
        if(!strcmp(a,"--data"))dir=v;else if(!strcmp(a,"--load"))in=v;else if(!strcmp(a,"--save"))out=v;else if(!strcmp(a,"--mode"))mode=v;
        else if(!strcmp(a,"--seconds"))seconds=atof(v);else if(!strcmp(a,"--entropy-seconds"))es=atof(v);else if(!strcmp(a,"--lr"))lr=atof(v);
        else if(!strcmp(a,"--aux"))aux=atof(v);else if(!strcmp(a,"--threshold"))thresh=atof(v);else if(!strcmp(a,"--batch"))B=atoi(v);
        else if(!strcmp(a,"--threads"))threads=atoi(v);else if(!strcmp(a,"--seed"))seed=atoi(v);else if(!strcmp(a,"--width"))width=atoi(v);else if(!strcmp(a,"--ngram-bins"))bins=atoi(v);else if(!strcmp(a,"--ask"))ask=v;
        else{fprintf(stderr,"unknown option %s\n",a);return 2;}
    }
    if(!strcmp(cmd,"train"))return cmd_train(dir,mode,seconds,es,B,threads,lr,aux,seed,out,width,bins);
    if(!in){fprintf(stderr,"--load required\n");return 2;}
    load_answers(dir);Net n=net_load(in,threads);
    if(!strcmp(cmd,"eval")){int nte;Ex*te=load_ex(dir,"test.tsv",&nte);double acc,dacc;evaluate(&n,te,nte,0,&acc,&dacc);printf("test %d  intent acc %.4f  domain acc %.4f\n",nte,acc,dacc);return 0;}
    if(!strcmp(cmd,"chat")){
        regex_init();
        if(ask){char buf[4096];snprintf(buf,sizeof buf,"%s",ask);for(char*p=strtok(buf,"|");p;p=strtok(NULL,"|")){printf("you> %s\n",p);reply(&n,p,thresh,verbose);}return 0;}
        char line[1024];printf("umi-1M-chat (Ctrl-D to quit)\n");
        while(printf("you> "),fflush(stdout),fgets(line,sizeof line,stdin)){line[strcspn(line,"\r\n")]=0;if(*line)reply(&n,line,thresh,verbose);}
        return 0;
    }
    fprintf(stderr,"unknown command %s\n",cmd);return 2;
}
