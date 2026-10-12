/*
 * umi_wasm.c -- WebAssembly entry points for umi-1M-chat (browser chat window).
 * The same C code as the CLI: umi_chat.c's turn() runs both models in the browser.
 *
 * Build (Emscripten):  see build.sh
 * JS:  umi_load(model, dir, beta, gamma), optionally umi_load_web(model, dir, gate, wrap_gate),
 *      then umi_say(text) per turn (returns JSON).
 */
#define main umi_cli_main
#include "../umi_chat.c"
#undef main
#include <emscripten.h>

static Net net,wnet;static char*out;static size_t outcap;

/* Inference never touches optimizer state or gradients: free them (about 120 MB per model). */
static void slim(Net*n){
    Faithful*f=n->f;if(!f)return;
    free(f->g);free(f->m);free(f->v);f->g=f->m=f->v=NULL;
    for(int t=0;t<f->threads;t++){free(f->tg[t]);f->tg[t]=NULL;}
}
static void put(size_t*o,const char*s){size_t l=strlen(s);if(*o+l+1>outcap){outcap=(*o+l+1)*2;out=realloc(out,outcap);}memcpy(out+*o,s,l+1);*o+=l;}
static void jstr(size_t*o,const char*s){ /* append s as a JSON string */
    char b[8];put(o,"\"");
    for(;*s;s++){unsigned char ch=*s;
        if(ch=='"')put(o,"\\\"");else if(ch=='\\')put(o,"\\\\");else if(ch=='\n')put(o,"\\n");
        else if(ch<0x20){snprintf(b,sizeof b,"\\u%04x",ch);put(o,b);}
        else{b[0]=ch;b[1]=0;put(o,b);}}
    put(o,"\"");
}
static int load_cache(const char*dir){
    load_answers(dir);int n;Ex*tr=load_ex(dir,"train.tsv",&n);ngram_build(tr,n);
    for(int i=0;i<n;i++)free(tr[i].msg);free(tr);return n_answers;
}
EMSCRIPTEN_KEEPALIVE int umi_load(const char*model,const char*dir,double beta,double gamma){
    C=&cache_chat;int n=load_cache(dir);
    net=net_load(model,1);slim(&net);regex_init();hy_beta=beta;hy_gamma=gamma;return n;
}
EMSCRIPTEN_KEEPALIVE int umi_load_web(const char*model,const char*dir,double gate,double wrap_gate,double gamma){
    C=&cache_web;int n=load_cache(dir);
    wnet=net_load(model,1);slim(&wnet);webnet=&wnet;web_thresh=gate;wrap_thresh=wrap_gate;web_gamma=gamma;C=&cache_chat;return n;
}
EMSCRIPTEN_KEEPALIVE const char*umi_say(const char*line,double thresh){
    TurnOut t;turn(&net,line,thresh,&t);size_t o=0;char b[160];
    put(&o,"{\"reply\":");jstr(&o,t.msg);
    if(t.restyle)put(&o,",\"id\":\"\",\"intent\":\"restyle previous reply\"");
    else if(t.refine)put(&o,",\"id\":\"\",\"intent\":\"refine the widget on screen\"");
    else{Ans*a=answers+ans_at[t.r];snprintf(b,sizeof b,",\"id\":\"%d.%d.%d\",\"intent\":",ID_A(t.r),ID_B(t.r),ID_T(t.r));put(&o,b);jstr(&o,a->name);}
    snprintf(b,sizeof b,",\"p\":%.4f,\"fallback\":%s,\"restyle\":%s,\"refine\":%s,\"wrap\":%s,\"wp\":%.4f,\"seen\":",
             t.c,t.fell?"true":"false",t.restyle?"true":"false",t.refine?"true":"false",t.wrap?"true":"false",t.wc);put(&o,b);jstr(&o,t.text);
    put(&o,",\"name\":");jstr(&o,user_name);
    put(&o,",\"widget\":");if(t.widget){jstr(&o,t.widget);free(t.widget);}else put(&o,"null");
    put(&o,",\"widget_name\":");if(t.wname)jstr(&o,t.wname);else put(&o,"null");
    put(&o,"}");return out;
}
EMSCRIPTEN_KEEPALIVE void umi_reset(void){hist[0]=hist[1]=-1;whist[0]=whist[1]=-1;user_name[0]=0;last_reply[0]=0;}
