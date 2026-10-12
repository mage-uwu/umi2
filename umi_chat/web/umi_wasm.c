/*
 * umi_wasm.c -- WebAssembly entry points for umi-1M-chat (browser chat window).
 * The same C retrieval code as the CLI: umi_chat.c's respond() runs in the browser.
 *
 * Build (Emscripten):  see build.sh
 * JS:  umi_load("/m/umi.bin", "/d") once, then umi_say(text) per turn (returns JSON).
 */
#define main umi_cli_main
#include "../umi_chat.c"
#undef main
#include <emscripten.h>

static Net net;static char out[16384];

static void jcat(size_t*o,const char*s){ /* append s as a JSON string body */
    for(;*s&&*o+8<sizeof out;s++){unsigned char ch=*s;
        if(ch=='"'||ch=='\\'){out[(*o)++]='\\';out[(*o)++]=ch;}
        else if(ch=='\n'){out[(*o)++]='\\';out[(*o)++]='n';}
        else if(ch<0x20){*o+=snprintf(out+*o,sizeof out-*o,"\\u%04x",ch);}
        else out[(*o)++]=ch;}
}
EMSCRIPTEN_KEEPALIVE int umi_load(const char*model,const char*dir,double beta,double gamma){
    C=&cache_chat;load_answers(dir);
    int n;Ex*tr=load_ex(dir,"train.tsv",&n);ngram_build(tr,n);
    for(int i=0;i<n;i++)free(tr[i].msg);free(tr);
    net=net_load(model,1);regex_init();hy_beta=beta;hy_gamma=gamma;
    return n_answers;
}
EMSCRIPTEN_KEEPALIVE const char*umi_say(const char*line,double thresh){
    char text[1024],msg[4096];double c;int fell;
    int r=respond(&net,line,thresh,msg,sizeof msg,text,&c,&fell);Ans*a=answers+ans_at[r];
    size_t o=0;o+=snprintf(out,sizeof out,"{\"reply\":\"");jcat(&o,msg);
    o+=snprintf(out+o,sizeof out-o,"\",\"id\":\"%d.%d.%d\",\"intent\":\"",ID_A(r),ID_B(r),ID_T(r));jcat(&o,a->name);
    o+=snprintf(out+o,sizeof out-o,"\",\"p\":%.4f,\"fallback\":%s,\"seen\":\"",c,fell?"true":"false");jcat(&o,text);
    o+=snprintf(out+o,sizeof out-o,"\",\"name\":\"");jcat(&o,user_name);
    snprintf(out+o,sizeof out-o,"\"}");return out;
}
EMSCRIPTEN_KEEPALIVE void umi_reset(void){hist[0]=hist[1]=-1;user_name[0]=0;}
