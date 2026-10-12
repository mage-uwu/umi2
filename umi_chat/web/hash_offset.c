/* Prints the byte offset and float count of the hashed n-gram table inside a BLT checkpoint
 * (the layout written by faithful_save), so pack_web.py can store just that table as fp16. */
#define NANOTSS_NO_MAIN
#include "../../nanotss_blt_mlm.c"
int main(int argc,char**argv){
    Faithful*f=faithful_load(argv[1],1,96);if(!f){fprintf(stderr,"cannot load %s\n",argv[1]);return 1;}
    size_t off=4+sizeof(int)+(f->global->c.bidir?sizeof(Cfg):(size_t)CFG_PREFIX)+sizeof(FConfig);
    Model*mods[3]={f->global,f->encoder,f->decoder};
    for(int j=0;j<3;j++){size_t z[N_FIELDS];sizes(&mods[j]->c,z);for(int k=0;k<N_FIELDS;k++)off+=z[k]*sizeof(real);}
    off+=f->dense*sizeof(real);
    printf("%zu %zu\n",off,f->hashsize);return 0;
}
