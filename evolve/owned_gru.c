/* BERHOT-owned byte GRU: forward, BPTT, gradient clipping and AdamW.
 * Only C/POSIX system headers and libm. No BLAS, ML runtime or vendor SDK.
 * Flat parameter order preserves the project's existing trained GRU weights.
 */
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef struct {
    int e,h,p; uint64_t step,rng;
    float *w,*m,*v,*g;
    int emb,wi,wh,bi,bh,head,bias;
} Net;
static float sig(float x) { return x>=0 ? 1.f/(1.f+expf(-x)) : expf(x)/(1.f+expf(x)); }
static uint64_t next_random(Net *n) { uint64_t x=n->rng; x^=x<<13; x^=x>>7; x^=x<<17; return n->rng=x; }
int owned_count(int e,int h) { return 256*e+3*h*e+3*h*h+6*h+256*h+256; }
void *owned_create(int e,int h,uint64_t seed) {
    if(e<1||e>1024||h<1||h>1024)return NULL;
    Net *n=calloc(1,sizeof(Net)); if(!n)return NULL;
    n->e=e;n->h=h;n->p=owned_count(e,h);n->rng=seed?seed:1;
    n->emb=0;n->wi=256*e;n->wh=n->wi+3*h*e;n->bi=n->wh+3*h*h;
    n->bh=n->bi+3*h;n->head=n->bh+3*h;n->bias=n->head+256*h;
    n->w=calloc((size_t)n->p*4,sizeof(float));
    if(!n->w){free(n);return NULL;} n->m=n->w+n->p;n->v=n->m+n->p;n->g=n->v+n->p;
    for(int i=0;i<n->p;i++)n->w[i]=(float)((next_random(n)>>11)*(1.0/9007199254740992.0)*2-1)/sqrtf((float)h);
    memset(n->w+n->bi,0,(size_t)6*h*sizeof(float));memset(n->w+n->bias,0,256*sizeof(float));
    return n;
}
void owned_free(void *ptr) { Net*n=ptr;if(n){free(n->w);free(n);} }
int owned_size(void *ptr) { return ((Net*)ptr)->p; }
void owned_get(void *ptr,float *out,int state) { Net*n=ptr;memcpy(out,n->w,(size_t)n->p*(state?3:1)*sizeof(float)); }
int owned_set(void *ptr,const float *in,int count,uint64_t step) {
    Net*n=ptr;if(count!=n->p&&count!=3*n->p)return -1;
    for(int i=0;i<count;i++)if(!isfinite(in[i]))return -2;
    memcpy(n->w,in,(size_t)count*sizeof(float));n->step=step;return 0;
}
void owned_gradients(void *ptr,float *out) { Net*n=ptr;memcpy(out,n->g,(size_t)n->p*sizeof(float)); }
static float dot(const float *a,const float *b,int count) {
    float sum=0; for(int i=0;i<count;i++)sum+=a[i]*b[i];return sum;
}
/* Hidden states are private per call/request; checkpoint weights stay immutable
 * throughout inference. */
int owned_next(void *ptr,int token,const float *prev,float *hidden,float *logits) {
    Net*n=ptr;if(token<0||token>255)return -1;int h=n->h,e=n->e;
    const float*x=n->w+token*e;
    for(int j=0;j<h;j++){
        float r=sig(n->w[n->bi+j]+n->w[n->bh+j]+dot(n->w+n->wi+j*e,x,e)+dot(n->w+n->wh+j*h,prev,h));
        float z=sig(n->w[n->bi+h+j]+n->w[n->bh+h+j]+dot(n->w+n->wi+(h+j)*e,x,e)+dot(n->w+n->wh+(h+j)*h,prev,h));
        float hn=n->w[n->bh+2*h+j]+dot(n->w+n->wh+(2*h+j)*h,prev,h);
        float v=tanhf(n->w[n->bi+2*h+j]+dot(n->w+n->wi+(2*h+j)*e,x,e)+r*hn);
        hidden[j]=(1-z)*v+z*prev[j];
    }
    for(int k=0;k<256;k++)logits[k]=n->w[n->bias+k]+dot(n->w+n->head+k*h,hidden,h);
    return 0;
}
/* data has length+1 bytes. Accumulate exact mean token cross entropy and
 * its full gradient, with a zero hidden state at each document window. */
double owned_loss(void *ptr,const unsigned char *data,int length,int backward) {
    Net*n=ptr;int h=n->h,e=n->e;
    if(length<1||length>4096)return NAN;
    size_t cells=(size_t)(length+1)*h;
    float *cache=calloc(cells*5,sizeof(float));if(!cache)return NAN;
    float *hs=cache,*rs=hs+cells,*zs=rs+cells,*ns=zs+cells,*hns=ns+cells;
    float *dhs=calloc((size_t)h*2+e+(size_t)h*6+256,sizeof(float));
    if(!dhs){free(cache);return NAN;}
    float *dh=dhs,*dp=dh+h,*dx=dp+h,*di=dx+e,*dg=di+3*h,*prob=dg+3*h;
    double loss=0;
    if(backward)memset(n->g,0,(size_t)n->p*sizeof(float));
    for(int t=0;t<length;t++){
        float *prev=hs+(size_t)t*h,*cur=prev+h,*r=rs+(size_t)t*h,*z=zs+(size_t)t*h,*v=ns+(size_t)t*h,*hn=hns+(size_t)t*h;
        const float*x=n->w+data[t]*e;
        for(int j=0;j<h;j++){
            r[j]=sig(n->w[n->bi+j]+n->w[n->bh+j]+dot(n->w+n->wi+j*e,x,e)+dot(n->w+n->wh+j*h,prev,h));
            z[j]=sig(n->w[n->bi+h+j]+n->w[n->bh+h+j]+dot(n->w+n->wi+(h+j)*e,x,e)+dot(n->w+n->wh+(h+j)*h,prev,h));
            hn[j]=n->w[n->bh+2*h+j]+dot(n->w+n->wh+(2*h+j)*h,prev,h);
            v[j]=tanhf(n->w[n->bi+2*h+j]+dot(n->w+n->wi+(2*h+j)*e,x,e)+r[j]*hn[j]);
            cur[j]=(1-z[j])*v[j]+z[j]*prev[j];
        }
        float maximum=-INFINITY;
        for(int k=0;k<256;k++){prob[k]=n->w[n->bias+k]+dot(n->w+n->head+k*h,cur,h);if(prob[k]>maximum)maximum=prob[k];}
        double sum=0;for(int k=0;k<256;k++)sum+=exp((double)prob[k]-maximum);
        loss+=log(sum)+maximum-prob[data[t+1]];
    }
    if(backward)for(int t=length-1;t>=0;t--){
        float *prev=hs+(size_t)t*h,*cur=prev+h,*r=rs+(size_t)t*h,*z=zs+(size_t)t*h,*v=ns+(size_t)t*h,*hn=hns+(size_t)t*h;
        const float*x=n->w+data[t]*e;
        float maximum=-INFINITY;
        for(int k=0;k<256;k++){prob[k]=n->w[n->bias+k]+dot(n->w+n->head+k*h,cur,h);if(prob[k]>maximum)maximum=prob[k];}
        double sum=0;for(int k=0;k<256;k++)sum+=exp((double)prob[k]-maximum);
        for(int k=0;k<256;k++){
            float d=(float)((exp((double)prob[k]-maximum)/sum-(k==data[t+1]))/length);
            n->g[n->bias+k]+=d;
            float *gw=n->g+n->head+k*h;const float*w=n->w+n->head+k*h;
            for(int j=0;j<h;j++){gw[j]+=d*cur[j];dh[j]+=d*w[j];}
        }
        memset(dp,0,(size_t)h*sizeof(float));memset(dx,0,(size_t)e*sizeof(float));
        for(int j=0;j<h;j++){
            float dn=dh[j]*(1-z[j])*(1-v[j]*v[j]);
            float dr=dn*hn[j]*r[j]*(1-r[j]);
            float dz=dh[j]*(prev[j]-v[j])*z[j]*(1-z[j]);
            di[j]=dg[j]=dr;di[h+j]=dg[h+j]=dz;di[2*h+j]=dn;dg[2*h+j]=dn*r[j];dp[j]=dh[j]*z[j];
        }
        for(int j=0;j<3*h;j++){
            n->g[n->bi+j]+=di[j];n->g[n->bh+j]+=dg[j];
            float *gi=n->g+n->wi+j*e,*gh=n->g+n->wh+j*h;
            const float*wi=n->w+n->wi+j*e,*wh=n->w+n->wh+j*h;
            for(int k=0;k<e;k++){gi[k]+=di[j]*x[k];dx[k]+=di[j]*wi[k];}
            for(int k=0;k<h;k++){gh[k]+=dg[j]*prev[k];dp[k]+=dg[j]*wh[k];}
        }
        for(int k=0;k<e;k++)n->g[data[t]*e+k]+=dx[k];
        memcpy(dh,dp,(size_t)h*sizeof(float));
    }
    free(dhs);free(cache);return loss/length;
}
int owned_update(void *ptr,float rate,float decay,float clip) {
    Net*n=ptr;double norm=0;
    for(int i=0;i<n->p;i++){if(!isfinite(n->g[i]))return -1;norm+=(double)n->g[i]*n->g[i];}
    float scale=norm>(double)clip*clip?(float)(clip/sqrt(norm)):1;
    n->step++;double b1=1-pow(.9,(double)n->step),b2=1-pow(.999,(double)n->step);
    for(int i=0;i<n->p;i++){
        float g=n->g[i]*scale;n->m[i]=.9f*n->m[i]+.1f*g;n->v[i]=.999f*n->v[i]+.001f*g*g;
        n->w[i]=(1-rate*decay)*n->w[i]-rate*(float)((n->m[i]/b1)/(sqrt(n->v[i]/b2)+1e-8));
        if(!isfinite(n->w[i]))return -2;
    }return 0;
}
