/* rikfad.c -- long-horizon integration of the gamma=0 rank-1 iKFAD dynamics
 *             using the BACD operator splitting of the method itself.
 *
 * Sub-flows, each integrated exactly (gamma = 0, so D is the identity):
 *   B :  P_B     = P_n - h*grad f(X_n)
 *   A :  X_{n+1} = X_n + h*P_B
 *   C :  xi_n     = R_n C_n^T / (1^T R_n + eps)
 *        P_1      = P_B .* exp(-(h/2) xi_n)
 *        R_{n+1}  = e^{-a h} R_n + (1-e^{-a h})/(mu a) * [P_1]^2 1_n
 *        C_{n+1}  = e^{-a h} C_n + (1-e^{-a h})/(mu a) * 1_m^T [P_1]^2
 *        xi_{n+1} = R_{n+1} C_{n+1}^T / (1^T R_{n+1} + eps)
 *        P_{n+1}  = P_1 .* exp(-(h/2) xi_{n+1})
 *
 * f(X) = <X, A X>/2 with A = B B^T/d + 0.8 I, B Gaussian (reproducible seed).
 * Writes "t E S" at logarithmically spaced times.
 *
 * usage: rikfad m n eps alpha mu T h seed S0 outfile
 *   S0 = initial value of 1^T R = 1^T C (0 for the standard initialisation)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static unsigned long long rng_s;
static double u01(void){
    unsigned long long z = (rng_s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    return (double)(z >> 11) / 9007199254740992.0;
}
static double gauss(void){
    double u = u01(), v = u01();
    if (u < 1e-300) u = 1e-300;
    return sqrt(-2.0*log(u)) * cos(2.0*M_PI*v);
}

int main(int argc, char **argv){
    if (argc != 11){
        fprintf(stderr,"usage: %s m n eps alpha mu T h seed S0 out\n",argv[0]); return 1;
    }
    int    M = atoi(argv[1]), N = atoi(argv[2]);
    double EPS = atof(argv[3]), AL = atof(argv[4]), MU = atof(argv[5]);
    double T = atof(argv[6]), h = atof(argv[7]);
    rng_s = (unsigned long long)atoll(argv[8]);
    double S0 = atof(argv[9]);
    int D = M*N, i, j, k;

    double *B  = malloc((size_t)D*D*sizeof(double));
    double *A  = malloc((size_t)D*D*sizeof(double));
    for (i=0;i<D*D;i++) B[i] = gauss();
    for (i=0;i<D;i++) for (j=0;j<D;j++){
        double s = 0.0;
        for (k=0;k<D;k++) s += B[(size_t)i*D+k]*B[(size_t)j*D+k];
        A[(size_t)i*D+j] = s/D + (i==j ? 0.8 : 0.0);
    }
    free(B);

    double *X = calloc(D,sizeof(double)), *P = calloc(D,sizeof(double));
    double *R = calloc(M,sizeof(double)), *C = calloc(N,sizeof(double));
    double *G = calloc(D,sizeof(double));
    for (i=0;i<D;i++){ X[i] = 0.4*gauss(); P[i] = 0.3*gauss(); }
    for (i=0;i<M;i++) R[i] = S0/M;          /* keeps 1^T R = 1^T C = S0 */
    for (j=0;j<N;j++) C[j] = S0/N;

    const double ea = exp(-AL*h), cf = (1.0-exp(-AL*h))/(MU*AL);

    FILE *fp = fopen(argv[10],"w");
    if (!fp){ perror("fopen"); return 1; }
    fprintf(fp,"# t E S  (m=%d n=%d eps=%g alpha=%g mu=%g h=%g seed=%s) BACD splitting\n",
            M,N,EPS,AL,MU,h,argv[8]);
    fprintf(fp,"# S0=%g\n", S0);

    int nlog = 1200, ilog = 0;
    double lt0 = log(1.0), lt1 = log(T), next = 1.0, t = 0.0;
    long long nstep = (long long)(T/h);

    for (long long step = 0; step < nstep; step++){
        /* B: momentum kick */
        for (i=0;i<D;i++){
            double s = 0.0; const double *a = A + (size_t)i*D;
            for (j=0;j<D;j++) s += a[j]*X[j];
            G[i] = s;
        }
        for (i=0;i<D;i++) P[i] -= h*G[i];
        /* A: drift with the kicked momentum */
        for (i=0;i<D;i++) X[i] += h*P[i];
        /* C: symmetric half-damp around an exact factor update */
        double S = 0.0; for (i=0;i<M;i++) S += R[i];
        double den = S + EPS;
        if (den > 0.0)
            for (i=0;i<M;i++) for (j=0;j<N;j++)
                P[i*N+j] *= exp(-0.5*h*(R[i]*C[j]/den));
        for (i=0;i<M;i++){
            double s = 0.0; for (j=0;j<N;j++) s += P[i*N+j]*P[i*N+j];
            R[i] = ea*R[i] + cf*s;
        }
        for (j=0;j<N;j++){
            double s = 0.0; for (i=0;i<M;i++) s += P[i*N+j]*P[i*N+j];
            C[j] = ea*C[j] + cf*s;
        }
        S = 0.0; for (i=0;i<M;i++) S += R[i];
        den = S + EPS;
        if (den > 0.0)
            for (i=0;i<M;i++) for (j=0;j<N;j++)
                P[i*N+j] *= exp(-0.5*h*(R[i]*C[j]/den));
        /* D is the identity at gamma = 0 */
        t += h;

        if (ilog < nlog && t >= next){
            double e = 0.0;
            for (i=0;i<D;i++){
                double s = 0.0; const double *a = A + (size_t)i*D;
                for (j=0;j<D;j++) s += a[j]*X[j];
                e += 0.5*X[i]*s + 0.5*P[i]*P[i];
            }
            double SS = 0.0; for (i=0;i<M;i++) SS += R[i];
            fprintf(fp,"%.10e %.16e %.16e\n", t, e, SS);
            ilog++;
            next = exp(lt0 + (lt1-lt0)*ilog/(double)(nlog-1));
        }
    }
    fclose(fp);
    fprintf(stderr,"done: t=%g  steps=%lld\n", t, nstep);
    return 0;
}
