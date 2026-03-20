/*
 * vdp_core.c
 * Van der Pol alpha-wave oscillator engine.
 * Compiled as a shared library and called from Python via ctypes.
 *
 * Model (per oscillator):
 *   ẍ = μ(a² - x²)ẋ  −  ω²x  +  K(x_other − x)  +  σ·noise
 *
 * Alertness shifts ω upward:  ω = 2π(f0 + alertness * 1.5)
 *
 * Integration: 4th-order Runge-Kutta, fixed step dt.
 */

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ── state ─────────────────────────────────────────────────────────────── */

typedef struct {
    double x;    /* position (proxy for μV amplitude) */
    double v;    /* velocity */
} OscState;

typedef struct {
    /* oscillator parameters */
    double freq_a;      /* natural frequency A  (Hz)  */
    double freq_b;      /* natural frequency B  (Hz)  */
    double mu_a;        /* van der Pol damping A       */
    double mu_b;        /* van der Pol damping B       */
    double amp_a;       /* limit-cycle amplitude A     */
    double amp_b;       /* limit-cycle amplitude B     */
    double coupling;    /* diffusive coupling K        */
    double noise;       /* noise std-dev σ             */
    double alertness_a; /* alertness [0,1] → +1.5 Hz  */
    double alertness_b;

    /* internal state */
    OscState sa;
    OscState sb;
    double   time;

    /* ring buffers (allocated externally) */
    double  *buf_a;
    double  *buf_b;
    double  *buf_phase;
    int      buf_len;
    int      buf_ptr;

    /* simple LCG for fast noise */
    uint64_t rng;
} SimCtx;

/* ── RNG ───────────────────────────────────────────────────────────────── */

static double rng_normal(uint64_t *s)
{
    /* Box-Muller using two uniform LCG draws */
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(*s >> 11) / (double)(1ULL << 53);
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(*s >> 11) / (double)(1ULL << 53);
    u1 = (u1 < 1e-300) ? 1e-300 : u1;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ── derivatives ───────────────────────────────────────────────────────── */

static void deriv(
    double  xa,  double  va,
    double  xb,  double  vb,
    double  wa,  double  wb,
    double  mu_a, double  mu_b,
    double  amp_a, double amp_b,
    double  K,
    double  na,  double  nb,
    double *dxa, double *dva,
    double *dxb, double *dvb)
{
    *dxa = va;
    *dva = mu_a * (amp_a*amp_a - xa*xa) * va  -  wa*wa * xa
           + K * (xb - xa)  +  na;

    *dxb = vb;
    *dvb = mu_b * (amp_b*amp_b - xb*xb) * vb  -  wb*wb * xb
           + K * (xa - xb)  +  nb;
}

/* ── RK4 step ──────────────────────────────────────────────────────────── */

static void rk4_step(SimCtx *ctx, double dt)
{
    double wa = 2.0 * M_PI * (ctx->freq_a + ctx->alertness_a * 1.5);
    double wb = 2.0 * M_PI * (ctx->freq_b + ctx->alertness_b * 1.5);
    double K  = ctx->coupling;
    double na = rng_normal(&ctx->rng) * ctx->noise;
    double nb = rng_normal(&ctx->rng) * ctx->noise;

    double xa = ctx->sa.x, va = ctx->sa.v;
    double xb = ctx->sb.x, vb = ctx->sb.v;

    double k1xa, k1va, k1xb, k1vb;
    double k2xa, k2va, k2xb, k2vb;
    double k3xa, k3va, k3xb, k3vb;
    double k4xa, k4va, k4xb, k4vb;

    deriv(xa, va, xb, vb, wa, wb,
          ctx->mu_a, ctx->mu_b, ctx->amp_a, ctx->amp_b, K, na, nb,
          &k1xa, &k1va, &k1xb, &k1vb);

    deriv(xa + dt/2*k1xa, va + dt/2*k1va,
          xb + dt/2*k1xb, vb + dt/2*k1vb,
          wa, wb, ctx->mu_a, ctx->mu_b, ctx->amp_a, ctx->amp_b, K, na, nb,
          &k2xa, &k2va, &k2xb, &k2vb);

    deriv(xa + dt/2*k2xa, va + dt/2*k2va,
          xb + dt/2*k2xb, vb + dt/2*k2vb,
          wa, wb, ctx->mu_a, ctx->mu_b, ctx->amp_a, ctx->amp_b, K, na, nb,
          &k3xa, &k3va, &k3xb, &k3vb);

    deriv(xa + dt*k3xa, va + dt*k3va,
          xb + dt*k3xb, vb + dt*k3vb,
          wa, wb, ctx->mu_a, ctx->mu_b, ctx->amp_a, ctx->amp_b, K, na, nb,
          &k4xa, &k4va, &k4xb, &k4vb);

    ctx->sa.x += dt/6.0 * (k1xa + 2*k2xa + 2*k3xa + k4xa);
    ctx->sa.v += dt/6.0 * (k1va + 2*k2va + 2*k3va + k4va);
    ctx->sb.x += dt/6.0 * (k1xb + 2*k2xb + 2*k3xb + k4xb);
    ctx->sb.v += dt/6.0 * (k1vb + 2*k2vb + 2*k3vb + k4vb);
    ctx->time += dt;
}

/* ── phase helper ──────────────────────────────────────────────────────── */

static double phase_diff(SimCtx *ctx)
{
    double wa = 2.0 * M_PI * (ctx->freq_a + ctx->alertness_a * 1.5);
    double wb = 2.0 * M_PI * (ctx->freq_b + ctx->alertness_b * 1.5);
    double phA = atan2(-ctx->sa.v / wa,  ctx->sa.x);
    double phB = atan2(-ctx->sb.v / wb,  ctx->sb.x);
    double d   = phA - phB;
    while (d >  M_PI) d -= 2.0 * M_PI;
    while (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

/* ── public API ─────────────────────────────────────────────────────────── */

/* Allocate and return a new context. Call sim_free() when done. */
SimCtx *sim_create(int buf_len)
{
    SimCtx *ctx = (SimCtx *)calloc(1, sizeof(SimCtx));
    if (!ctx) return NULL;

    ctx->buf_a     = (double *)calloc(buf_len, sizeof(double));
    ctx->buf_b     = (double *)calloc(buf_len, sizeof(double));
    ctx->buf_phase = (double *)calloc(buf_len, sizeof(double));
    ctx->buf_len   = buf_len;

    /* defaults */
    ctx->freq_a      = 10.0;
    ctx->freq_b      =  9.0;
    ctx->mu_a        =  0.3;
    ctx->mu_b        =  0.3;
    ctx->amp_a       =  1.0;
    ctx->amp_b       =  1.0;
    ctx->coupling    =  0.05;
    ctx->noise       =  0.05;
    ctx->alertness_a =  0.5;
    ctx->alertness_b =  0.5;

    ctx->sa.x =  1.0; ctx->sa.v = 0.0;
    ctx->sb.x = -1.0; ctx->sb.v = 0.0;

    ctx->rng = 12345678901234567ULL;
    return ctx;
}

void sim_free(SimCtx *ctx)
{
    if (!ctx) return;
    free(ctx->buf_a);
    free(ctx->buf_b);
    free(ctx->buf_phase);
    free(ctx);
}

void sim_reset(SimCtx *ctx)
{
    ctx->sa.x =  1.0; ctx->sa.v = 0.0;
    ctx->sb.x = -1.0; ctx->sb.v = 0.0;
    ctx->time = 0.0;
    ctx->buf_ptr = 0;
    memset(ctx->buf_a,     0, ctx->buf_len * sizeof(double));
    memset(ctx->buf_b,     0, ctx->buf_len * sizeof(double));
    memset(ctx->buf_phase, 0, ctx->buf_len * sizeof(double));
}

/*
 * Advance the simulation by `steps` RK4 steps of size `dt` seconds.
 * After each step, one sample is written to the ring buffers.
 * Typically called once per display frame.
 */
void sim_advance(SimCtx *ctx, int steps, double dt)
{
    for (int i = 0; i < steps; i++) {
        rk4_step(ctx, dt);
        ctx->buf_a    [ctx->buf_ptr] = ctx->sa.x;
        ctx->buf_b    [ctx->buf_ptr] = ctx->sb.x;
        ctx->buf_phase[ctx->buf_ptr] = phase_diff(ctx);
        ctx->buf_ptr = (ctx->buf_ptr + 1) % ctx->buf_len;
    }
}

/* Copy ring-buffer contents into flat arrays in time-order. */
void sim_get_buffers(SimCtx *ctx,
                     double *out_a, double *out_b, double *out_phase)
{
    int n   = ctx->buf_len;
    int ptr = ctx->buf_ptr;
    for (int i = 0; i < n; i++) {
        int idx     = (ptr + i) % n;
        out_a    [i] = ctx->buf_a    [idx];
        out_b    [i] = ctx->buf_b    [idx];
        out_phase[i] = ctx->buf_phase[idx];
    }
}

/* Kuramoto synchrony index from the phase-diff history. */
double sim_sync_index(SimCtx *ctx)
{
    double sc = 0.0, ss = 0.0;
    int n = ctx->buf_len;
    for (int i = 0; i < n; i++) {
        sc += cos(ctx->buf_phase[i]);
        ss += sin(ctx->buf_phase[i]);
    }
    sc /= n; ss /= n;
    return sqrt(sc*sc + ss*ss);
}

/* Parameter setters (safe to call from any thread before sim_advance) */
void sim_set_freq_a     (SimCtx *c, double v){ c->freq_a      = v; }
void sim_set_freq_b     (SimCtx *c, double v){ c->freq_b      = v; }
void sim_set_mu_a       (SimCtx *c, double v){ c->mu_a        = v; }
void sim_set_mu_b       (SimCtx *c, double v){ c->mu_b        = v; }
void sim_set_amp_a      (SimCtx *c, double v){ c->amp_a       = v; }
void sim_set_amp_b      (SimCtx *c, double v){ c->amp_b       = v; }
void sim_set_coupling   (SimCtx *c, double v){ c->coupling    = v; }
void sim_set_noise      (SimCtx *c, double v){ c->noise       = v; }
void sim_set_alertness_a(SimCtx *c, double v){ c->alertness_a = v; }
void sim_set_alertness_b(SimCtx *c, double v){ c->alertness_b = v; }
double sim_get_time     (SimCtx *c)          { return c->time;     }