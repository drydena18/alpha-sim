/*
 * vdp_core.c
 *
 * Two coupled van der Pol oscillators exposed as a shared library for a
 * Python/PyQtGraph front end.
 *
 * This version is slightly more defensive and future-proof than the original:
 * - allocation checks in sim_create()
 * - explicit buffer length check in sim_get_buffers()
 * - lightweight getters for UI / replay mode
 * - optional exact reset via sim_reset_with_seed()
 * - clearer comments around threading and stochastic forcing
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double x;
    double v;
} OscState;

typedef struct {
    /* user parameters */
    double freq_a;
    double freq_b;
    double mu_a;
    double mu_b;
    double amp_a;
    double amp_b;
    double coupling;
    double noise;
    double alertness_a;
    double alertness_b;

    /* internal state */
    OscState sa;
    OscState sb;
    double time;

    /* ring buffers */
    double *buf_a;
    double *buf_b;
    double *buf_phase;
    int buf_len;
    int buf_ptr;

    /* latest point estimates */
    double last_phase_diff;

    /* RNG state */
    uint64_t rng;
    uint64_t rng_seed;
} SimCtx;

static uint64_t lcg_next(uint64_t *s)
{
    *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
    return *s;
}

static double rng_uniform_open01(uint64_t *s)
{
    /* 53-bit precision in (0,1). */
    uint64_t x = lcg_next(s);
    double u = (double)(x >> 11) / (double)(1ULL << 53);
    if (u <= 0.0) return 1e-300;
    if (u >= 1.0) return 1.0 - 1e-16;
    return u;
}

static double rng_normal(uint64_t *s)
{
    double u1 = rng_uniform_open01(s);
    double u2 = rng_uniform_open01(s);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static double effective_freq(double base_hz, double alertness)
{
    return base_hz + alertness * 1.5;
}

static double angular_freq(double base_hz, double alertness)
{
    return 2.0 * M_PI * effective_freq(base_hz, alertness);
}

static void deriv(
    double xa, double va,
    double xb, double vb,
    double wa, double wb,
    double mu_a, double mu_b,
    double amp_a, double amp_b,
    double K,
    double na, double nb,
    double *dxa, double *dva,
    double *dxb, double *dvb)
{
    *dxa = va;
    *dva = mu_a * (amp_a * amp_a - xa * xa) * va - wa * wa * xa
         + K * (xb - xa) + na;

    *dxb = vb;
    *dvb = mu_b * (amp_b * amp_b - xb * xb) * vb - wb * wb * xb
         + K * (xa - xb) + nb;
}

static double phase_diff_from_state(const SimCtx *ctx)
{
    double wa = angular_freq(ctx->freq_a, ctx->alertness_a);
    double wb = angular_freq(ctx->freq_b, ctx->alertness_b);

    if (wa == 0.0 || wb == 0.0) {
        return 0.0;
    }

    double ph_a = atan2(-ctx->sa.v / wa, ctx->sa.x);
    double ph_b = atan2(-ctx->sb.v / wb, ctx->sb.x);
    double d = ph_a - ph_b;

    while (d > M_PI)  d -= 2.0 * M_PI;
    while (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

/*
 * One fixed-step RK4 update.
 * Noise is piecewise constant within a timestep rather than re-sampled at
 * each RK sub-stage. That is acceptable for a demo / sandbox app.
 */
static void rk4_step(SimCtx *ctx, double dt)
{
    double wa = angular_freq(ctx->freq_a, ctx->alertness_a);
    double wb = angular_freq(ctx->freq_b, ctx->alertness_b);
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

    deriv(xa + 0.5 * dt * k1xa, va + 0.5 * dt * k1va,
          xb + 0.5 * dt * k1xb, vb + 0.5 * dt * k1vb,
          wa, wb, ctx->mu_a, ctx->mu_b, ctx->amp_a, ctx->amp_b, K, na, nb,
          &k2xa, &k2va, &k2xb, &k2vb);

    deriv(xa + 0.5 * dt * k2xa, va + 0.5 * dt * k2va,
          xb + 0.5 * dt * k2xb, vb + 0.5 * dt * k2vb,
          wa, wb, ctx->mu_a, ctx->mu_b, ctx->amp_a, ctx->amp_b, K, na, nb,
          &k3xa, &k3va, &k3xb, &k3vb);

    deriv(xa + dt * k3xa, va + dt * k3va,
          xb + dt * k3xb, vb + dt * k3vb,
          wa, wb, ctx->mu_a, ctx->mu_b, ctx->amp_a, ctx->amp_b, K, na, nb,
          &k4xa, &k4va, &k4xb, &k4vb);

    ctx->sa.x += (dt / 6.0) * (k1xa + 2.0 * k2xa + 2.0 * k3xa + k4xa);
    ctx->sa.v += (dt / 6.0) * (k1va + 2.0 * k2va + 2.0 * k3va + k4va);
    ctx->sb.x += (dt / 6.0) * (k1xb + 2.0 * k2xb + 2.0 * k3xb + k4xb);
    ctx->sb.v += (dt / 6.0) * (k1vb + 2.0 * k2vb + 2.0 * k3vb + k4vb);
    ctx->time += dt;

    ctx->last_phase_diff = phase_diff_from_state(ctx);
}

static void set_default_parameters(SimCtx *ctx)
{
    ctx->freq_a = 10.0;
    ctx->freq_b = 9.0;
    ctx->mu_a = 0.30;
    ctx->mu_b = 0.30;
    ctx->amp_a = 1.00;
    ctx->amp_b = 1.00;
    ctx->coupling = 0.05;
    ctx->noise = 0.05;
    ctx->alertness_a = 0.50;
    ctx->alertness_b = 0.50;
}

static void clear_buffers(SimCtx *ctx)
{
    if (!ctx) return;
    if (ctx->buf_a) memset(ctx->buf_a, 0, (size_t)ctx->buf_len * sizeof(double));
    if (ctx->buf_b) memset(ctx->buf_b, 0, (size_t)ctx->buf_len * sizeof(double));
    if (ctx->buf_phase) memset(ctx->buf_phase, 0, (size_t)ctx->buf_len * sizeof(double));
    ctx->buf_ptr = 0;
}

static void reset_state_only(SimCtx *ctx)
{
    ctx->sa.x = 1.0;
    ctx->sa.v = 0.0;
    ctx->sb.x = -1.0;
    ctx->sb.v = 0.0;
    ctx->time = 0.0;
    ctx->last_phase_diff = 0.0;
}

SimCtx *sim_create(int buf_len)
{
    if (buf_len <= 0) return NULL;

    SimCtx *ctx = (SimCtx *)calloc(1, sizeof(SimCtx));
    if (!ctx) return NULL;

    ctx->buf_len = buf_len;
    ctx->buf_a = (double *)calloc((size_t)buf_len, sizeof(double));
    ctx->buf_b = (double *)calloc((size_t)buf_len, sizeof(double));
    ctx->buf_phase = (double *)calloc((size_t)buf_len, sizeof(double));

    if (!ctx->buf_a || !ctx->buf_b || !ctx->buf_phase) {
        free(ctx->buf_a);
        free(ctx->buf_b);
        free(ctx->buf_phase);
        free(ctx);
        return NULL;
    }

    set_default_parameters(ctx);
    reset_state_only(ctx);
    clear_buffers(ctx);

    ctx->rng_seed = 12345678901234567ULL;
    ctx->rng = ctx->rng_seed;
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

void sim_set_seed(SimCtx *ctx, uint64_t seed)
{
    if (!ctx) return;
    ctx->rng_seed = seed ? seed : 12345678901234567ULL;
    ctx->rng = ctx->rng_seed;
}

void sim_reset(SimCtx *ctx)
{
    if (!ctx) return;
    reset_state_only(ctx);
    clear_buffers(ctx);
}

void sim_reset_with_seed(SimCtx *ctx)
{
    if (!ctx) return;
    reset_state_only(ctx);
    clear_buffers(ctx);
    ctx->rng = ctx->rng_seed;
}

void sim_advance(SimCtx *ctx, int steps, double dt)
{
    if (!ctx || steps <= 0 || dt <= 0.0) return;

    for (int i = 0; i < steps; ++i) {
        rk4_step(ctx, dt);
        ctx->buf_a[ctx->buf_ptr] = ctx->sa.x;
        ctx->buf_b[ctx->buf_ptr] = ctx->sb.x;
        ctx->buf_phase[ctx->buf_ptr] = ctx->last_phase_diff;
        ctx->buf_ptr = (ctx->buf_ptr + 1) % ctx->buf_len;
    }
}

void sim_get_buffers(
    SimCtx *ctx,
    double *out_a,
    double *out_b,
    double *out_phase,
    int out_len)
{
    if (!ctx || !out_a || !out_b || !out_phase) return;
    if (out_len < ctx->buf_len) return;

    int n = ctx->buf_len;
    int ptr = ctx->buf_ptr;
    for (int i = 0; i < n; ++i) {
        int idx = (ptr + i) % n;
        out_a[i] = ctx->buf_a[idx];
        out_b[i] = ctx->buf_b[idx];
        out_phase[i] = ctx->buf_phase[idx];
    }
}

double sim_sync_index(SimCtx *ctx)
{
    if (!ctx || ctx->buf_len <= 0) return 0.0;

    double sc = 0.0;
    double ss = 0.0;
    int n = ctx->buf_len;

    for (int i = 0; i < n; ++i) {
        sc += cos(ctx->buf_phase[i]);
        ss += sin(ctx->buf_phase[i]);
    }

    sc /= (double)n;
    ss /= (double)n;
    return sqrt(sc * sc + ss * ss);
}

/* Setters: call when not concurrently advancing the simulation. */
void sim_set_freq_a(SimCtx *c, double v)       { if (c) c->freq_a = v; }
void sim_set_freq_b(SimCtx *c, double v)       { if (c) c->freq_b = v; }
void sim_set_mu_a(SimCtx *c, double v)         { if (c) c->mu_a = v; }
void sim_set_mu_b(SimCtx *c, double v)         { if (c) c->mu_b = v; }
void sim_set_amp_a(SimCtx *c, double v)        { if (c) c->amp_a = v; }
void sim_set_amp_b(SimCtx *c, double v)        { if (c) c->amp_b = v; }
void sim_set_coupling(SimCtx *c, double v)     { if (c) c->coupling = v; }
void sim_set_noise(SimCtx *c, double v)        { if (c) c->noise = v; }
void sim_set_alertness_a(SimCtx *c, double v)  { if (c) c->alertness_a = v; }
void sim_set_alertness_b(SimCtx *c, double v)  { if (c) c->alertness_b = v; }

/* Getters */
double sim_get_time(SimCtx *c)                 { return c ? c->time : 0.0; }
double sim_get_phase_diff(SimCtx *c)           { return c ? c->last_phase_diff : 0.0; }
double sim_get_freq_a(SimCtx *c)               { return c ? c->freq_a : 0.0; }
double sim_get_freq_b(SimCtx *c)               { return c ? c->freq_b : 0.0; }
double sim_get_alertness_a(SimCtx *c)          { return c ? c->alertness_a : 0.0; }
double sim_get_alertness_b(SimCtx *c)          { return c ? c->alertness_b : 0.0; }
double sim_get_effective_freq_a(SimCtx *c)     { return c ? effective_freq(c->freq_a, c->alertness_a) : 0.0; }
double sim_get_effective_freq_b(SimCtx *c)     { return c ? effective_freq(c->freq_b, c->alertness_b) : 0.0; }
int sim_get_buffer_len(SimCtx *c)              { return c ? c->buf_len : 0; }