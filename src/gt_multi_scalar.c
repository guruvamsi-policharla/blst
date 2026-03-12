/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * GT multi-scalar multiplication via Pippenger's algorithm.
 *
 * Computes product: g_1^s_1 * g_2^s_2 * ... * g_n^s_n
 * where g_i are GT elements (Fp12) and s_i are scalars.
 *
 * GT is a multiplicative group, so:
 *   - "addition" = Fp12 multiplication
 *   - "doubling" = Fp12 squaring
 *   - "identity" = Fp12 one
 *   - "negation" = Fp12 conjugation
 *
 * All-zero memory is used as a sentinel for "empty" (uninitialized)
 * buckets/accumulators, since the Fp12 identity (one) is not all-zeros.
 */

#include "fields.h"
#include "ec_mult.h"

/*
 * Window size calculation, same heuristic as EC Pippenger.
 * Could be tuned separately for GT since Fp12 ops are more expensive
 * relative to EC ops, but the bucket/window tradeoff is similar.
 */
static size_t gt_pippenger_window_size(size_t npoints)
{
    size_t wbits;

    for (wbits = 0; npoints >>= 1; wbits++) ;

    if (wbits > 12)
        return wbits - 3;
    else if (wbits > 8)
        return wbits - 2;
    else if (wbits > 4)
        return wbits - 1;

    return wbits ? 2 : 1;
}

/*
 * Multiply two Fp12 elements, treating all-zero as identity.
 * Result is stored in |ret|. Either or both of |a|, |b| may be all-zero.
 */
static void gt_mul_or_copy(vec384fp12 ret, const vec384fp12 a,
                                           const vec384fp12 b)
{
    if (vec_is_zero(a, sizeof(vec384fp12))) {
        vec_copy(ret, b, sizeof(vec384fp12));
    } else if (vec_is_zero(b, sizeof(vec384fp12))) {
        vec_copy(ret, a, sizeof(vec384fp12));
    } else {
        mul_fp12(ret, a, b);
    }
}

/*
 * Accumulate a GT element into a bucket based on Booth-encoded index.
 * Buckets use all-zero memory as "empty" sentinel.
 */
static void gt_bucket(vec384fp12 buckets[], limb_t booth_idx,
                       size_t wbits, const vec384fp12 point)
{
    bool_t booth_sign = (booth_idx >> wbits) & 1;

    booth_idx &= ((limb_t)1 << wbits) - 1;
    if (booth_idx--) {
        vec384fp12 *bucket = &buckets[booth_idx];

        if (vec_is_zero(*bucket, sizeof(vec384fp12))) {
            /* Empty bucket: copy point, possibly conjugated */
            vec_copy(*bucket, point, sizeof(vec384fp12));
            if (booth_sign) conjugate_fp12(*bucket);
        } else {
            if (booth_sign) {
                /* Multiply by conjugate of point */
                vec384fp12 tmp;
                vec_copy(tmp, point, sizeof(vec384fp12));
                conjugate_fp12(tmp);
                mul_fp12(*bucket, *bucket, tmp);
            } else {
                mul_fp12(*bucket, *bucket, point);
            }
        }
    }
}

static void gt_prefetch(const vec384fp12 buckets[], limb_t booth_idx,
                         size_t wbits)
{
    booth_idx &= ((limb_t)1 << wbits) - 1;
    if (booth_idx--)
        vec_prefetch(&buckets[booth_idx], sizeof(buckets[booth_idx]));
}

/*
 * Integrate buckets: compute weighted product
 *   bucket[0]^1 * bucket[1]^2 * ... * bucket[n-1]^n
 * using the running-sum technique.
 */
static void gt_integrate_buckets(vec384fp12 out, vec384fp12 buckets[],
                                                 size_t wbits)
{
    vec384fp12 acc, ret;
    size_t n = (size_t)1 << wbits;

    /* Start from the topmost bucket */
    vec_copy(acc, buckets[--n], sizeof(acc));
    vec_copy(ret, buckets[n], sizeof(ret));
    vec_zero(buckets[n], sizeof(vec384fp12));
    while (n--) {
        gt_mul_or_copy(acc, acc, buckets[n]);
        gt_mul_or_copy(ret, ret, acc);
        vec_zero(buckets[n], sizeof(vec384fp12));
    }
    vec_copy(out, ret, sizeof(vec384fp12));
}

/*
 * Process one window (tile) of scalar bits across all points.
 */
static void gt_tile_pippenger(vec384fp12 ret,
                               const vec384fp12 *const points[],
                               size_t npoints,
                               const byte *const scalars[], size_t nbits,
                               vec384fp12 buckets[],
                               size_t bit0, size_t wbits, size_t cbits)
{
    limb_t wmask, wval, wnxt;
    size_t i, z, nbytes;
    const byte *scalar = *scalars++;
    const vec384fp12 *point = *points++;

    nbytes = (nbits + 7) / 8;
    wmask = ((limb_t)1 << (wbits + 1)) - 1;
    z = is_zero(bit0);
    bit0 -= z ^ 1; wbits += z ^ 1;
    wval = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    wval = booth_encode(wval, cbits);
    scalar = *scalars ? *scalars++ : scalar + nbytes;
    wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    wnxt = booth_encode(wnxt, cbits);
    npoints--;  /* account for prefetch */

    gt_bucket(buckets, wval, cbits, *point);
    for (i = 1; i < npoints; i++) {
        wval = wnxt;
        scalar = *scalars ? *scalars++ : scalar + nbytes;
        wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
        wnxt = booth_encode(wnxt, cbits);
        gt_prefetch(buckets, wnxt, cbits);
        point = *points ? *points++ : point + 1;
        gt_bucket(buckets, wval, cbits, *point);
    }
    point = *points ? *points++ : point + 1;
    gt_bucket(buckets, wnxt, cbits, *point);
    gt_integrate_buckets(ret, buckets, cbits - 1);
}

/*
 * Simple square-and-multiply exponentiation for single GT element.
 * Computes ret = base^scalar where scalar has nbits bits.
 */
static void gt_exp(vec384fp12 ret, const vec384fp12 base,
                                   const byte *scalar, size_t nbits)
{
    size_t i;
    limb_t bit;

    if (nbits == 0) {
        vec_copy(ret, BLS12_381_Rx.p12, sizeof(vec384fp12));
        return;
    }

    /* Find the topmost set bit */
    for (i = nbits; i > 0; i--) {
        bit = (scalar[(i - 1) / 8] >> ((i - 1) % 8)) & 1;
        if (bit) break;
    }

    if (i == 0) {
        /* Scalar is zero, return one */
        vec_copy(ret, BLS12_381_Rx.p12, sizeof(vec384fp12));
        return;
    }

    /* Initialize with base */
    vec_copy(ret, base, sizeof(vec384fp12));
    i--;

    /* Square-and-multiply from the second-highest bit down */
    while (i > 0) {
        i--;
        sqr_fp12(ret, ret);
        bit = (scalar[i / 8] >> (i % 8)) & 1;
        if (bit) mul_fp12(ret, ret, base);
    }
}

/*
 * Main Pippenger MSM for GT elements.
 *
 * Computes ret = points[0]^scalars[0] * ... * points[n-1]^scalars[n-1]
 */
static void gt_mult_pippenger(vec384fp12 ret,
                               const vec384fp12 *const points[],
                               size_t npoints,
                               const byte *const scalars[], size_t nbits,
                               vec384fp12 buckets[], size_t window)
{
    size_t i, wbits, cbits, bit0 = nbits;
    vec384fp12 tile;

    window = window ? window : gt_pippenger_window_size(npoints);
    vec_zero(buckets, sizeof(vec384fp12) << (window - 1));
    vec_zero(ret, sizeof(vec384fp12));  /* all-zero = empty sentinel */

    /* top excess bits modulo target window size */
    wbits = nbits % window;    /* yes, it may be zero */
    cbits = wbits + 1;
    while (bit0 -= wbits) {
        gt_tile_pippenger(tile, points, npoints, scalars, nbits,
                                buckets, bit0, wbits, cbits);
        gt_mul_or_copy(ret, ret, tile);
        for (i = 0; i < window; i++) {
            if (!vec_is_zero(ret, sizeof(vec384fp12)))
                sqr_fp12(ret, ret);
        }
        cbits = wbits = window;
    }
    gt_tile_pippenger(tile, points, npoints, scalars, nbits,
                            buckets, 0, wbits, cbits);
    gt_mul_or_copy(ret, ret, tile);

    /* If result is still empty (e.g. all scalars were zero), return one */
    if (vec_is_zero(ret, sizeof(vec384fp12)))
        vec_copy(ret, BLS12_381_Rx.p12, sizeof(vec384fp12));
}

/*
 * Public API
 */
size_t blst_fp12s_mult_pippenger_scratch_sizeof(size_t npoints)
{   return sizeof(vec384fp12) << (gt_pippenger_window_size(npoints) - 1);   }

void blst_fp12s_mult_pippenger(vec384fp12 ret,
                                const vec384fp12 *const points[],
                                size_t npoints,
                                const byte *const scalars[], size_t nbits,
                                limb_t *scratch)
{
    if (npoints == 0) {
        vec_copy(ret, BLS12_381_Rx.p12, sizeof(vec384fp12));
        return;
    }
    if (npoints == 1) {
        gt_exp(ret, *points[0], scalars[0], nbits);
        return;
    }
    gt_mult_pippenger(ret, points, npoints, scalars, nbits,
                      (vec384fp12 *)scratch, 0);
}

void blst_fp12s_tile_pippenger(vec384fp12 ret,
                                const vec384fp12 *const points[],
                                size_t npoints,
                                const byte *const scalars[], size_t nbits,
                                limb_t *scratch,
                                size_t bit0, size_t window)
{
    size_t wbits, cbits;

    if (bit0 + window > nbits)  wbits = nbits - bit0, cbits = wbits + 1;
    else                        wbits = cbits = window;
    gt_tile_pippenger(ret, points, npoints, scalars, nbits,
                           (vec384fp12 *)scratch, bit0, wbits, cbits);
}
