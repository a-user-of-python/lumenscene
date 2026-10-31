// Shaders.metal — LumenScene's renderer.
//
// pathTraceKernel is a Monte Carlo path tracer running on the M3's
// hardware ray tracing units (metal_raytracing intersector over a
// primitive acceleration structure built from the whole scene's
// triangles, baked into world space):
//
//   * Emissive surfaces are real area lights. Direct light from
//     them is sampled explicitly (next-event estimation over an
//     area-weighted CDF of emissive triangles), so a glowing sign
//     throws coloured light onto walls and floor at low sample
//     counts; emissive hits on specular/primary paths add their
//     radiance directly.
//   * Glass is a dielectric: Fresnel (Schlick, from IOR) chooses
//     reflection vs refraction; roughness jitters both directions
//     (clear -> frosted as roughness rises); light travelling
//     INSIDE the glass is attenuated by Beer-Lambert absorption
//     (tint raised to the travelled distance), so thick glass
//     darkens and colours more than thin glass. A per-path
//     refraction budget disciplines transmission work: past the
//     budget, deeper glass layers shade as tinted diffuse instead
//     of spawning more transmissive paths (glass-heavy views stay
//     interactive; the budget is generous at rest).
//   * Textures are procedural PBR sets (albedo / tangent normal /
//     roughness+height) sampled per material slice, with one-step
//     parallax from the height channel; imported models (no UVs)
//     use triplanar world-space sampling INCLUDING triplanar
//     normal mapping, so texture depth works there too.
//   * Everything else is Lambertian with cosine-weighted bounces
//     and Russian roulette.
//
// Temporal reprojection (v2): the accumulation texture's alpha
// channel stores each pixel's effective sample count. A cheap
// guide pass writes every pixel's primary-hit world position;
// when the camera moves, reprojectKernel seeds the new frame's
// accumulation from the previous frame wherever the world position
// seen through the old camera matches the one seen now — diffuse
// and emissive first hits only (glass and environment are
// view-dependent and always restart). Motion therefore keeps most
// of the converged image instead of restarting from noise. While
// still, reprojection is the identity map and accumulation simply
// continues, so the settled image is the full-quality one. At the
// sample cap the accumulator becomes a moving average (oldest
// weight decays), which keeps Play mode smooth instead of noisy.
//
// V3 additions:
//   * Firefly clamp: each new sample is capped against the
//     pixel's own running average before accumulation, so one
//     hot path can't sparkle a settled pixel.
//   * Spatial denoise: a 3x3 cross-bilateral pass (guide position
//     + guide normal + luminance) over the accumulation, faded
//     out as pixels accumulate — interactive frames and Play
//     read clean; the settled image is displayed raw.
//   * Weather: wetness darkens up-facing surfaces and rolls
//     their bounces toward glossy sheen; snow whitens them and
//     flattens the shading normal. Precipitation (rain streaks /
//     snow sprites) is a separate camera-box overlay pass.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// Must match Renderer.swift's GPUUniforms / GPUMaterial byte layout.
struct Uniforms {
    float3 camOrigin;
    float3 camRight;
    float3 camUp;
    float3 camForward;
    float2 tanFovAspect;   // x: tan(half vertical fov), y: aspect
    uint frameSeed;
    uint maxBounces;
    uint triCount;
    uint emissiveCount;
    float totalEmissiveArea;
    float envIntensity;
    float historyCap;          // max effective samples kept per pixel
    uint glassRefractBudget;   // max refraction events per path
    float wetLevel;            // v3: 0 dry .. 1 soaked (up-facing)
    float snowLevel;           // v3: 0 none .. 1 full cover (up-facing)
    float dynTime;             // v3.2: weather clock (retained for layout; v3.3: unused by the trace kernel)
    float rainFrac;            // v3.2: live rain intensity (retained for layout; v3.3: unused by the trace kernel)
};

struct Material {
    float4 baseColor;    // rgb; a unused
    float4 emissive;     // rgb colour; a = intensity
    float4 glassParams;  // x: transmission, y: roughness, z: ior, w unused
    float4 tint;         // rgb; a = textureScale
    uint texIndex;       // 0 none, else texture array slice+1 (slice = texIndex-1)
    uint flags;          // bit0 emissive, bit1 glass, bit2 triplanar, bit3 fit
    uint2 pad;
};

constant uint FLAG_EMISSIVE = 1u;
constant uint FLAG_GLASS = 2u;
constant uint FLAG_TRIPLANAR = 4u;
constant uint FLAG_FIT = 8u;

// MARK: - RNG (xorshift128, seeded per pixel + frame)

struct RNG {
    uint s0, s1;
    uint next() {
        uint x = s0;
        uint y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s1 + y;
    }
    float nextFloat() { return float(next() >> 8) * (1.0f / 16777216.0f); }
};

static uint hash32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// MARK: - Weather noise (v3.1: shaped wet / snow fields)
//
// 2D value noise in [0,1] over an integer lattice. Corner values
// come from hash32 of the lattice coords mixed with two large
// odd constants; uint(ix) on a negative coord wraps modulo 2^32,
// which is fine here — the wrap IS the hash input.
static float wnoise2(float2 p) {
    int2 ip = int2(floor(p));
    float2 f = fract(p);
    float2 s = f * f * (3.0f - 2.0f * f);
    uint h00 = hash32((uint(ip.x) * 0x8da6b343u) ^ (uint(ip.y) * 0xd8163841u));
    uint h10 = hash32((uint(ip.x + 1) * 0x8da6b343u) ^ (uint(ip.y) * 0xd8163841u));
    uint h01 = hash32((uint(ip.x) * 0x8da6b343u) ^ (uint(ip.y + 1) * 0xd8163841u));
    uint h11 = hash32((uint(ip.x + 1) * 0x8da6b343u) ^ (uint(ip.y + 1) * 0xd8163841u));
    float v00 = float(h00 >> 8) * (1.0f / 16777216.0f);
    float v10 = float(h10 >> 8) * (1.0f / 16777216.0f);
    float v01 = float(h01 >> 8) * (1.0f / 16777216.0f);
    float v11 = float(h11 >> 8) * (1.0f / 16777216.0f);
    return mix(mix(v00, v10, s.x), mix(v01, v11, s.x), s.y);
}

// 3-octave fbm of wnoise2, output roughly [0,1].
static float wfbm2(float2 p) {
    return 0.55f * wnoise2(p)
         + 0.30f * wnoise2(p * 2.0f)
         + 0.15f * wnoise2(p * 4.0f);
}

// Metaball puddle potential (v3.3): summed smooth kernels
// around the jittered points of the 3x3 cell neighbourhood,
// each cell weighted by a fixed function of its hash random
// (eligible cells weigh ~1, ineligible ~0 — the random is
// uniform BY CONSTRUCTION, so a coverage threshold on the
// potential selects the same share of cells anywhere in the
// world, which is what guarantees puddles on a perfectly flat
// surface). The sample position is domain-warped by a
// low-frequency fbm pair first, so level sets of the potential
// are irregular, rounded blobs that grow and merge like water
// — never the straight Voronoi edges a nearest-distance (F1)
// radius cut produced in v3.2 (Sean's "square slab" puddles).
// Evaluated ONLY by the field bake kernel below; the path
// tracer reads the baked result from a texture.
static float wpuddlePotential(float2 p) {
    float2 warp = float2(wfbm2(p * 0.16f + float2(3.1f, 7.7f)),
                         wfbm2(p * 0.16f + float2(11.9f, 1.3f)));
    float2 q = p + (warp - 0.5f) * 1.0f; // up to ±0.5 cells of wobble
    int2 ip = int2(floor(q));
    float2 fp = fract(q);
    float sum = 0.0f;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 c = ip + int2(i, j);
            uint h = hash32((uint(c.x) * 0x8da6b343u) ^ (uint(c.y) * 0xd8163841u));
            float r0 = float(h >> 8) * (1.0f / 16777216.0f);
            uint h2 = hash32(h ^ 0x9e3779b9u);
            float r1 = float(h2 >> 8) * (1.0f / 16777216.0f);
            uint h3 = hash32(h2 ^ 0x85ebca6bu);
            float cr = float(h3 >> 8) * (1.0f / 16777216.0f);
            float a = pow(clamp((0.58f - cr) / 0.58f, 0.0f, 1.0f), 0.7f);
            if (a <= 0.001f) continue;
            float2 pt = float2(float(i), float(j)) + float2(r0, r1);
            float d = length(pt - fp);
            float s = clamp(1.0f - (d / 1.45f) * (d / 1.45f), 0.0f, 1.0f);
            sum += a * s * s;
        }
    }
    return sum;
}

// MARK: - Helpers

static float3 cosineDirection(float3 n, thread RNG& rng) {
    float u = rng.nextFloat(), v = rng.nextFloat();
    float r = sqrt(u), phi = 2.0f * M_PI_F * v;
    float3 t = normalize(abs(n.y) < 0.99f ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    float3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0f, 1.0f - u)));
}

static float3 jitterDirection(float3 dir, float amount, thread RNG& rng) {
    if (amount <= 0.0001f) return dir;
    float3 rnd = float3(rng.nextFloat() * 2 - 1, rng.nextFloat() * 2 - 1, rng.nextFloat() * 2 - 1);
    return normalize(dir + rnd * amount);
}

static float fresnelSchlick(float cosTheta, float f0) {
    float m = clamp(1.0f - cosTheta, 0.0f, 1.0f);
    return f0 + (1.0f - f0) * m * m * m * m * m;
}

static float3 envColor(float3 dir, float intensity) {
    float t = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float3 sky = mix(float3(0.035, 0.035, 0.04), float3(0.13, 0.16, 0.22), t);
    return sky * intensity;
}

// MARK: - Weather field bake (v3.3)
//
// One-shot bake at renderer init: the level-independent
// weather field over a 64 m world window centred on the
// origin. R = puddle metaball potential, G = wet pooling fbm,
// B = snow drift fbm, A = broad pooling fbm — the exact fields
// the v3.1/v3.2 path kernel used to evaluate procedurally per
// hit per sample (~75 hash evaluations). Wet/snow LEVELS are
// thresholds applied analytically in the path kernel, so this
// texture never needs refreshing when build-up changes.
kernel void weatherFieldKernel(
    texture2d<float, access::write> field [[texture(0)]],
    uint2 pixel [[thread_position_in_grid]])
{
    uint w = field.get_width();
    uint h = field.get_height();
    if (pixel.x >= w || pixel.y >= h) return;
    float2 uv = (float2(pixel) + 0.5f) / float2(float(w), float(h));
    float2 xz = (uv - 0.5f) * 64.0f;
    float pot = wpuddlePotential(xz * (1.0f / 1.15f));
    float wetPool = wfbm2(xz * 1.3f + float2(4.7f, 9.1f));
    float drift = wfbm2(xz * 0.5f + float2(9.17f, 27.3f));
    float broad = wfbm2(xz * 0.32f + float2(17.31f, 3.97f));
    field.write(float4(pot, wetPool, drift, broad), pixel);
}

// MARK: - Path trace kernel

kernel void pathTraceKernel(
    texture2d<float, access::read_write> accum [[texture(0)]],
    texture2d_array<float> texAlbedo [[texture(1)]],
    texture2d_array<float> texNormal [[texture(2)]],
    texture2d_array<float> texData [[texture(3)]],
    texture2d<float> weatherField [[texture(4)]],
    constant Uniforms& u [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device const float* normals [[buffer(2)]],
    device const float* uvs [[buffer(3)]],
    device const float* colors [[buffer(4)]],
    device const uint* indices [[buffer(5)]],
    device const uint* triMat [[buffer(6)]],
    device const Material* materials [[buffer(7)]],
    device const uint* emissiveTris [[buffer(8)]],
    device const float* emissiveCDF [[buffer(9)]],
    acceleration_structure<> accel [[buffer(10)]],
    uint2 pixel [[thread_position_in_grid]])
{
    uint width = accum.get_width();
    uint height = accum.get_height();
    if (pixel.x >= width || pixel.y >= height) return;

    RNG rng;
    uint seedBase = hash32((pixel.y * width + pixel.x) ^ hash32(u.frameSeed * 2654435761u));
    rng.s0 = seedBase | 1u;
    rng.s1 = hash32(seedBase ^ 0x9e3779b9u) | 3u;

    constexpr sampler texSampler(address::repeat, filter::linear);
    constexpr sampler fieldSampler(address::clamp_to_edge, filter::linear);

    // Primary ray.
    float2 ndc = float2((float(pixel.x) + rng.nextFloat()) / float(width),
                        (float(pixel.y) + rng.nextFloat()) / float(height));
    float2 d = (ndc * 2.0f - 1.0f) * float2(u.tanFovAspect.y, 1.0f) * u.tanFovAspect.x;
    raytracing::ray ray;
    ray.origin = u.camOrigin;
    ray.direction = normalize(u.camForward + u.camRight * d.x + u.camUp * (-d.y));
    ray.min_distance = 0.0f;
    ray.max_distance = 1e30f;

    intersector<triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.accept_any_intersection(false);
    intersector<triangle_data> shadowIsect;
    shadowIsect.assume_geometry_type(geometry_type::triangle);
    shadowIsect.accept_any_intersection(true);

    float3 throughput = float3(1);
    float3 radiance = float3(0);
    bool prevSpecular = true; // primary ray: emissive hits count directly
    // Beer-Lambert medium state: while the path travels inside
    // glass, each segment attenuates by tint^distance.
    bool inMedium = false;
    float3 mediumLog = float3(0); // log(tint) of the enclosing glass
    uint refracts = 0;

    for (uint bounce = 0; bounce <= u.maxBounces; ++bounce) {
        auto hit = isect.intersect(ray, accel);
        if (hit.type == intersection_type::none) {
            radiance += throughput * envColor(ray.direction, u.envIntensity);
            break;
        }
        // Attenuate the segment just travelled inside glass.
        if (inMedium) {
            throughput *= exp(mediumLog * hit.distance);
        }

        uint tri = hit.primitive_id;
        uint i0 = indices[3 * tri], i1 = indices[3 * tri + 1], i2 = indices[3 * tri + 2];
        float3 p0 = float3(positions[3 * i0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
        float3 p1 = float3(positions[3 * i1], positions[3 * i1 + 1], positions[3 * i1 + 2]);
        float3 p2 = float3(positions[3 * i2], positions[3 * i2 + 1], positions[3 * i2 + 2]);
        float bc1 = hit.triangle_barycentric_coord.x;
        float bc2 = hit.triangle_barycentric_coord.y;
        float bc0 = 1.0f - bc1 - bc2;
        float3 hitPos = p0 * bc0 + p1 * bc1 + p2 * bc2;
        float3 nInterp = normalize(
            float3(normals[3 * i0], normals[3 * i0 + 1], normals[3 * i0 + 2]) * bc0 +
            float3(normals[3 * i1], normals[3 * i1 + 1], normals[3 * i1 + 2]) * bc1 +
            float3(normals[3 * i2], normals[3 * i2 + 1], normals[3 * i2 + 2]) * bc2);
        float3 vColor =
            float3(colors[3 * i0], colors[3 * i0 + 1], colors[3 * i0 + 2]) * bc0 +
            float3(colors[3 * i1], colors[3 * i1 + 1], colors[3 * i1 + 2]) * bc1 +
            float3(colors[3 * i2], colors[3 * i2 + 1], colors[3 * i2 + 2]) * bc2;
        float2 vUV =
            float2(uvs[2 * i0], uvs[2 * i0 + 1]) * bc0 +
            float2(uvs[2 * i1], uvs[2 * i1 + 1]) * bc1 +
            float2(uvs[2 * i2], uvs[2 * i2 + 1]) * bc2;

        Material mat = materials[triMat[tri]];
        bool front = hit.triangle_front_facing;
        float3 ns = front ? nInterp : -nInterp; // shading normal, against the ray
        float3 geoN = normalize(cross(p1 - p0, p2 - p0));
        if (dot(geoN, ns) < 0) geoN = -geoN;

        // ---- Emissive surface: it's a light. Count it when the path
        // arrived specularly (or from the camera); diffuse paths were
        // already lit by next-event estimation below.
        if (mat.flags & FLAG_EMISSIVE) {
            if (prevSpecular) {
                radiance += throughput * mat.emissive.rgb * mat.emissive.a;
            }
            break;
        }

        // ---- Texture fetch
        float3 albedo = mat.baseColor.rgb * vColor;
        float texRough = 0.9f;
        float3 shadingN = ns;
        if (mat.texIndex > 0) {
            uint slice = mat.texIndex - 1;
            float scale = mat.tint.a;
            if (mat.flags & FLAG_TRIPLANAR) {
                float3 wp = hitPos * scale;
                float3 w = pow(abs(nInterp), float3(4));
                w /= max(w.x + w.y + w.z, 1e-5f);
                float3 a = texAlbedo.sample(texSampler, wp.zy, slice, level(0.0f)).rgb * w.x
                         + texAlbedo.sample(texSampler, wp.xz, slice, level(0.0f)).rgb * w.y
                         + texAlbedo.sample(texSampler, wp.xy, slice, level(0.0f)).rgb * w.z;
                float rgh = texData.sample(texSampler, wp.zy, slice, level(0.0f)).r * w.x
                          + texData.sample(texSampler, wp.xz, slice, level(0.0f)).r * w.y
                          + texData.sample(texSampler, wp.xy, slice, level(0.0f)).r * w.z;
                albedo *= a;
                texRough = rgh;
                // Triplanar normal mapping: sample the tangent-space
                // normal on each axis plane, swizzle into world
                // space (sign from the geometric normal), blend with
                // the same weights — the texture's relief now rakes
                // under light on imported models too.
                float3 tnx = texNormal.sample(texSampler, wp.zy, slice, level(0.0f)).rgb * 2.0f - 1.0f;
                float3 tny = texNormal.sample(texSampler, wp.xz, slice, level(0.0f)).rgb * 2.0f - 1.0f;
                float3 tnz = texNormal.sample(texSampler, wp.xy, slice, level(0.0f)).rgb * 2.0f - 1.0f;
                float3 wnx = float3(tnx.z * (nInterp.x >= 0 ? 1.0f : -1.0f), tnx.y, tnx.x);
                float3 wny = float3(tny.x, tny.z * (nInterp.y >= 0 ? 1.0f : -1.0f), tny.y);
                float3 wnz = float3(tnz.x, tnz.y, tnz.z * (nInterp.z >= 0 ? 1.0f : -1.0f));
                float3 blended = normalize(wnx * w.x + wny * w.y + wnz * w.z);
                if (dot(blended, geoN) > 0.05f) shadingN = blended;
            } else {
                // FIT materials sample the texture once across the
                // face (vUV as authored); others tile by scale.
                float2 tuv = (mat.flags & FLAG_FIT) ? vUV : vUV * scale;
                // One-step parallax from the height channel, then
                // tangent-space normal mapping — light rakes across
                // the surface relief.
                float3 e1 = p1 - p0, e2 = p2 - p0;
                float2 duv1 = float2(uvs[2 * i1] - uvs[2 * i0], uvs[2 * i1 + 1] - uvs[2 * i0 + 1]);
                float2 duv2 = float2(uvs[2 * i2] - uvs[2 * i0], uvs[2 * i2 + 1] - uvs[2 * i0 + 1]);
                float det = duv1.x * duv2.y - duv2.x * duv1.y;
                float3 T = abs(det) > 1e-10f
                    ? normalize((e1 * duv2.y - e2 * duv1.y) / det)
                    : normalize(abs(ns.y) < 0.99f ? cross(ns, float3(0, 1, 0)) : cross(ns, float3(1, 0, 0)));
                float3 B = cross(ns, T);
                float hgt = texData.sample(texSampler, tuv, slice, level(0.0f)).g;
                float3 viewDir = -ray.direction;
                float3 vt = float3(dot(viewDir, T), dot(viewDir, B), dot(viewDir, ns));
                float2 puv = tuv + (vt.xy / max(vt.z, 0.25f)) * (hgt - 0.5f) * 0.05f;
                albedo *= texAlbedo.sample(texSampler, puv, slice, level(0.0f)).rgb;
                texRough = texData.sample(texSampler, puv, slice, level(0.0f)).r;
                float3 nm = texNormal.sample(texSampler, puv, slice, level(0.0f)).rgb * 2.0f - 1.0f;
                shadingN = normalize(T * nm.x + B * nm.y + ns * nm.z);
                if (dot(shadingN, geoN) < 0) shadingN = ns;
            }
        }

        // ---- Weather + gloss (v3). Texture roughness finally
        // shades: gloss = 1 - roughness for textured surfaces.
        // Wetness darkens up-facing surfaces and raises their
        // gloss; snow whitens them, kills the gloss, and pulls
        // the shading normal toward the geometric up-face.
        float upFace = smoothstep(0.5f, 0.85f, geoN.y);
        float wet = u.wetLevel * upFace;
        float snowGlobal = u.snowLevel * upFace;
        float gloss = (mat.texIndex > 0) ? clamp(1.0f - texRough, 0.0f, 1.0f) : 0.0f;
        // Shaped weather field: defaults are exactly the
        // v3 global behaviour (wetF = wet, snowCover = snowGlobal,
        // no puddle, drift 0.5, depth = snowGlobal). The shaped
        // field is only evaluated on the first two bounces —
        // deeper bounces keep the global levels, since the spatial
        // detail is invisible there. Since v3.3 the field itself
        // comes from the baked weatherField texture (one fetch)
        // instead of per-hit procedural noise. When both levels
        // are ~0 nothing below changes anything, so the dry
        // settled image is unchanged from v3.
        float wetF = wet;
        float snowCover = snowGlobal;
        float puddle = 0.0f;
        float drift = 0.5f;
        float depth = snowGlobal;
        // Baked field sample + window fade (v3.3). Neutral
        // values (potential 0, modulators at their mid) make
        // the un-faded case identical to no field at all.
        float4 wfield = float4(0.0f, 0.5f, 0.5f, 0.489f);
        float fieldFade = 0.0f;
        if ((u.wetLevel > 0.001f || u.snowLevel > 0.001f) && bounce <= 1u) {
            // ONE bilinear fetch replaces the ~75 per-hit hash
            // evaluations of the procedural v3.1/v3.2 block. The
            // field window is 64 m across, centred on the
            // origin; outside it the sample fades to neutral so
            // distant geometry keeps the plain global levels.
            float2 fuv = hitPos.xz * (1.0f / 64.0f) + 0.5f;
            fieldFade = (1.0f - smoothstep(29.0f, 32.0f, abs(hitPos.x)))
                      * (1.0f - smoothstep(29.0f, 32.0f, abs(hitPos.z)));
            float4 fs = weatherField.sample(fieldSampler, fuv);
            wfield = mix(float4(0.0f, 0.5f, 0.5f, 0.489f), fs, fieldFade);
            // Wetness spreads and pools unevenly.
            wetF = clamp(wet * (0.7f + 0.6f * wfield.g), 0.0f, 1.0f);
            // Geometry response (v3.2): water runs downhill.
            // elevBias favours low world positions (dips, holes,
            // slope feet) and penalises raised ones; slopeShed
            // drains slanted faces so water does not stand on a
            // slant. On a dead-flat floor both are constant
            // (elevBias +0.10 at floor level, shed 1), so the
            // pattern layer below rules alone there.
            float elevBias = clamp(0.10f - hitPos.y * 0.55f, -0.22f, 0.38f);
            float slopeShed = clamp((geoN.y - 0.70f) / 0.30f, 0.0f, 1.0f);
            // Trickle runnels: on slanted ground the wet
            // darkening arrives as streaks elongated along the
            // fall line (the shading normal's xz lean points
            // downhill for an up-facing slope). One value-noise
            // evaluation, slanted faces only — kept procedural
            // because the streak frame is surface-dependent.
            float slopeAmt = clamp((0.98f - geoN.y) / 0.28f, 0.0f, 1.0f);
            if (slopeAmt > 0.01f && length(geoN.xz) > 1e-4f) {
                float2 down = geoN.xz / length(geoN.xz);
                float2 along = float2(dot(hitPos.xz, down),
                                      dot(hitPos.xz, float2(-down.y, down.x)));
                float streak = wnoise2(float2(along.y * 5.0f, along.x * 0.7f));
                wetF = clamp(wetF + slopeAmt
                             * smoothstep(0.52f, 0.85f, streak)
                             * u.wetLevel * 0.55f, 0.0f, 1.0f);
            }
            // Puddles (v3.3): a threshold sweep of the baked
            // METABALL potential. The threshold falls as
            // wetness rises, so pools appear at the most
            // eligible spots first, then grow and merge —
            // coverage stays a deterministic function of
            // wetness on ANY surface (the flat guarantee).
            // elevBias shifts the threshold so dips, holes and
            // slope feet fill earlier and deeper; slopeShed
            // keeps standing water off slants; the broad
            // pooling field modulates. Outlines are level sets
            // of a smooth summed field: rounded, irregular,
            // feathered — square corners are impossible.
            float wetN = clamp(u.wetLevel, 0.0f, 1.0f);
            float thr = mix(1.15f, 0.08f, pow(wetN, 1.15f)) - elevBias * 0.45f;
            float mask = smoothstep(thr - 0.14f, thr + 0.14f, wfield.r);
            float broad = 0.78f + 0.45f * wfield.a;
            // fieldFade multiplies the mask as well as the
            // sample: at the lowest (full-wetness) threshold the
            // neutral potential 0 would otherwise read as a
            // faint uniform puddle outside the field window.
            puddle = clamp(mask * broad, 0.0f, 1.0f) * slopeShed * upFace
                   * fieldFade * smoothstep(0.03f, 0.15f, u.wetLevel);
            // Snow piles in drifts: depth varies with the
            // baked drift field, cover turns on where the
            // drifted depth is enough.
            drift = wfield.b;
            depth = u.snowLevel * mix(0.3f, 1.7f, drift);
            snowCover = smoothstep(0.45f, 0.75f, depth) * upFace;
        }
        // 1. Wet: darkened, glossier.
        if (wetF > 0.001f) {
            albedo *= (1.0f - 0.45f * wetF);
            gloss = max(gloss, wetF * 0.85f);
        }
        // 2. Puddle: mirror-flat standing water over the wet base.
        // The shading normal is the flat geometric normal so the
        // reflection converges cleanly and stays put; the
        // rain-ripple shimmer is applied at display time by the
        // tonemap pass (v3.3), which is what lets rain trace —
        // and settle — exactly like snow.
        if (puddle > 0.001f) {
            albedo *= (1.0f - 0.5f * puddle);
            gloss = max(gloss, puddle);
            shadingN = normalize(mix(shadingN, geoN, puddle));
        }
        // 3. Snow: whitened, gloss killed, normal mounded by the
        // drift gradient (or softened toward the face, as v3).
        if (snowCover > 0.001f) {
            snowCover *= (1.0f - puddle); // no snow inside puddles
        }
        if (snowCover > 0.001f) {
            albedo = mix(albedo, float3(0.88f, 0.91f, 0.97f), snowCover * 0.92f);
            albedo *= 1.0f - 0.18f * snowCover * (1.0f - drift);
            gloss *= (1.0f - snowCover);
            bool relief = false;
            if (bounce <= 1u && geoN.y > 0.7f) {
                // Drift gradient from the baked field (v3.3):
                // two extra fetches replace two fbm evaluations.
                // Offset samples are faded toward neutral BEFORE
                // the subtraction so the baseline matches the
                // faded centre sample at the window fringe.
                float gbx = mix(0.5f, weatherField.sample(fieldSampler,
                        (hitPos.xz + float2(0.35f, 0.0f)) * (1.0f / 64.0f) + 0.5f).b,
                        fieldFade);
                float gbz = mix(0.5f, weatherField.sample(fieldSampler,
                        (hitPos.xz + float2(0.0f, 0.35f)) * (1.0f / 64.0f) + 0.5f).b,
                        fieldFade);
                float gx = gbx - drift;
                float gz = gbz - drift;
                float3 mound = normalize(float3(-gx * depth * 1.4f, 1.0f,
                                                -gz * depth * 1.4f));
                shadingN = normalize(mix(shadingN, mound, snowCover * 0.6f));
                relief = true;
            }
            if (!relief) {
                shadingN = normalize(mix(shadingN, geoN, snowCover * 0.6f));
            }
        }

        // ---- Glass: reflect / refract with Fresnel. Past the
        // refraction budget the surface falls through to diffuse
        // shading (tinted), which bounds transmission cost in
        // glass-heavy views.
        bool glassPath = (mat.flags & FLAG_GLASS) && refracts < u.glassRefractBudget
                         && rng.nextFloat() < mat.glassParams.x;
        if (glassPath) {
            float ior = mat.glassParams.z;
            float eta = front ? (1.0f / ior) : ior;
            float f0 = (ior - 1.0f) / (ior + 1.0f);
            f0 *= f0;
            float cosT = clamp(dot(-ray.direction, ns), 0.0f, 1.0f);
            float F = fresnelSchlick(cosT, f0);
            float3 newDir;
            if (rng.nextFloat() < F) {
                newDir = reflect(ray.direction, ns);
                newDir = jitterDirection(newDir, mat.glassParams.y * 0.45f, rng);
                if (dot(newDir, ns) < 0) newDir = reflect(ray.direction, ns);
            } else {
                newDir = refract(ray.direction, ns, eta);
                if (dot(newDir, newDir) < 0.001f) { // total internal reflection
                    newDir = reflect(ray.direction, ns);
                } else {
                    newDir = jitterDirection(normalize(newDir), mat.glassParams.y * 0.45f, rng);
                    // Crossing the boundary toggles the medium;
                    // the travelled distance is charged at the
                    // NEXT hit via Beer-Lambert (see loop top).
                    refracts += 1;
                    if (front) {
                        inMedium = true;
                        mediumLog = log(max(mat.tint.rgb, float3(1e-4f)));
                    } else {
                        inMedium = false;
                    }
                }
            }
            float3 off = dot(newDir, geoN) >= 0 ? geoN : -geoN;
            ray.origin = hitPos + off * 0.0015f;
            ray.direction = newDir;
            ray.min_distance = 0.0f;
            prevSpecular = true;
            continue;
        }
        // A glass surface that took the diffuse fall-through (budget
        // spent or transmission roll failed) shades with its tint.
        if (mat.flags & FLAG_GLASS) {
            albedo *= mat.tint.rgb;
        }

        // ---- Diffuse: next-event estimation toward emissive tris.
        if (u.emissiveCount > 0 && u.totalEmissiveArea > 0.0f) {
            float pickU = rng.nextFloat() * u.totalEmissiveArea;
            uint lo = 0, hi = u.emissiveCount;
            while (lo < hi) { // binary search the area CDF
                uint mid = (lo + hi) / 2;
                if (emissiveCDF[mid] <= pickU) lo = mid + 1; else hi = mid;
            }
            uint etri = emissiveTris[min(lo, u.emissiveCount - 1)];
            uint e0 = indices[3 * etri], e1 = indices[3 * etri + 1], e2 = indices[3 * etri + 2];
            float3 q0 = float3(positions[3 * e0], positions[3 * e0 + 1], positions[3 * e0 + 2]);
            float3 q1 = float3(positions[3 * e1], positions[3 * e1 + 1], positions[3 * e1 + 2]);
            float3 q2 = float3(positions[3 * e2], positions[3 * e2 + 1], positions[3 * e2 + 2]);
            float r1 = rng.nextFloat(), r2 = rng.nextFloat();
            float sq = sqrt(r1);
            float3 lp = q0 * (1 - sq) + q1 * (sq * (1 - r2)) + q2 * (sq * r2);
            float3 toL = lp - hitPos;
            float d2 = dot(toL, toL);
            float dist = sqrt(d2);
            float3 wi = toL / dist;
            float cosS = dot(shadingN, wi);
            float3 eN = normalize(cross(q1 - q0, q2 - q0));
            float cosE = abs(dot(eN, -wi));
            if (cosS > 0.0f && cosE > 0.0f && dist > 0.01f) {
                raytracing::ray sray;
                sray.origin = hitPos + geoN * 0.0015f;
                sray.direction = wi;
                sray.min_distance = 0.0f;
                sray.max_distance = dist - 0.003f;
                auto shit = shadowIsect.intersect(sray, accel);
                if (shit.type == intersection_type::none) {
                    Material em = materials[triMat[etri]];
                    float3 Le = em.emissive.rgb * em.emissive.a;
                    radiance += throughput * (albedo / M_PI_F) * Le
                              * cosS * cosE * u.totalEmissiveArea / d2;
                }
            }
        }

        // ---- Diffuse bounce, with a glossy sheen roll for wet
        // or low-roughness textured surfaces: with probability
        // pSpec (Fresnel- and gloss-driven) the bounce goes
        // specular instead — a jittered reflection, tinted only
        // partway toward the albedo, and allowed to count
        // emissive hits directly on the next segment.
        float3 newDir = cosineDirection(shadingN, rng);
        float3 bounceTint = albedo;
        bool bounceSpecular = false;
        if (gloss > 0.02f) {
            float cosI = clamp(dot(-ray.direction, shadingN), 0.0f, 1.0f);
            float fres = fresnelSchlick(cosI, 0.04f);
            float pSpec = clamp(gloss * (0.25f + 0.75f * fres) * 1.6f, 0.0f, 0.85f);
            if (rng.nextFloat() < pSpec) {
                float3 specDir = jitterDirection(reflect(ray.direction, shadingN),
                                                 (1.0f - gloss) * 0.35f, rng);
                if (dot(specDir, shadingN) > 0.0f) {
                    newDir = specDir;
                    bounceTint = mix(float3(1.0f), albedo, 0.45f) * (0.35f + 0.65f * gloss);
                    bounceSpecular = true;
                }
            }
        }
        throughput *= bounceTint;
        ray.origin = hitPos + geoN * 0.0015f;
        ray.direction = newDir;
        ray.min_distance = 0.0f;
        prevSpecular = bounceSpecular;

        // Russian roulette once the path is deep.
        if (bounce >= 3) {
            float q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (rng.nextFloat() > q) break;
            throughput /= q;
        }
    }

    // Firefly clamp (v3): cap this sample against the pixel's
    // own running average before it lands, so a single hot path
    // can't blow out a pixel that has otherwise settled. Young
    // pixels (no trustworthy average yet) get a generous fixed
    // cap; a settled pixel's own average is high, so legitimate
    // bright content is not biased down.
    float4 prevA = accum.read(pixel);
    float capLum;
    if (prevA.a > 0.5f) {
        float3 avg = prevA.rgb / prevA.a;
        capLum = max(max(avg.x, max(avg.y, avg.z)) * 6.0f, 1.5f);
    } else {
        capLum = 30.0f;
    }
    float sampleLum = max(max(radiance.x, max(radiance.y, radiance.z)), 0.0f);
    if (sampleLum > capLum && sampleLum > 0.0f) {
        radiance *= capLum / sampleLum;
    }

    // Accumulate: alpha tracks the effective sample count. At the
    // history cap, decay the stored sum so it becomes a moving
    // average over the last `historyCap` samples (Play mode and
    // in-motion frames stay smooth instead of freezing or noising).
    float a = prevA.a;
    float3 rgb = prevA.rgb;
    if (a >= u.historyCap) {
        float s = (u.historyCap - 1.0f) / u.historyCap;
        rgb *= s;
        a = u.historyCap - 1.0f;
    }
    accum.write(float4(rgb + radiance, a + 1.0f), pixel);
}

// MARK: - Guide kernel (primary-hit world positions, for reprojection)
//
// One jitterless primary ray per pixel. Writes (worldPos, reusable)
// where reusable = 1 for diffuse and emissive first hits (their
// radiance is view-independent) and 0 for glass, environment, and
// misses (view-dependent: always traced fresh). Also writes the
// guideNormal texture (v3): the interpolated vertex normal at the
// same hit (shading-agnostic — no texture normal mapping), with
// the same reusable flag in w; the spatial denoise pass uses it
// to stop blur crossing surface orientations.

kernel void guideKernel(
    texture2d<float, access::write> guide [[texture(0)]],
    texture2d<float, access::write> guideNormal [[texture(1)]],
    constant Uniforms& u [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device const float* normals [[buffer(2)]],
    device const uint* indices [[buffer(5)]],
    device const uint* triMat [[buffer(6)]],
    device const Material* materials [[buffer(7)]],
    acceleration_structure<> accel [[buffer(10)]],
    uint2 pixel [[thread_position_in_grid]])
{
    uint width = guide.get_width();
    uint height = guide.get_height();
    if (pixel.x >= width || pixel.y >= height) return;

    float2 ndc = float2((float(pixel.x) + 0.5f) / float(width),
                        (float(pixel.y) + 0.5f) / float(height));
    float2 d = (ndc * 2.0f - 1.0f) * float2(u.tanFovAspect.y, 1.0f) * u.tanFovAspect.x;
    raytracing::ray ray;
    ray.origin = u.camOrigin;
    ray.direction = normalize(u.camForward + u.camRight * d.x + u.camUp * (-d.y));
    ray.min_distance = 0.0f;
    ray.max_distance = 1e30f;

    intersector<triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.accept_any_intersection(false);
    auto hit = isect.intersect(ray, accel);
    if (hit.type == intersection_type::none) {
        guide.write(float4(0, 0, 0, 0), pixel);
        guideNormal.write(float4(0, 0, 0, 0), pixel);
        return;
    }
    uint tri = hit.primitive_id;
    uint i0 = indices[3 * tri], i1 = indices[3 * tri + 1], i2 = indices[3 * tri + 2];
    float3 p0 = float3(positions[3 * i0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
    float3 p1 = float3(positions[3 * i1], positions[3 * i1 + 1], positions[3 * i1 + 2]);
    float3 p2 = float3(positions[3 * i2], positions[3 * i2 + 1], positions[3 * i2 + 2]);
    float bc1 = hit.triangle_barycentric_coord.x;
    float bc2 = hit.triangle_barycentric_coord.y;
    float bc0 = 1.0f - bc1 - bc2;
    float3 hitPos = p0 * bc0 + p1 * bc1 + p2 * bc2;
    float3 nInterp = normalize(
        float3(normals[3 * i0], normals[3 * i0 + 1], normals[3 * i0 + 2]) * bc0 +
        float3(normals[3 * i1], normals[3 * i1 + 1], normals[3 * i1 + 2]) * bc1 +
        float3(normals[3 * i2], normals[3 * i2 + 1], normals[3 * i2 + 2]) * bc2);
    Material mat = materials[triMat[tri]];
    float reusable = (mat.flags & FLAG_GLASS) ? 0.0f : 1.0f;
    guide.write(float4(hitPos, reusable), pixel);
    guideNormal.write(float4(nInterp, reusable), pixel);
}

// MARK: - Spatial denoise kernel (v3: interactive anti-sparkle)
//
// A 3x3 cross-bilateral filter over the accumulation, guided by
// the primary-hit position, the interpolated surface normal, and
// luminance. Neighbours blend only within the same reusability
// class (diffuse/emissive vs glass/environment), so the filter
// never smears across glass or silhouette edges. The blend fades
// out as a pixel's sample count climbs toward ~96, handing over
// to the raw (unbiased) image; the renderer only runs this pass
// for interactive / Play display and always shows the settled
// frame raw. Output keeps the accumulated-sum convention
// (rgb = mean * alpha) so the tonemap is unchanged.

struct DenoiseUniforms {
    float3 camOrigin;
    float strength;
    float pad0;
    float pad1;
};

static float denoiseLuminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

kernel void denoiseKernel(
    texture2d<float, access::read> accum [[texture(0)]],
    texture2d<float, access::read> guide [[texture(1)]],
    texture2d<float, access::read> guideNormal [[texture(2)]],
    texture2d<float, access::write> output [[texture(3)]],
    constant DenoiseUniforms& u [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]])
{
    uint width = accum.get_width();
    uint height = accum.get_height();
    if (pixel.x >= width || pixel.y >= height) return;

    float4 c = accum.read(pixel);
    if (c.a < 1.0f) {
        output.write(c, pixel);
        return;
    }
    float3 mean = c.rgb / max(c.a, 1.0f);
    float4 gc = guide.read(pixel);
    float3 nc = guideNormal.read(pixel).xyz;
    float lumC = denoiseLuminance(mean);
    // Position tolerance scales with distance from the camera.
    float sp = max(0.04f, 0.012f * distance(u.camOrigin, gc.xyz));

    float3 sum = float3(0);
    float wSum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 np = int2(pixel) + int2(dx, dy);
            if (np.x < 0 || np.y < 0 || np.x >= int(width) || np.y >= int(height)) continue;
            float4 g = guide.read(uint2(np));
            if (g.w != gc.w) continue; // same reusability class only
            float4 nAcc = accum.read(uint2(np));
            if (nAcc.a < 1.0f) continue;
            float3 meanN = nAcc.rgb / max(nAcc.a, 1.0f);
            float3 nn = guideNormal.read(uint2(np)).xyz;
            float3 dp = g.xyz - gc.xyz;
            float wp = exp(-dot(dp, dp) / (2.0f * sp * sp));
            float wn = pow(clamp(dot(nc, nn), 0.0f, 1.0f), 6.0f);
            float wl = exp(-abs(denoiseLuminance(meanN) - lumC) / (0.12f + 0.25f * lumC));
            float ws = (dx == 0 && dy == 0) ? (4.0f / 16.0f)
                     : ((dx == 0 || dy == 0) ? (2.0f / 16.0f) : (1.0f / 16.0f));
            float w = wp * wn * wl * ws;
            sum += meanN * w;
            wSum += w;
        }
    }
    float3 filtered = wSum > 1e-6f ? sum / wSum : mean;
    float fade = clamp(1.0f - c.a / 96.0f, 0.0f, 1.0f);
    float3 blended = mix(mean, filtered, u.strength * fade);
    output.write(float4(blended * c.a, c.a), pixel);
}

// MARK: - Reprojection kernel (seed current accumulation from history)
//
// For each pixel, project its current world position into the
// previous frame's camera. If the previous frame saw (nearly) the
// same world point through that pixel and it was a reusable
// surface, carry its accumulated radiance over with weight
// min(prevSamples, reuseCap) in the alpha channel. Anything else
// starts fresh at zero.

struct ReprojUniforms {
    float3 prevOrigin;
    float3 prevRight;
    float3 prevUp;
    float3 prevForward;
    float2 tanFovAspect; // x: tan(half fov), y: aspect — of the PREVIOUS frame
    float reuseCap;
    float pad;
};

kernel void reprojectKernel(
    texture2d<float, access::read> prevAccum [[texture(0)]],
    texture2d<float, access::read> prevGuide [[texture(1)]],
    texture2d<float, access::read> curGuide [[texture(2)]],
    texture2d<float, access::write> curAccum [[texture(3)]],
    constant ReprojUniforms& u [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]])
{
    uint width = curGuide.get_width();
    uint height = curGuide.get_height();
    if (pixel.x >= width || pixel.y >= height) return;

    float4 g = curGuide.read(pixel);
    if (g.w < 0.5f || u.reuseCap <= 0.0f) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        return;
    }
    float3 rel = g.xyz - u.prevOrigin;
    float z = dot(rel, u.prevForward);
    if (z <= 1e-4f) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        return;
    }
    float x = dot(rel, u.prevRight);
    float y = dot(rel, u.prevUp);
    // Invert the primary-ray construction of the trace kernel:
    // dir ∝ fwd + right*(ndc.x*tan*aspect) + up*(-(ndc.y*tan)).
    float2 ndc = float2((x / z) / (u.tanFovAspect.x * u.tanFovAspect.y),
                        -(y / z) / u.tanFovAspect.x);
    float2 uv = ndc * 0.5f + 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        return;
    }
    uint pw = prevGuide.get_width();
    uint ph = prevGuide.get_height();
    uint2 pp = uint2(min(uint(uv.x * float(pw)), pw - 1),
                     min(uint(uv.y * float(ph)), ph - 1));
    float4 pg = prevGuide.read(pp);
    if (pg.w < 0.5f) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        return;
    }
    float eps = max(0.02f, 0.002f * length(rel));
    if (distance(pg.xyz, g.xyz) > eps) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        return;
    }
    float4 h = prevAccum.read(pp);
    float wgt = min(h.a, u.reuseCap);
    float3 rgb = h.a > 0.001f ? h.rgb * (wgt / h.a) : float3(0);
    curAccum.write(float4(rgb, wgt), pixel);
}

// MARK: - Tonemap / display

struct TonemapParams {
    uint samples;
    float exposure;
    float shimmerTime;  // v3.3: weather clock (s) for the ripple shimmer
    float shimmerRain;  // v3.3: live rain intensity (0 unless raining)
    float wetLevel;     // v3.3: ground wetness, for the puddle mask
    float pad;
};

struct VSOut {
    float4 pos [[position]];
    float2 uv;
};

vertex VSOut tonemapVertex(uint vid [[vertex_id]]) {
    VSOut out;
    float2 p = float2((vid << 1) & 2, vid & 2);
    out.pos = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    out.uv = float2(p.x, 1.0f - p.y);
    return out;
}

static float3 aces(float3 x) {
    return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}

fragment float4 tonemapFragment(VSOut in [[stage_in]],
                                texture2d<float> accum [[texture(0)]],
                                texture2d<float> guide [[texture(1)]],
                                texture2d<float> guideNormal [[texture(2)]],
                                texture2d<float> weatherField [[texture(3)]],
                                constant TonemapParams& p [[buffer(0)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float4 a = accum.sample(s, in.uv);
    // Alpha carries the per-pixel effective sample count (v2);
    // fall back to the uniform count where alpha is unset.
    float n = a.a > 0.5f ? a.a : max(1.0f, float(p.samples));
    float3 c = a.rgb / n;
    // Rain-ripple shimmer (v3.3): while it rains, puddle pixels
    // get a small time-varying brightness modulation in linear
    // space — the display-rate echo of the normal tilt the
    // path kernel used to re-trace every frame (v3.2's lag).
    // Puddles are located exactly as the tracer shaded them:
    // primary-hit world position from the guide, the baked
    // field mask at that spot, up-facing hits only. The
    // converged mirror underneath stays untouched; stills and
    // exports pass shimmerRain = 0 and skip this entirely.
    if (p.shimmerRain > 0.01f) {
        float4 g = guide.sample(s, in.uv);
        float4 gn = guideNormal.sample(s, in.uv);
        if (g.a > 0.5f && gn.y > 0.5f) {
            float3 wp = g.xyz;
            float2 fuv = wp.xz * (1.0f / 64.0f) + 0.5f;
            float fade = (1.0f - smoothstep(29.0f, 32.0f, abs(wp.x)))
                       * (1.0f - smoothstep(29.0f, 32.0f, abs(wp.z)));
            float4 fs = weatherField.sample(s, fuv);
            float pot = fs.r * fade;
            float wetN = clamp(p.wetLevel, 0.0f, 1.0f);
            float elevBias = clamp(0.10f - wp.y * 0.55f, -0.22f, 0.38f);
            float thr = mix(1.15f, 0.08f, pow(wetN, 1.15f)) - elevBias * 0.45f;
            float mask = smoothstep(thr - 0.14f, thr + 0.14f, pot);
            float broad = 0.78f + 0.45f * mix(0.489f, fs.a, fade);
            float slopeShed = clamp((gn.y - 0.70f) / 0.30f, 0.0f, 1.0f);
            float upFace = smoothstep(0.5f, 0.85f, gn.y);
            float puddleM = clamp(mask * broad, 0.0f, 1.0f) * slopeShed * upFace
                          * fade * smoothstep(0.03f, 0.15f, p.wetLevel);
            if (puddleM > 0.01f) {
                // Two layers of expanding rings, as in the old
                // in-tracer version — but only their brightness
                // echo: rings alternately lift and dip the
                // reflected light a little as they pass.
                float rings = 0.0f;
                for (int layer = 0; layer < 2; layer++) {
                    float sc = (layer == 0) ? 6.0f : 10.5f;
                    float sp = (layer == 0) ? 0.85f : 1.25f;
                    float2 rp = wp.xz * sc;
                    int2 rc = int2(floor(rp));
                    float2 rf = fract(rp) - 0.5f;
                    uint rh = hash32((uint(rc.x) * 0x8da6b343u)
                                     ^ (uint(rc.y) * 0xd8163841u)
                                     ^ (uint(layer) * 0x27d4eb2du));
                    float ph = float(rh >> 8) * (1.0f / 16777216.0f);
                    float age = fract(p.shimmerTime * sp + ph);
                    float rr = length(rf);
                    float ring = sin(3.14159265f * age)
                               * exp(-rr * 3.5f) * (1.0f - age);
                    rings += ring * ((layer == 0) ? 0.65f : 0.40f)
                           * (ph > 0.5f ? 1.0f : -1.0f);
                }
                float m = 1.0f + 0.16f * rings * puddleM
                              * clamp(p.shimmerRain, 0.0f, 1.2f);
                c *= m;
            }
        }
    }
    c = aces(c * p.exposure);
    c = pow(c, float3(1.0f / 2.2f));
    return float4(c, 1.0f);
}

// MARK: - Precipitation overlay (v3: rain / snow, live draw only)
//
// Procedural particles in a camera-centred box, seeded from a
// per-particle buffer (v3.1: buffer(1), 2048 float4 entries of
// (cellX, cellY, cellZ, rand), each in [0,1] — placement and
// speed come from the seed, not a vertex-id hash). Each particle
// is a camera-aware quad (two triangles, 6 vertices): rain is a
// short streak aligned to its fall velocity; snow is a camera-
// facing soft sprite. Projection is manual (same basis maths as
// the primary ray) and the pass has no depth attachment, so
// particles simply blend over the tonemapped frame in draw order.

struct PrecipUniforms {
    float3 camOrigin;
    float3 camRight;
    float3 camUp;
    float3 camForward;
    float2 tanFovAspect; // x: tan(half vertical fov), y: aspect
    float time;
    uint mode;           // 1 = rain, 2 = snow
    float intensity;     // v3.2: live intensity 0..~1.2 (surge cycle)
    float2 windXZ;       // v3.2: wind velocity in xz (m/s), time-varying
    float precipPad;
};

struct PrecipOut {
    float4 pos [[position]];
    float2 uv;
    float fade;
    float kind; // 1 rain, 2 snow (mirrors PrecipUniforms.mode)
};

vertex PrecipOut precipVertex(uint vid [[vertex_id]],
                              constant PrecipUniforms& u [[buffer(0)]],
                              device const float4* seeds [[buffer(1)]]) {
    PrecipOut out;
    out.uv = float2(0);
    out.fade = 0.0f;
    out.kind = float(u.mode);
    out.pos = float4(3.0f, 3.0f, 0.5f, 1.0f); // clipped until proven visible

    uint inst = vid / 6u;
    uint corner = vid % 6u;
    // Quad corners: (-1,-1) (1,-1) (-1,1) (-1,1) (1,-1) (1,1).
    float cx = (corner == 1u || corner == 4u || corner == 5u) ? 1.0f : -1.0f;
    float cy = (corner == 2u || corner == 3u || corner == 5u) ? 1.0f : -1.0f;
    out.uv = float2(cx * 0.5f + 0.5f, cy * 0.5f + 0.5f);

    // Buffer-driven seed: xyz = base cell in [0,1]^3, w = the
    // particle's random (speed, sway phase, flake size).
    float4 seed = seeds[inst];
    float3 base = seed.xyz;

    bool rain = u.mode == 1u;
    float3 box = rain ? float3(36.0f, 22.0f, 36.0f) : float3(34.0f, 20.0f, 34.0f);
    // v3.2 dynamics: rain falls faster as the surge intensity
    // rises; the wind is the session's time-varying vector, so
    // streaks slant and snow drifts with the gusts.
    float speed = rain ? (10.0f + seed.w * 4.0f) * (0.85f + 0.30f * u.intensity)
                       : (0.9f + seed.w * 1.1f);
    float3 wind = float3(u.windXZ.x, 0.0f, u.windXZ.y);

    float3 p = base * box;
    p.y = fmod(p.y - u.time * speed, box.y);
    if (p.y < 0.0f) p.y += box.y;
    // Wind carry: the particle field drifts sideways with the
    // wind, wrapped inside the box. Rain falls too fast to be
    // carried far (0.25x); snow rides the wind (0.55x).
    float carry = rain ? 0.25f : 0.55f;
    p.x = fmod(p.x + u.time * wind.x * carry, box.x);
    if (p.x < 0.0f) p.x += box.x;
    p.z = fmod(p.z + u.time * wind.z * carry, box.z);
    if (p.z < 0.0f) p.z += box.z;
    if (!rain) {
        float phase = seed.w * 6.2831853f;
        p.x += sin(u.time * 0.8f + phase) * 0.35f;
    }
    // Box is centred on the camera in xz; y runs from 2 m below
    // the camera to box.y - 2 above it.
    float3 centre = float3(u.camOrigin.x + (p.x - box.x * 0.5f),
                           u.camOrigin.y - 2.0f + p.y,
                           u.camOrigin.z + (p.z - box.z * 0.5f));

    float3 centreRel = centre - u.camOrigin;
    float centreDist = length(centreRel);
    float centreZ = dot(centreRel, u.camForward);
    if (centreZ < 0.3f) return out; // behind / too close: degenerate

    float3 world;
    if (rain) {
        float3 axis = normalize(float3(wind.x, -speed, wind.z));
        float3 viewDir = centreDist > 1e-4f ? centreRel / centreDist : u.camForward;
        float3 sideRaw = cross(axis, viewDir);
        float sideLen = length(sideRaw);
        float3 side = sideLen > 1e-4f ? sideRaw / sideLen : u.camRight;
        // Streak: 0.016 m half-width, 0.6 m long (0.3 half-length).
        world = centre + side * (cx * 0.016f) + axis * (cy * 0.3f);
    } else {
        // Camera-facing disc, diameter 0.09–0.18 m by seed.
        float size = 0.09f + seed.w * 0.09f;
        world = centre + u.camRight * (cx * size * 0.5f) + u.camUp * (cy * size * 0.5f);
    }

    float3 rel = world - u.camOrigin;
    float z = dot(rel, u.camForward);
    float tanF = u.tanFovAspect.x;
    float2 ndc = float2(dot(rel, u.camRight) / (z * tanF * u.tanFovAspect.y),
                        dot(rel, u.camUp) / (z * tanF));
    out.pos = float4(ndc.x, ndc.y, 0.5f, 1.0f);

    float farDist = rain ? 30.0f : 28.0f;
    float fadeFar = 1.0f - smoothstep(farDist * 0.65f, farDist, centreDist);
    float fadeNear = smoothstep(0.0f, 0.8f, centreDist);
    // Intensity thins the field's alpha in a lull and thickens
    // it in a surge (the draw count scales alongside, Swift-side).
    out.fade = fadeFar * fadeNear
             * (0.35f + 0.65f * clamp(u.intensity, 0.0f, 1.2f));
    return out;
}

fragment float4 precipFragment(PrecipOut in [[stage_in]]) {
    if (in.kind < 1.5f) {
        // Rain: faint blue-grey streak, fading along its length.
        float lengthFade = sin(3.14159265f * clamp(in.uv.y, 0.0f, 1.0f));
        float alpha = 0.50f * lengthFade * in.fade;
        return float4(float3(0.72f, 0.80f, 0.95f), alpha);
    }
    // Snow: soft radial disc.
    float d = length(in.uv - float2(0.5f)) * 2.0f;
    float disc = 1.0f - smoothstep(0.35f, 1.0f, d);
    float alpha = 0.9f * disc * in.fade;
    return float4(float3(0.93f, 0.96f, 1.0f), alpha);
}
