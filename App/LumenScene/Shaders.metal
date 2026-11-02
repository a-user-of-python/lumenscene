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
//
// V4.15 additions (engine optimisation round 2 — every change
// is expectation-preserving: the settled image converges to
// the same values, reached with less work):
//   * Light: emissive triangles are selected for next-event
//     estimation by area x luminance (importance) instead of
//     area alone — same integral, far less variance when dim
//     panels and hot spots share a scene; deep-bounce NEE is
//     thinned by compensated Russian roulette.
//   * Reflections: the glossy specular roll gets a
//     roughness-aware continuation roulette (rough reflections
//     end early, survivors are compensated), and path Russian
//     roulette starts one bounce earlier.
//   * Glass: the reflect/refract decision is taken BEFORE any
//     texture fetch (a pure glass event never pays for PBR
//     sampling it cannot use), nested interfaces past the
//     second crossing face a throughput roulette, and the
//     Beer-Lambert step is skipped for untinted glass.
//   * Emissive hits resolve before attribute interpolation
//     (their radiance needs the material alone).
//   * Live path only: per-pixel sample moments feed a
//     variance gate — a pixel whose accumulated mean is
//     already tight stops consuming samples (stills and the
//     Final Render never set the flag and trace exactly as
//     before).

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
    float totalEmissiveArea;   // v4.15: the light-selection weight total W = sum(area_i * lum(Le_i)) — the field keeps its name/layout; the NEE estimator divides by the picked light's luminance (see pathTraceKernel)
    float envIntensity;
    float historyCap;          // max effective samples kept per pixel
    uint glassRefractBudget;   // max refraction events per path
    float wetLevel;            // v3: 0 dry .. 1 soaked (up-facing)
    float snowLevel;           // v3: 0 none .. 1 full cover (up-facing)
    float dynTime;             // v3.2: weather clock (retained for layout; v3.3: unused by the trace kernel)
    float rainFrac;            // v3.2: live rain intensity (retained for layout; v3.3: unused by the trace kernel)
    uint dynTriCount;          // v4: dynamic-layer triangle count (unified ids start at triCount)
    uint hidePlayerHead;       // v4.5: 1 while the camera is the Play first-person eye — the player figure's head/hair (FLAG_PLAYER_HEAD) are excluded from PRIMARY hits only (was padDyn; layout unchanged)
    float freezeFrac;          // v4.7: puddle freeze fraction 0...1 (tail field; mirrors Renderer.swift GPUUniforms)
    float waterReflScale;      // v4.8: water reflection resolution 1.0 / 0.5 / 0.25 (tail; mirrors GPUUniforms)
    uint frameCounter;         // v4.8: per-presented-frame counter for the rotating reflection pattern (tail; mirrors GPUUniforms)
    uint traceFlags;           // v4.15 (was reflPad; layout unchanged): bit0 = variance-gated adaptive sampling, bit1 = track per-pixel sample moments. Set ONLY by the live draw path; stills and the Final Render pass 0 and trace exactly as pre-v4.15.
};

struct Material {
    float4 baseColor;    // rgb; a unused
    float4 emissive;     // rgb colour; a = intensity
    float4 glassParams;  // x: transmission, y: roughness, z: ior, w: mean texture roughness (v4.1 F3, guide gloss only)
    float4 tint;         // rgb; a = textureScale
    uint texIndex;       // 0 none; 1..4095 main array slice+1 (slice = texIndex-1);
                         // >= 4096 road array (road slice = texIndex-4096-1) (v4.1)
    uint flags;          // bit0 emissive, bit1 glass, bit2 triplanar, bit3 fit
    uint2 pad;
};

constant uint FLAG_EMISSIVE = 1u;
constant uint FLAG_GLASS = 2u;
constant uint FLAG_TRIPLANAR = 4u;
constant uint FLAG_FIT = 8u;
constant uint FLAG_SHEET = 16u;   // v3.4: puddle water sheet (real geometry)
constant uint FLAG_SNOW = 32u;    // v4.4: snow sheet (real geometry, always snow material)
constant uint FLAG_PLAYER_HEAD = 64u; // v4.5: player figure head/hair — invisible to primary rays from the Play eye (u.hidePlayerHead), visible to every other ray

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
// Bake at renderer init (v4.3: re-run whenever the weather
// SEED changes): the level-independent weather field over a
// 64 m world window centred on the origin. R = puddle
// metaball potential, G = wet pooling fbm, B = snow drift
// fbm, A = broad pooling fbm — the exact fields the
// v3.1/v3.2 path kernel used to evaluate procedurally per
// hit per sample (~75 hash evaluations). Wet/snow LEVELS are
// thresholds applied analytically in the path kernel, so
// this texture never needs refreshing when build-up changes.
// v4.3: seed parameters for the bake, derived CPU-side by
// SplitMix32 from the project's weather seed (see
// weatherSeedDerived in Renderer.swift): offset = field
// translation in metres (each component in [-28, 28)), wind =
// the literal (cos, sin) of the seed's wind heading (45°
// steps — the same literal Float constants both sides), and
// seeded = 0 reproducing the legacy v4.2 field bit-exactly.
struct WeatherFieldParams {
    float2 offset;
    float2 wind;
    uint seeded;
    uint pad;
};


// MARK: - Path trace kernel

// v4.1: roads live in a SEPARATE 1024 texture2d_array set
// (trace bindings 5-7); materials encode them as
// texIndex >= 4096 (road slice = texIndex - 4096 - 1), while
// main-array materials use texIndex in 1..4095
// (slice = texIndex - 1). This one helper keeps every PBR
// fetch site in the trace kernel on the same branch.
static float4 samplePBR(texture2d_array<float> mainTex,
                        texture2d_array<float> roadTex,
                        bool isRoad, sampler smp,
                        float2 uv, uint slice, uint roadSlice) {
    return isRoad ? roadTex.sample(smp, uv, roadSlice, level(0.0f))
                  : mainTex.sample(smp, uv, slice, level(0.0f));
}

// MARK: - Shader snow accumulation (v4.12: one shared factor)
//
// Snow ON OBJECTS is a material layer, never geometry — the
// standard games gate: cover grows with the drifted depth and
// with how UP-FACING the surface is, so a vertical face
// (geoN.y = 0) accumulates EXACTLY zero by construction, flat
// tops the most, steep roofs little. Factored into one
// function so the trace kernel, the guide kernel and the
// v4.12 gate audit (snowGateAuditKernel) all evaluate the
// identical math.
static float upFaceGate(float ny) {
    return smoothstep(0.5f, 0.85f, ny);
}
// v4.12 (RDR2): caked-snow breakup. The gate above decides
// WHERE snow can sit; this multiplier decides how it LOOKS
// there — a patchy, noise-broken coating (broad lumps at
// ~0.4 m, grain at ~9 cm) instead of uniform white paint.
// It only ever MULTIPLIES the gated cover, so the gate's
// guarantee survives untouched: a vertical face's cover is
// 0 x anything = exactly 0.
// v4.12 (RDR2): two-scale world-anchored micro normal detail
// for snow surfaces — keeps troughs and tracks reading at
// glancing angles (where the 0.20 m lattice normals alone go
// flat) and gives the snow gloss its sparkle facets. The tilt
// is applied in world XZ and scaled by n.y, valid because
// snow lives on up-facing surfaces.
// v4.12 (RDR2): the glint stage — RDR2's signature sparkle.
// Sparse world-anchored micro-facets (~4 mm cells, ~0.35% of
// the surface) flash when their random facet normal reflects
// the sky toward the camera. Static in world space: stills
// keep their specks, camera motion makes them twinkle.

kernel void pathTraceKernel(
    texture2d<float, access::read_write> accum [[texture(0)]],
    texture2d_array<float> texAlbedo [[texture(1)]],
    texture2d_array<float> texNormal [[texture(2)]],
    texture2d_array<float> texData [[texture(3)]],
    texture2d<float> weatherField [[texture(4)]],
    texture2d_array<float> roadAlbedo [[texture(5)]],
    texture2d_array<float> roadNormal [[texture(6)]],
    texture2d_array<float> roadData [[texture(7)]],
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
    device const float* dynPositions [[buffer(11)]],
    device const float* dynNormals [[buffer(12)]],
    device const float* dynUVs [[buffer(13)]],
    device const float* dynColors [[buffer(14)]],
    device const uint* dynIndices [[buffer(15)]],
    device const uint* dynTriMat [[buffer(16)]],
    acceleration_structure<> dynAccel [[buffer(17)]],
    device float2* moments [[buffer(18)]],
    uint2 pixel [[thread_position_in_grid]])
{
    uint width = accum.get_width();
    uint height = accum.get_height();
    if (pixel.x >= width || pixel.y >= height) return;

    // v4.15: the previous accumulation is read ONCE, up front —
    // the variance gate below needs it before tracing, and the
    // firefly clamp + accumulation at the end reuse the same
    // registers (this thread is the only writer of this pixel).
    float4 prevA = accum.read(pixel);
    // Per-pixel sample moments (sum of sample luminance, sum of
    // its square), maintained alongside the accumulation in the
    // live path only (traceFlags bit1). The buffer is bound but
    // never touched when the flags are clear (stills / Final
    // Render bind a dummy).
    float2 momentsIn = float2(0);
    if (u.traceFlags & 2u) {
        momentsIn = moments[pixel.y * width + pixel.x];
        // Variance gate (bit0): a pixel whose accumulated mean
        // is already tight stops consuming samples. The
        // threshold is a standard error of 2% of the mean plus
        // a small absolute floor for near-black pixels; noisy
        // pixels (glass, reflections, caustic paths) carry
        // large variance and keep sampling on their own.
        if ((u.traceFlags & 1u) && prevA.a >= 32.0f) {
            float n = prevA.a;
            float meanL = momentsIn.x / n;
            float varL = momentsIn.y / n - meanL * meanL;
            float se = sqrt(max(varL, 0.0f) / n);
            if (se <= 0.02f * meanL + 0.0015f) return;
        }
    }

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
        // v4: merged static + dynamic intersection. Static tris
        // are ids 0..<triCount, dynamic tris are triCount..<
        // triCount+dynTriCount; the nearer hit of the two AS
        // traversals wins and selects the attribute set below.
        // v4.5: at bounce 0 with u.hidePlayerHead set (the Play
        // first-person eye), a winning hit on the player
        // figure's head or hair (FLAG_PLAYER_HEAD) is stepped
        // over and the merged query re-runs past it — the eye
        // sits INSIDE the head box, so without this its inner
        // wall is the whole screen. Bounces >= 1 and shadow
        // rays are NOT filtered: the head still renders in
        // reflections and mirror views and casts shadows, and
        // the rest of the body stays visible to primary rays
        // (look down and you see your torso, arms and legs).
        bool haveHit = false;
        float hitDist = 1e30f;
        uint primID = 0u;
        float2 hitBary = float2(0);
        bool hitFront = true;
        bool isDynHit = false;
        {
            raytracing::ray qray = ray;
            for (uint attempt = 0u; attempt < 8u; ++attempt) {
                auto sHit = isect.intersect(qray, accel);
                haveHit = sHit.type != intersection_type::none;
                hitDist = haveHit ? sHit.distance : 1e30f;
                primID = haveHit ? sHit.primitive_id : 0u;
                hitBary = haveHit ? sHit.triangle_barycentric_coord : float2(0);
                hitFront = haveHit ? sHit.triangle_front_facing : true;
                isDynHit = false;
                if (u.dynTriCount > 0) {
                    raytracing::ray dynRay = qray;
                    if (haveHit) dynRay.max_distance = hitDist;
                    auto dHit = isect.intersect(dynRay, dynAccel);
                    if (dHit.type != intersection_type::none
                        && (!haveHit || dHit.distance < hitDist)) {
                        haveHit = true; isDynHit = true;
                        hitDist = dHit.distance; primID = dHit.primitive_id;
                        hitBary = dHit.triangle_barycentric_coord;
                        hitFront = dHit.triangle_front_facing;
                    }
                }
                if (!haveHit || bounce != 0u || u.hidePlayerHead == 0u) break;
                uint hMat = (isDynHit ? dynTriMat : triMat)[primID];
                if ((materials[hMat].flags & FLAG_PLAYER_HEAD) == 0u) break;
                qray.min_distance = hitDist + 0.002f;
                haveHit = false;
            }
        }
        if (!haveHit) {
            radiance += throughput * envColor(ray.direction, u.envIntensity);
            break;
        }
        // Attenuate the segment just travelled inside glass.
        // v4.15: untinted glass (tint exactly 1) has a zero log
        // and exp(0 * d) is exactly 1 — skip the transcendental.
        if (inMedium && any(mediumLog != float3(0.0f))) {
            throughput *= exp(mediumLog * hitDist);
        }

        device const float* hPositions = isDynHit ? dynPositions : positions;
        device const float* hNormals = isDynHit ? dynNormals : normals;
        device const float* hUVs = isDynHit ? dynUVs : uvs;
        device const float* hColors = isDynHit ? dynColors : colors;
        device const uint* hIndices = isDynHit ? dynIndices : indices;
        device const uint* hTriMat = isDynHit ? dynTriMat : triMat;
        uint tri = primID;
        Material mat = materials[hTriMat[tri]];
        // ---- Emissive surface: it's a light. Count it when the path
        // arrived specularly (or from the camera); diffuse paths were
        // already lit by next-event estimation below.
        // v4.15: resolved BEFORE the attribute interpolation —
        // the emissive contribution needs the material alone, so
        // an emissive hit no longer pays for the 33-float
        // interpolation (in light-heavy scenes, emissive hits
        // are a large share of all hits).
        if (mat.flags & FLAG_EMISSIVE) {
            if (prevSpecular) {
                radiance += throughput * mat.emissive.rgb * mat.emissive.a;
            }
            break;
        }
        uint i0 = hIndices[3 * tri], i1 = hIndices[3 * tri + 1], i2 = hIndices[3 * tri + 2];
        float3 p0 = float3(hPositions[3 * i0], hPositions[3 * i0 + 1], hPositions[3 * i0 + 2]);
        float3 p1 = float3(hPositions[3 * i1], hPositions[3 * i1 + 1], hPositions[3 * i1 + 2]);
        float3 p2 = float3(hPositions[3 * i2], hPositions[3 * i2 + 1], hPositions[3 * i2 + 2]);
        float bc1 = hitBary.x;
        float bc2 = hitBary.y;
        float bc0 = 1.0f - bc1 - bc2;
        float3 hitPos = p0 * bc0 + p1 * bc1 + p2 * bc2;
        float3 nInterp = normalize(
            float3(hNormals[3 * i0], hNormals[3 * i0 + 1], hNormals[3 * i0 + 2]) * bc0 +
            float3(hNormals[3 * i1], hNormals[3 * i1 + 1], hNormals[3 * i1 + 2]) * bc1 +
            float3(hNormals[3 * i2], hNormals[3 * i2 + 1], hNormals[3 * i2 + 2]) * bc2);
        float3 vColor =
            float3(hColors[3 * i0], hColors[3 * i0 + 1], hColors[3 * i0 + 2]) * bc0 +
            float3(hColors[3 * i1], hColors[3 * i1 + 1], hColors[3 * i1 + 2]) * bc1 +
            float3(hColors[3 * i2], hColors[3 * i2 + 1], hColors[3 * i2 + 2]) * bc2;
        float2 vUV =
            float2(hUVs[2 * i0], hUVs[2 * i0 + 1]) * bc0 +
            float2(hUVs[2 * i1], hUVs[2 * i1 + 1]) * bc1 +
            float2(hUVs[2 * i2], hUVs[2 * i2 + 1]) * bc2;

        bool front = hitFront;
        float3 ns = front ? nInterp : -nInterp; // shading normal, against the ray
        float3 geoN = normalize(cross(p1 - p0, p2 - p0));
        if (dot(geoN, ns) < 0) geoN = -geoN;

        // ---- Glass: reflect / refract with Fresnel. Past the
        // refraction budget the surface falls through to diffuse
        // shading (tinted), which bounds transmission cost in
        // glass-heavy views.
        // v4.15: the decision is taken BEFORE any texture fetch.
        // A pure glass event uses only the material's glass
        // parameters and the interpolated normal — the PBR
        // sampling below (up to 9 texture reads for triplanar
        // materials) was pure waste on every glass interface.
        // The fetch consumes no RNG, so the random stream — and
        // every downstream value — is identical; only the
        // fall-through path still pays for textures, exactly as
        // before.
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
            // v4.15: nested-interface throughput roulette. Once
            // a path has crossed at least two interfaces, each
            // further glass continuation survives with
            // probability q (from the path throughput) and is
            // compensated by 1/q — the long tail of paths
            // rattling inside stacked glass stops consuming
            // intersections for negligible carried light, while
            // the estimator stays unbiased.
            if (refracts >= 2u) {
                float q = clamp(max(throughput.x, max(throughput.y, throughput.z)),
                                0.05f, 0.95f);
                if (rng.nextFloat() > q) break;
                throughput /= q;
            }
            float3 off = dot(newDir, geoN) >= 0 ? geoN : -geoN;
            ray.origin = hitPos + off * 0.0015f;
            ray.direction = newDir;
            ray.min_distance = 0.0f;
            prevSpecular = true;
            continue;
        }

        // ---- Texture fetch
        float3 albedo = mat.baseColor.rgb * vColor;
        float texRough = 0.9f;
        float3 shadingN = ns;
        if (mat.texIndex > 0) {
            // v4.1: decode the array + slice from the encoding
            // (roads: texIndex >= 4096 -> road arrays).
            bool isRoad = mat.texIndex >= 4096u;
            uint slice = isRoad ? 0u : (mat.texIndex - 1);
            uint roadSlice = isRoad ? (mat.texIndex - 4096u - 1u) : 0u;
            float scale = mat.tint.a;
            if (mat.flags & FLAG_TRIPLANAR) {
                float3 wp = hitPos * scale;
                float3 w = pow(abs(nInterp), float3(4));
                w /= max(w.x + w.y + w.z, 1e-5f);
                float3 a = samplePBR(texAlbedo, roadAlbedo, isRoad, texSampler, wp.zy, slice, roadSlice).rgb * w.x
                         + samplePBR(texAlbedo, roadAlbedo, isRoad, texSampler, wp.xz, slice, roadSlice).rgb * w.y
                         + samplePBR(texAlbedo, roadAlbedo, isRoad, texSampler, wp.xy, slice, roadSlice).rgb * w.z;
                float rgh = samplePBR(texData, roadData, isRoad, texSampler, wp.zy, slice, roadSlice).r * w.x
                          + samplePBR(texData, roadData, isRoad, texSampler, wp.xz, slice, roadSlice).r * w.y
                          + samplePBR(texData, roadData, isRoad, texSampler, wp.xy, slice, roadSlice).r * w.z;
                albedo *= a;
                texRough = rgh;
                // Triplanar normal mapping: sample the tangent-space
                // normal on each axis plane, swizzle into world
                // space (sign from the geometric normal), blend with
                // the same weights — the texture's relief now rakes
                // under light on imported models too.
                float3 tnx = samplePBR(texNormal, roadNormal, isRoad, texSampler, wp.zy, slice, roadSlice).rgb * 2.0f - 1.0f;
                float3 tny = samplePBR(texNormal, roadNormal, isRoad, texSampler, wp.xz, slice, roadSlice).rgb * 2.0f - 1.0f;
                float3 tnz = samplePBR(texNormal, roadNormal, isRoad, texSampler, wp.xy, slice, roadSlice).rgb * 2.0f - 1.0f;
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
                float2 duv1 = float2(hUVs[2 * i1] - hUVs[2 * i0], hUVs[2 * i1 + 1] - hUVs[2 * i0 + 1]);
                float2 duv2 = float2(hUVs[2 * i2] - hUVs[2 * i0], hUVs[2 * i2 + 1] - hUVs[2 * i0 + 1]);
                float det = duv1.x * duv2.y - duv2.x * duv1.y;
                float3 T = abs(det) > 1e-10f
                    ? normalize((e1 * duv2.y - e2 * duv1.y) / det)
                    : normalize(abs(ns.y) < 0.99f ? cross(ns, float3(0, 1, 0)) : cross(ns, float3(1, 0, 0)));
                float3 B = cross(ns, T);
                float hgt = samplePBR(texData, roadData, isRoad, texSampler, tuv, slice, roadSlice).g;
                float3 viewDir = -ray.direction;
                float3 vt = float3(dot(viewDir, T), dot(viewDir, B), dot(viewDir, ns));
                float2 puv = tuv + (vt.xy / max(vt.z, 0.25f)) * (hgt - 0.5f) * 0.05f;
                albedo *= samplePBR(texAlbedo, roadAlbedo, isRoad, texSampler, puv, slice, roadSlice).rgb;
                texRough = samplePBR(texData, roadData, isRoad, texSampler, puv, slice, roadSlice).r;
                float3 nm = samplePBR(texNormal, roadNormal, isRoad, texSampler, puv, slice, roadSlice).rgb * 2.0f - 1.0f;
                shadingN = normalize(T * nm.x + B * nm.y + ns * nm.z);
                if (dot(shadingN, geoN) < 0) shadingN = ns;
            }
        }

        // ---- Weather (v4.13): flat ground wetness only.
        // Rain is falling streaks + ONE wetness value: as it
        // rises, up-facing surfaces get darker, glossier and
        // more reflective (the sheen); when the rain stops
        // they dry back. No snow, no standing water, nothing
        // that can climb a wall.
        float upFace = upFaceGate(geoN.y);
        float wet = u.wetLevel * upFace;
        float gloss = (mat.texIndex > 0) ? clamp(1.0f - texRough, 0.0f, 1.0f) : 0.0f;
        // v4.8's water-reflection economy retired with the
        // standing water it served; the flag stays declared
        // (always false) for the specular roll below.
        bool waterCheap = false;
        if (wet > 0.001f) {
            albedo *= (1.0f - 0.45f * wet);
            gloss = max(gloss, wet * 0.85f);
        }

        // A glass surface that took the diffuse fall-through (budget
        // spent or transmission roll failed) shades with its tint.
        if (mat.flags & FLAG_GLASS) {
            albedo *= mat.tint.rgb;
        }

        // ---- Diffuse: next-event estimation toward emissive tris.
        // v4.15 (light): the pick is IMPORTANCE-weighted — the
        // CDF is built over area x luminance(Le), so a small hot
        // light is picked as often as its contribution warrants
        // instead of drowning among big dim panels (the picked
        // point's pdf is lum(picked) / W, hence the W / yPick
        // factor below; the integral estimated is unchanged).
        // Deep bounces are additionally THINNED: from bounce 2
        // on, NEE runs with probability pNEE (throughput-driven,
        // floored at 0.25) and its contribution is scaled by
        // 1/pNEE — compensated roulette, so the estimator stays
        // unbiased while weak paths stop paying for shadow rays.
        float neeScale = 1.0f;
        bool doNEE = u.emissiveCount > 0 && u.totalEmissiveArea > 0.0f;
        if (doNEE && bounce >= 2u) {
            float pNEE = clamp(max(throughput.x, max(throughput.y, throughput.z)) * 1.5f,
                               0.25f, 1.0f);
            if (rng.nextFloat() > pNEE) doNEE = false;
            else neeScale = 1.0f / pNEE;
        }
        if (doNEE) {
            float pickU = rng.nextFloat() * u.totalEmissiveArea;
            uint lo = 0, hi = u.emissiveCount;
            while (lo < hi) { // binary search the weight CDF
                uint mid = (lo + hi) / 2;
                if (emissiveCDF[mid] <= pickU) lo = mid + 1; else hi = mid;
            }
            uint etri = emissiveTris[min(lo, u.emissiveCount - 1)];
            // v4: emissive ids are unified across both sets.
            bool eIsDyn = etri >= u.triCount;
            uint eLocal = eIsDyn ? (etri - u.triCount) : etri;
            device const float* ePositions = eIsDyn ? dynPositions : positions;
            device const uint* eIndices = eIsDyn ? dynIndices : indices;
            device const uint* eTriMat = eIsDyn ? dynTriMat : triMat;
            uint e0 = eIndices[3 * eLocal], e1 = eIndices[3 * eLocal + 1], e2 = eIndices[3 * eLocal + 2];
            float3 q0 = float3(ePositions[3 * e0], ePositions[3 * e0 + 1], ePositions[3 * e0 + 2]);
            float3 q1 = float3(ePositions[3 * e1], ePositions[3 * e1 + 1], ePositions[3 * e1 + 2]);
            float3 q2 = float3(ePositions[3 * e2], ePositions[3 * e2 + 1], ePositions[3 * e2 + 2]);
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
                bool occluded = shit.type != intersection_type::none;
                if (!occluded && u.dynTriCount > 0) {
                    auto dshit = shadowIsect.intersect(sray, dynAccel);
                    occluded = dshit.type != intersection_type::none;
                }
                if (!occluded) {
                    Material em = materials[eTriMat[eLocal]];
                    float3 Le = em.emissive.rgb * em.emissive.a;
                    // v4.15: pdf of the sampled point under
                    // importance selection is yPick / W (W is
                    // carried in totalEmissiveArea) — divide by
                    // it. A picked light always has yPick > 0
                    // (zero-luminance tris have zero CDF weight
                    // and are unreachable); the guard covers the
                    // degenerate all-dark table.
                    float yPick = dot(Le, float3(0.2126f, 0.7152f, 0.0722f));
                    if (yPick > 1e-6f) {
                        radiance += throughput * (albedo / M_PI_F) * Le
                                  * cosS * cosE
                                  * (u.totalEmissiveArea / yPick) / d2 * neeScale;
                    }
                }
            }
        }

        // ---- Diffuse bounce, with a glossy sheen roll for wet
        // or low-roughness textured surfaces: with probability
        // pSpec (Fresnel- and gloss-driven) the bounce goes
        // specular instead — a jittered reflection, tinted only
        // partway toward the albedo, and allowed to count
        // emissive hits directly on the next segment.
        // v4.8: unselected water pixels take the environment
        // fill for their reflection and end the path here --
        // the structural saving is the reflection ray itself
        // (and everything it would have hit downstream), which
        // is why reflection rays per frame on water pixels
        // scale by ~waterReflScale. The fill is the same
        // envColor gradient a miss returns, tinted like the
        // specular roll tints its bounce, so the puddle still
        // reads as water -- flatter and dimmer than a true
        // reflection, which is the honest cost of the setting.
        if (waterCheap) {
            float3 rdir = reflect(ray.direction, shadingN);
            float3 fillTint = mix(float3(1.0f), albedo, 0.45f)
                            * (0.35f + 0.65f * gloss);
            radiance += throughput * fillTint
                      * envColor(rdir, u.envIntensity);
            break;
        }
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
        // v4.15 (reflections): roughness-aware continuation
        // roulette for the specular roll. A specular bounce
        // survives with probability qSpec = 0.35 + 0.65 * gloss
        // and survivors are compensated by 1/qSpec: mirror-like
        // reflections (gloss near 1) continue almost always,
        // while rough sheen — whose continuation averages a
        // wide, dim cone anyway — ends after this hit's own
        // shading (its next-event estimate above already
        // landed). Compensated, so the estimator is unbiased.
        float specSurvive = 1.0f;
        if (bounceSpecular) {
            specSurvive = clamp(0.35f + 0.65f * gloss, 0.35f, 1.0f);
            if (rng.nextFloat() > specSurvive) break;
        }
        throughput *= bounceTint / specSurvive;
        ray.origin = hitPos + geoN * 0.0015f;
        ray.direction = newDir;
        ray.min_distance = 0.0f;
        prevSpecular = bounceSpecular;

        // Russian roulette once the path is deep (v4.15: from
        // bounce 2, one earlier than before — same throughput
        // rule, same compensation, so the expectation is
        // unchanged and weak paths die one event sooner).
        if (bounce >= 2) {
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
    // bright content is not biased down. (v4.15: prevA is the
    // read taken at the top of the kernel.)
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

    // v4.15: maintain the sample moments alongside the sum
    // (live path only). The moving-average decay mirrors the
    // accumulation's exactly — the sum decays by s, the sum of
    // squares by s^2 — so the variance gate always estimates
    // the same window the accumulator holds.
    if (u.traceFlags & 2u) {
        float lumS = dot(radiance, float3(0.2126f, 0.7152f, 0.0722f));
        float2 m = momentsIn;
        if (prevA.a >= u.historyCap) {
            float s = (u.historyCap - 1.0f) / u.historyCap;
            m *= float2(s, s * s);
        }
        m += float2(lumS, lumS * lumS);
        moments[pixel.y * width + pixel.x] = m;
    }
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
    texture2d<float> weatherField [[texture(4)]],
    constant Uniforms& u [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device const float* normals [[buffer(2)]],
    device const uint* indices [[buffer(5)]],
    device const uint* triMat [[buffer(6)]],
    device const Material* materials [[buffer(7)]],
    acceleration_structure<> accel [[buffer(10)]],
    device const float* dynPositions [[buffer(11)]],
    device const float* dynNormals [[buffer(12)]],
    device const float* dynUVs [[buffer(13)]],
    device const float* dynColors [[buffer(14)]],
    device const uint* dynIndices [[buffer(15)]],
    device const uint* dynTriMat [[buffer(16)]],
    acceleration_structure<> dynAccel [[buffer(17)]],
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
    constexpr sampler fieldSampler(address::clamp_to_edge, filter::linear);
    // v4: merged static + dynamic primary hit (nearer wins),
    // same material resolution as the trace kernel — a dynamic
    // FLAG_SHEET / glass / reflection-dominant hit therefore
    // gets exactly the trace kernel's reusability treatment in
    // the ghosting logic below.
    // v4.5: the FLAG_PLAYER_HEAD step-over is applied here
    // identically (this kernel IS the primary ray), so the
    // guide position/normal/class describe the same surface
    // the trace kernel shades — no head-shaped ghost in the
    // denoised or reprojected image.
    bool haveHit = false;
    float hitDist = 1e30f;
    uint primID = 0u;
    float2 hitBary = float2(0);
    bool isDynHit = false;
    {
        raytracing::ray qray = ray;
        for (uint attempt = 0u; attempt < 8u; ++attempt) {
            auto sHit = isect.intersect(qray, accel);
            haveHit = sHit.type != intersection_type::none;
            hitDist = haveHit ? sHit.distance : 1e30f;
            primID = haveHit ? sHit.primitive_id : 0u;
            hitBary = haveHit ? sHit.triangle_barycentric_coord : float2(0);
            isDynHit = false;
            if (u.dynTriCount > 0) {
                raytracing::ray dynRay = qray;
                if (haveHit) dynRay.max_distance = hitDist;
                auto dHit = isect.intersect(dynRay, dynAccel);
                if (dHit.type != intersection_type::none
                    && (!haveHit || dHit.distance < hitDist)) {
                    haveHit = true; isDynHit = true;
                    hitDist = dHit.distance; primID = dHit.primitive_id;
                    hitBary = dHit.triangle_barycentric_coord;
                }
            }
            if (!haveHit || u.hidePlayerHead == 0u) break;
            uint hMat = (isDynHit ? dynTriMat : triMat)[primID];
            if ((materials[hMat].flags & FLAG_PLAYER_HEAD) == 0u) break;
            qray.min_distance = hitDist + 0.002f;
            haveHit = false;
        }
    }
    if (!haveHit) {
        guide.write(float4(0, 0, 0, 0), pixel);
        guideNormal.write(float4(0, 0, 0, 0), pixel);
        return;
    }
    (void)dynUVs; (void)dynColors;
    device const float* hPositions = isDynHit ? dynPositions : positions;
    device const float* hNormals = isDynHit ? dynNormals : normals;
    device const uint* hIndices = isDynHit ? dynIndices : indices;
    device const uint* hTriMat = isDynHit ? dynTriMat : triMat;
    uint tri = primID;
    uint i0 = hIndices[3 * tri], i1 = hIndices[3 * tri + 1], i2 = hIndices[3 * tri + 2];
    float3 p0 = float3(hPositions[3 * i0], hPositions[3 * i0 + 1], hPositions[3 * i0 + 2]);
    float3 p1 = float3(hPositions[3 * i1], hPositions[3 * i1 + 1], hPositions[3 * i1 + 2]);
    float3 p2 = float3(hPositions[3 * i2], hPositions[3 * i2 + 1], hPositions[3 * i2 + 2]);
    float bc1 = hitBary.x;
    float bc2 = hitBary.y;
    float bc0 = 1.0f - bc1 - bc2;
    float3 hitPos = p0 * bc0 + p1 * bc1 + p2 * bc2;
    float3 nInterp = normalize(
        float3(hNormals[3 * i0], hNormals[3 * i0 + 1], hNormals[3 * i0 + 2]) * bc0 +
        float3(hNormals[3 * i1], hNormals[3 * i1 + 1], hNormals[3 * i1 + 2]) * bc1 +
        float3(hNormals[3 * i2], hNormals[3 * i2 + 1], hNormals[3 * i2 + 2]) * bc2);
    Material mat = materials[hTriMat[tri]];
    float reusable = (mat.flags & FLAG_GLASS) ? 0.0f : 1.0f;
    if (reusable > 0.5f) {
        // v3.4 (reflection ghosting fix): history reuse is
        // anchored on the primary hit POSITION — the right
        // anchor for diffuse radiance, the wrong one for
        // reflections, whose content changes the instant the
        // viewpoint moves. Reflection-dominant first hits
        // restart like glass: near-mirror gloss at any angle,
        // or strong gloss whose Fresnel term at this view
        // angle makes the reflection carry the pixel.
        // v4.1 (F3): textured gloss comes from the material's
        // MEAN texture roughness (glassParams.w).
        float glossEff = (mat.texIndex > 0)
            ? clamp(1.0f - mat.glassParams.w, 0.0f, 1.0f) : 0.0f;
        // v4.13: the weather gloss model mirrors the trace
        // kernel's flat wetness — up-facing hits gloss up
        // with the wet level.
        float3 geoN = normalize(cross(p1 - p0, p2 - p0));
        if (dot(geoN, nInterp) < 0.0f) geoN = -geoN;
        float upFace = upFaceGate(geoN.y);
        glossEff = max(glossEff, u.wetLevel * upFace * 0.85f);
        float cosV = clamp(dot(-ray.direction, nInterp), 0.15f, 1.0f);
        float fres = 0.04f + 0.96f * pow(1.0f - cosV, 5.0f);
        if (glossEff >= 0.92f
            || (glossEff >= 0.55f && glossEff * fres > 0.18f)) {
            reusable = 0.0f;
        }
    }
    guide.write(float4(hitPos, reusable), pixel);
    guideNormal.write(float4(nInterp, reusable), pixel);
}

// MARK: - Spatial denoise kernel (v3: interactive anti-sparkle)
//
// A 3x3 cross-bilateral filter over the accumulation, guided by
// the primary-hit position, the interpolated surface normal, and
// luminance. Neighbours blend only within the same reusability
// class (diffuse/emissive vs glass/environment), so the filter
// never smears across glass or silhouette edges. Output keeps
// the accumulated-sum convention (rgb = mean * alpha) so the
// tonemap is unchanged. This pass is DISPLAY-ONLY: it reads the
// accumulation and writes a separate display texture; the
// accumulation itself — and therefore the settled frame and
// every still export — is never touched by it.
//
// v4.2 REWORK (device report 2026-10-31: "still getting a
// little bit of fuzz", worst on an emissive-lit textured face
// and in a dark glossy reflection): the v3 fade handed over to
// raw at ~96 effective samples for EVERY class, and the pass
// stopped running at 160 global samples — leaving the whole
// 160→768 settle tail raw. Diffuse at 96 samples is fairly
// clean; glossy/reflection pixels (the non-reusable class)
// carry several times the per-sample variance and got the
// same 96 budget. Now the fade is class-aware and driven by
// uniforms: the blend reaches zero at fadeEnd for the reusable
// class and at fadeEndSpec (the settle cap) for the
// non-reusable class, along a smoothstep that starts at a
// quarter of the hand-off. The non-reusable class also gets a
// wider luminance sigma and a softer normal exponent below:
// its neighbours legitimately differ (each reflects a
// different part of the room), so the v3 weights rejected
// exactly the neighbours the noisiest class needed. Young
// pixels (< 32 effective samples — reflection pixels restart
// from zero on every camera move) additionally sample four
// diagonal taps at distance 2, phased out by 32 samples.

struct DenoiseUniforms {
    float3 camOrigin;
    float strength;
    // Effective-sample count where the blend reaches zero, per
    // reusability class (v4.2; formerly two pad floats).
    float fadeEnd;
    float fadeEndSpec;
};

static float denoiseLuminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

// One neighbour's bilateral weight (position × normal ×
// luminance), before the spatial kernel weight is applied.
static float denoiseTapWeight(float3 cPos, float3 cN, float cLum,
                              float3 nPos, float3 nN, float nLum,
                              float sp, float lumSigma, float nPow) {
    float3 dp = nPos - cPos;
    float wp = exp(-dot(dp, dp) / (2.0f * sp * sp));
    float wn = pow(clamp(dot(cN, nN), 0.0f, 1.0f), nPow);
    float wl = exp(-abs(nLum - cLum) / lumSigma);
    return wp * wn * wl;
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

    // v4.2: the non-reusable class (glass / environment /
    // reflection-dominant) gets a wider luminance sigma and a
    // softer normal exponent — its neighbours legitimately
    // differ in radiance (each reflects a different part of
    // the room), and the v3 weights rejected them, leaving
    // the highest-variance class with the least filtering.
    bool specClass = gc.w < 0.5f;
    float lumSigma = (0.12f + 0.25f * lumC) * (specClass ? 2.25f : 1.0f);
    float nPow = specClass ? 4.0f : 6.0f;

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
            float ws = (dx == 0 && dy == 0) ? (4.0f / 16.0f)
                     : ((dx == 0 || dy == 0) ? (2.0f / 16.0f) : (1.0f / 16.0f));
            float w = denoiseTapWeight(gc.xyz, nc, lumC, g.xyz, nn,
                                       denoiseLuminance(meanN),
                                       sp, lumSigma, nPow) * ws;
            sum += meanN * w;
            wSum += w;
        }
    }
    // v4.2: young pixels get four extra diagonal taps at
    // distance 2 (a 5x5-ish footprint for the worst noise),
    // phased out by 32 effective samples.
    float young = clamp((32.0f - c.a) / 32.0f, 0.0f, 1.0f);
    if (young > 0.0f) {
        for (int sy = -1; sy <= 1; sy += 2) {
            for (int sx = -1; sx <= 1; sx += 2) {
                int2 np = int2(pixel) + int2(2 * sx, 2 * sy);
                if (np.x < 0 || np.y < 0 || np.x >= int(width) || np.y >= int(height)) continue;
                float4 g = guide.read(uint2(np));
                if (g.w != gc.w) continue;
                float4 nAcc = accum.read(uint2(np));
                if (nAcc.a < 1.0f) continue;
                float3 meanN = nAcc.rgb / max(nAcc.a, 1.0f);
                float3 nn = guideNormal.read(uint2(np)).xyz;
                float w = denoiseTapWeight(gc.xyz, nc, lumC, g.xyz, nn,
                                           denoiseLuminance(meanN),
                                           sp, lumSigma, nPow)
                        * (1.0f / 16.0f) * young;
                sum += meanN * w;
                wSum += w;
            }
        }
    }
    float3 filtered = wSum > 1e-6f ? sum / wSum : mean;
    // v4.2: class-aware hand-off — the blend now reaches zero
    // at the per-class fade end (a smoothstep starting at a
    // quarter of it) instead of a flat 96 samples for all.
    float fadeEnd = specClass ? u.fadeEndSpec : u.fadeEnd;
    float fade = fadeEnd > 0.0f
        ? 1.0f - smoothstep(0.25f * fadeEnd, fadeEnd, c.a) : 0.0f;
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
    device const float2* prevMoments [[buffer(1)]],
    device float2* curMoments [[buffer(2)]],
    uint2 pixel [[thread_position_in_grid]])
{
    uint width = curGuide.get_width();
    uint height = curGuide.get_height();
    if (pixel.x >= width || pixel.y >= height) return;
    // v4.15: the sample moments travel with the accumulation —
    // every branch that zeroes the accumulation zeroes the
    // moments; the carry branch rescales them with the same
    // weight ratio (sum by s, sum of squares by s^2).
    uint midx = pixel.y * width + pixel.x;

    float4 g = curGuide.read(pixel);
    if (g.w < 0.5f || u.reuseCap <= 0.0f) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        curMoments[midx] = float2(0);
        return;
    }
    float3 rel = g.xyz - u.prevOrigin;
    float z = dot(rel, u.prevForward);
    if (z <= 1e-4f) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        curMoments[midx] = float2(0);
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
        curMoments[midx] = float2(0);
        return;
    }
    uint pw = prevGuide.get_width();
    uint ph = prevGuide.get_height();
    uint2 pp = uint2(min(uint(uv.x * float(pw)), pw - 1),
                     min(uint(uv.y * float(ph)), ph - 1));
    float4 pg = prevGuide.read(pp);
    if (pg.w < 0.5f) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        curMoments[midx] = float2(0);
        return;
    }
    float eps = max(0.02f, 0.002f * length(rel));
    if (distance(pg.xyz, g.xyz) > eps) {
        curAccum.write(float4(0, 0, 0, 0), pixel);
        curMoments[midx] = float2(0);
        return;
    }
    float4 h = prevAccum.read(pp);
    float wgt = min(h.a, u.reuseCap);
    float3 rgb = h.a > 0.001f ? h.rgb * (wgt / h.a) : float3(0);
    curAccum.write(float4(rgb, wgt), pixel);
    float2 pm = prevMoments[pp.y * pw + pp.x];
    float ms = h.a > 0.001f ? (wgt / h.a) : 0.0f;
    curMoments[midx] = pm * float2(ms, ms * ms);
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
    // Rain-ripple shimmer (v4.13): while it rains, wet
    // up-facing hits get a small time-varying brightness
    // modulation in linear space — expanding rain rings over
    // the flat wetness, uniform wherever the ground is wet
    // (no puddle mask; there is no standing water anymore).
    // The converged image underneath stays untouched; stills
    // and exports pass shimmerRain = 0 and skip this entirely.
    if (p.shimmerRain > 0.01f) {
        float4 g = guide.sample(s, in.uv);
        float4 gn = guideNormal.sample(s, in.uv);
        if (g.a > 0.5f && gn.y > 0.5f) {
            float3 wp = g.xyz;
            float upFace = smoothstep(0.5f, 0.85f, gn.y);
            float wetM = upFace * smoothstep(0.03f, 0.15f, p.wetLevel);
            if (wetM > 0.01f) {
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
                float m = 1.0f + 0.16f * rings * wetM
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
    // v4.8 (steering 5): alpha is computed PROCEDURALLY from the
    // quad UV with signed-distance shapes, and is exactly 0 in a
    // border band on every side of the quad for BOTH kinds -- so
    // no square card silhouette can ever show, whatever the
    // blend mode or texture state. Rain is a capsule around the
    // streak axis; snow is a radial disc reaching exactly 0 by
    // r = 0.85 of the half-extent.
    float2 p = in.uv - float2(0.5f); // -0.5..0.5, quad borders at |p| = 0.5
    if (in.kind < 1.5f) {
        // Rain capsule: distance to the vertical segment
        // y in [-0.33, 0.33] on the streak axis (x = 0).
        float segY = clamp(p.y, -0.33f, 0.33f);
        float dist = length(float2(p.x, p.y - segY));
        // Soft edge: full inside 0.05, zero at 0.16 (quad units);
        // the quad half-width is 0.5, so alpha is 0 for the
        // outer 68% of the half-width on each side.
        float body = 1.0f - smoothstep(0.05f, 0.16f, dist);
        // Tip fade along the length so the ends round off.
        float tip = 1.0f - smoothstep(0.34f, 0.50f, abs(p.y));
        float alpha = 0.50f * body * tip * in.fade;
        return float4(float3(0.72f, 0.80f, 0.95f), alpha);
    }
    // Snow: radial falloff, exactly 0 at r >= 0.85 half-extents.
    float r = length(p) * 2.0f; // 0 centre, 1 at edge midpoints
    float disc = 1.0f - smoothstep(0.45f, 0.85f, r);
    float alpha = 0.9f * disc * in.fade;
    return float4(float3(0.93f, 0.96f, 1.0f), alpha);
}

// MARK: - Area-box overlay (v4: edit-mode debug/area boxes)
//
// Tiny line pipeline: Swift expands each debug box into 24
// line-list vertices (12 edges) with a per-vertex colour.
// Projection is manual, exactly like the precipitation vertex:
// rel = world - camOrigin, z = dot(rel, forward), NDC from the
// right/up components over (z * tanFov * aspect).

struct LineUniforms {
    float3 camOrigin;
    float3 camRight;
    float3 camUp;
    float3 camForward;
    float2 tanFovAspect; // x: tan(half vertical fov), y: aspect
    float pad0;
    float pad1;
};

struct LineOut {
    float4 pos [[position]];
    float4 color;
};

vertex LineOut lineVertex(uint vid [[vertex_id]],
                          constant LineUniforms& u [[buffer(0)]],
                          device const float4* verts [[buffer(1)]],
                          device const float4* cols [[buffer(2)]]) {
    LineOut out;
    out.color = cols[vid];
    float3 world = verts[vid].xyz;
    float3 rel = world - u.camOrigin;
    float z = dot(rel, u.camForward);
    if (z < 0.05f) {
        out.pos = float4(3.0f, 3.0f, 0.5f, 1.0f); // clipped
        return out;
    }
    float tanF = u.tanFovAspect.x;
    float2 ndc = float2(dot(rel, u.camRight) / (z * tanF * u.tanFovAspect.y),
                        dot(rel, u.camUp) / (z * tanF));
    out.pos = float4(ndc.x, ndc.y, 0.5f, 1.0f);
    return out;
}

fragment float4 lineFragment(LineOut in [[stage_in]]) {
    return in.color;
}

// MARK: - Weather simulation (v4.8 water / v4.12 snow)
//
// WATER is the v4.8 system, unchanged: depths live in a
// persistent per-lattice-vertex buffer (waterDepth) stepped by
// simSourceSinkKernel (rain fill toward the pooling ceiling,
// drying, edge drainage) and simFlowKernel kind==1 (levelling
// in red-black order). Sean's puddle behaviour — round,
// confined, no climbing — is exactly that system's, and the
// v4.8 audit still covers it.
//
// SNOW is the v4.12 GAMES MODEL (Sean, 2026-11-02: "Do what
// games typically do"): authored depth + carve-only local
// deformation + shader accumulation — the stack games actually
// ship (Arkham Origins -> Rise of the Tomb Raider -> God of
// War / RDR2). It replaces v4.11's granular mass-transport sim,
// whose support-switch transfer piled a smooth ramp up a box's
// vertical face on device: snow is never transported sideways,
// so no bank can ever form. The model (snowFieldKernel +
// snowRenderKernel, below): per GROUND vertex, field (authored
// depth — a value, not mass) and carve (compression below the
// field); rendered = clamp(field − carve + rim, 0,
// field + 0.10), the rim being the one permitted raise
// (≤ 0.10 m lip at a depression's edge). The lattice covers
// ground supports only; snow on objects is the shader layer
// (snowAccumFactor), which is exactly zero on vertical faces
// by construction.
//   simSourceSinkKernel's snow half stays gated OFF by
//   sp.snowMode == 1 (kept set); it still runs water.
//   simFlowKernel now runs kind==1 (water) only.
//   simApplyEditsKernel is dead (kept compiled; no dispatch).
//
// The closed-form height function survives as (a) the
// per-vertex authored ceiling + drift-rate field (inputs to
// the field's growth), and (b) the load-time initial
// condition. The rendered sheet is a skin over this state,
// clipped at object footprints (see Renderer.appendSnowSheet).

