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
    uint dynTriCount;          // v4: dynamic-layer triangle count (unified ids start at triCount)
    uint hidePlayerHead;       // v4.5: 1 while the camera is the Play first-person eye — the player figure's head/hair (FLAG_PLAYER_HEAD) are excluded from PRIMARY hits only (was padDyn; layout unchanged)
    float freezeFrac;          // v4.7: puddle freeze fraction 0...1 (tail field; mirrors Renderer.swift GPUUniforms)
    float waterReflScale;      // v4.8: water reflection resolution 1.0 / 0.5 / 0.25 (tail; mirrors GPUUniforms)
    uint frameCounter;         // v4.8: per-presented-frame counter for the rotating reflection pattern (tail; mirrors GPUUniforms)
    uint reflPad;
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

kernel void weatherFieldKernel(
    texture2d<float, access::write> field [[texture(0)]],
    constant WeatherFieldParams& fp [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]])
{
    uint w = field.get_width();
    uint h = field.get_height();
    if (pixel.x >= w || pixel.y >= h) return;
    float2 uv = (float2(pixel) + 0.5f) / float2(float(w), float(h));
    float2 xz = (uv - 0.5f) * 64.0f;
    // v4.3: EVERY channel is evaluated at the seed-shifted
    // point; the puddle / wet / broad formulas themselves are
    // unchanged (offset only).
    float2 sp = xz + fp.offset;
    float pot = wpuddlePotential(sp * (1.0f / 1.15f));
    float wetPool = wfbm2(sp * 1.3f + float2(4.7f, 9.1f));
    float drift;
    if (fp.seeded != 0u) {
        // Multi-scale, wind-anisotropic drift (v4.3), in
        // lockstep with wgfDrift in Renderer.swift: rotate the
        // shifted point by the wind heading, then broad
        // (stretched along the wind -> elongated drifts) +
        // lumps + fine detail. The bake writes the RAW value;
        // the window fade is applied at sample time, exactly
        // as the legacy channel was.
        float2 q = float2(fp.wind.x * sp.x - fp.wind.y * sp.y,
                          fp.wind.y * sp.x + fp.wind.x * sp.y);
        float driftBroad = wfbm2(float2(q.x * 0.30f, q.y * 0.52f)
                                 + float2(9.17f, 27.3f));
        float lumps = wfbm2(q * 0.95f + float2(31.7f, 5.3f));
        float fine = wnoise2(q * 2.7f + float2(13.1f, 41.9f));
        drift = 0.58f * driftBroad + 0.30f * lumps + 0.12f * fine;
    } else {
        drift = wfbm2(sp * 0.5f + float2(9.17f, 27.3f));
    }
    float broad = wfbm2(sp * 0.32f + float2(17.31f, 3.97f));
    field.write(float4(pot, wetPool, drift, broad), pixel);
}

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
        if (inMedium) {
            throughput *= exp(mediumLog * hitDist);
        }

        device const float* hPositions = isDynHit ? dynPositions : positions;
        device const float* hNormals = isDynHit ? dynNormals : normals;
        device const float* hUVs = isDynHit ? dynUVs : uvs;
        device const float* hColors = isDynHit ? dynColors : colors;
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
        float3 vColor =
            float3(hColors[3 * i0], hColors[3 * i0 + 1], hColors[3 * i0 + 2]) * bc0 +
            float3(hColors[3 * i1], hColors[3 * i1 + 1], hColors[3 * i1 + 2]) * bc1 +
            float3(hColors[3 * i2], hColors[3 * i2 + 1], hColors[3 * i2 + 2]) * bc2;
        float2 vUV =
            float2(hUVs[2 * i0], hUVs[2 * i0 + 1]) * bc0 +
            float2(hUVs[2 * i1], hUVs[2 * i1 + 1]) * bc1 +
            float2(hUVs[2 * i2], hUVs[2 * i2 + 1]) * bc2;

        Material mat = materials[hTriMat[tri]];
        bool front = hitFront;
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

        // ---- Weather + gloss (v3). Texture roughness finally
        // shades: gloss = 1 - roughness for textured surfaces.
        // Wetness darkens up-facing surfaces and raises their
        // gloss; snow whitens them, kills the gloss, and pulls
        // the shading normal toward the geometric up-face.
        float upFace = smoothstep(0.5f, 0.85f, geoN.y);
        float wet = u.wetLevel * upFace;
        float snowGlobal = u.snowLevel * upFace;
        float gloss = (mat.texIndex > 0) ? clamp(1.0f - texRough, 0.0f, 1.0f) : 0.0f;
        // v3.4: puddle water sheets are real geometry carrying
        // FLAG_SHEET. They shade as near-mirror standing water and
        // skip the whole weather block below — no wet darkening,
        // no snow cover, no second puddle mask on the water
        // itself (the sheet's placement already IS the mask).
        bool isSheet = (mat.flags & FLAG_SHEET) != 0u;
        // v4.4: snow sheet geometry (FLAG_SNOW) shades as
        // snow and only snow — it skips the weather block
        // below exactly like the water sheet does (no wet
        // darkening, no gloss change, no SECOND snow cover
        // computed over the snow itself), and since the flag
        // replaces the host material entirely, the glass and
        // emissive branches above can never claim it either:
        // its albedo is the material's snow colour times the
        // per-vertex drift shade baked CPU-side.
        bool isSnow = (mat.flags & FLAG_SNOW) != 0u;
        // v4.8 (steering 1): water reflection resolution. At the
        // default scale 1.0 nothing below ever triggers and water
        // shades exactly as v4.7. Below 1.0, a rotating per-pixel
        // pattern (pixel hash XOR frame counter, so coverage
        // cycles across frames) selects which water-hit pixels
        // trace the full scene reflection; the rest take the
        // cheaper environment fill at the bounce below. Only
        // sheet (standing-water) hits pay or save this cost --
        // an audit of the v4.7 kernel showed reflection work for
        // water lives entirely in the glossy specular roll for
        // isSheet / mirror-puddle hits, and no other pixel class
        // enters that roll with water gloss, so nothing is
        // removed from non-water pixels (there was nothing to
        // remove) and no reflection is faked at the default.
        bool waterCheap = false;
        if (isSheet && u.waterReflScale < 0.999f) {
            uint rh = hash32(pixel.x * 1973u ^ pixel.y * 9277u
                             ^ (u.frameCounter * 2654435761u)
                             ^ (bounce * 40503u));
            float rf = float(rh >> 8) * (1.0f / 16777216.0f);
            waterCheap = rf >= u.waterReflScale;
        }
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
        if (!isSheet && !isSnow && (u.wetLevel > 0.001f || u.snowLevel > 0.001f) && bounce <= 1u) {
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
        if (!isSheet && !isSnow) {
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
            // v4.8: a dominant mask puddle is water too -- the
            // same rotating selection applies to its reflection.
            if (puddle > 0.5f && u.waterReflScale < 0.999f) {
                uint rh = hash32(pixel.x * 1973u ^ pixel.y * 9277u
                                 ^ (u.frameCounter * 2654435761u)
                                 ^ (bounce * 40503u));
                float rf = float(rh >> 8) * (1.0f / 16777216.0f);
                waterCheap = rf >= u.waterReflScale;
            }
            // v4.7: mask puddles freeze coherently with sheets.
            if (u.freezeFrac > 0.001f) {
                float3 iceCol = float3(0.55f, 0.65f, 0.75f);
                albedo = mix(albedo, iceCol, clamp(puddle * u.freezeFrac, 0.0f, 1.0f) * 0.92f);
            }
        }
        // 3. Snow: whitened, gloss killed, normal mounded by the
        // drift gradient (or softened toward the face, as v3).
        if (snowCover > 0.001f) {
            snowCover *= (1.0f - puddle); // no snow inside puddles
        }
        // v4.3: per-object snow exposure — Material.pad.x is a
        // Float bit pattern carrying this object's accumulated
        // share of the ambient snow (spawned objects start at
        // 0 = bare; long-standing objects reach 1 = full).
        {
            float snowScale = as_type<float>(mat.pad.x);
            snowCover *= snowScale;
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
        } // !isSheet && !isSnow — weather steps 1-3 skipped for water and snow sheets
        if (isSheet) {
            // Standing water: near-mirror gloss, and the shading
            // normal is the sheet's real (flat) surface normal,
            // so the reflection is stable, has true parallax
            // against the floor beneath, and tracks the camera.
            gloss = 0.97f;
            shadingN = geoN;
            // v4.7: frozen sheets shade as pale ice (still mirror).
            if (u.freezeFrac > 0.001f) {
                float3 iceCol = float3(0.55f, 0.65f, 0.75f);
                albedo = mix(albedo, iceCol, clamp(u.freezeFrac, 0.0f, 1.0f) * 0.92f);
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
    if ((mat.flags & FLAG_SHEET) != 0u) {
        // v3.4: water sheets are pure reflection — view-dependent,
        // never seeded from history (same treatment as glass).
        reusable = 0.0f;
    } else if ((mat.flags & FLAG_SNOW) != 0u) {
        // v4.4: the snow sheet is a matte diffuse surface —
        // full history reuse, and the weather mirror below
        // must not run for it (its puddle term would restart
        // history on snow that happens to lie over a puddle
        // area of the field, and its gloss model describes
        // the HOST surface under the snow, not the snow).
    } else if (reusable > 0.5f) {
        // v3.4 (reflection ghosting fix): history reuse is
        // anchored on the primary hit POSITION — the right anchor
        // for diffuse radiance, the wrong one for reflections,
        // whose content changes the instant the viewpoint moves.
        // Position-anchored reuse smeared the old reflection
        // across the new one for the life of the history tail
        // (device report 2026-10-30: a light lingering in puddles
        // ~1 s after the camera moved). Reflection-dominant first
        // hits now restart like glass: near-mirror gloss at any
        // angle, or strong gloss whose Fresnel term at this view
        // angle makes the reflection carry the pixel. Diffuse and
        // matte surfaces keep full reuse — the motion clarity
        // the reprojection exists for is untouched there.
        // v4.1 (F3): textured gloss comes from the material's
        // MEAN texture roughness (glassParams.w, baked CPU-side
        // from the slice's data channel) — NOT glassParams.y,
        // which is the glass frost roughness (default 0.05) and
        // made every textured surface read as near-mirror here,
        // silently disabling reprojection for all of them.
        float glossEff = (mat.texIndex > 0)
            ? clamp(1.0f - mat.glassParams.w, 0.0f, 1.0f) : 0.0f;
        // v4.1 (F7): the weather gloss model now MIRRORS the
        // trace kernel, from the baked field sampled at the
        // hit's world position (the guide has world positions):
        // wetF gloss, the thresholded puddle MASK (a mirror
        // puddle reaches gloss 1 and restarts, as in the trace),
        // and snowCover from the DRIFTED depth with upFace
        // multiplied OUTSIDE the smoothstep — the trace kernel's
        // exact structure and constants. (The old code used the
        // raw global wet level and smoothed snowLevel * upFace
        // INSIDE the smoothstep with no drift.)
        float3 geoN = normalize(cross(p1 - p0, p2 - p0));
        if (dot(geoN, nInterp) < 0.0f) geoN = -geoN;
        float upFace = smoothstep(0.5f, 0.85f, geoN.y);
        float puddle = 0.0f;
        float snowCover = u.snowLevel * upFace; // trace default (no field)
        if (u.wetLevel > 0.001f || u.snowLevel > 0.001f) {
            float2 fuv = hitPos.xz * (1.0f / 64.0f) + 0.5f;
            float fieldFade = (1.0f - smoothstep(29.0f, 32.0f, abs(hitPos.x)))
                            * (1.0f - smoothstep(29.0f, 32.0f, abs(hitPos.z)));
            float4 wfield = mix(float4(0.0f, 0.5f, 0.5f, 0.489f),
                                weatherField.sample(fieldSampler, fuv),
                                fieldFade);
            // Wet: darkened, glossier (trace step 1, field-shaped).
            float wetF = clamp(u.wetLevel * upFace * (0.7f + 0.6f * wfield.g),
                               0.0f, 1.0f);
            // Trickle runnels (v4.1 N3): the trace kernel's
            // slanted-face streak term, ported exactly — slope
            // amount and downhill frame from the GEOMETRIC
            // normal (geoN, from the hit triangle p0/p1/p2
            // above, as in trace), one wnoise2 evaluation,
            // added to wetF after the base field-shaped wetF
            // and before the gloss terms below consume it.
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
            glossEff = max(glossEff, wetF * 0.85f);
            // Puddle mask (trace step 2): threshold sweep of
            // the baked metaball potential.
            float elevBias = clamp(0.10f - hitPos.y * 0.55f, -0.22f, 0.38f);
            float slopeShed = clamp((geoN.y - 0.70f) / 0.30f, 0.0f, 1.0f);
            float wetN = clamp(u.wetLevel, 0.0f, 1.0f);
            float thr = mix(1.15f, 0.08f, pow(wetN, 1.15f)) - elevBias * 0.45f;
            float mask = smoothstep(thr - 0.14f, thr + 0.14f, wfield.r);
            float broad = 0.78f + 0.45f * wfield.a;
            puddle = clamp(mask * broad, 0.0f, 1.0f) * slopeShed * upFace
                   * fieldFade * smoothstep(0.03f, 0.15f, u.wetLevel);
            glossEff = max(glossEff, puddle); // mirror puddle => restart
            // Snow (trace step 3): drifted depth -> cover.
            float drift = wfield.b;
            float depth = u.snowLevel * mix(0.3f, 1.7f, drift);
            snowCover = smoothstep(0.45f, 0.75f, depth) * upFace;
            snowCover *= (1.0f - puddle); // no snow inside puddles
        }
        // v4.3: per-object snow exposure, mirroring the trace
        // kernel — Material.pad.x is a Float bit pattern with
        // this object's accumulated share of the ambient snow,
        // applied after the puddle suppression above (and to
        // the no-field default snowCover as well).
        {
            float snowScale = as_type<float>(mat.pad.x);
            snowCover *= snowScale;
        }
        // Snow cover kills the gloss again in the trace kernel.
        glossEff *= 1.0f - snowCover;
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

// MARK: - Weather simulation (v4.8 water / v4.11 snow physics)
//
// WATER is the v4.8 system, unchanged: depths live in a
// persistent per-lattice-vertex buffer (waterDepth) stepped by
// simSourceSinkKernel (rain fill toward the pooling ceiling,
// drying, edge drainage) and simFlowKernel kind==1 (levelling
// in red-black order). Sean's puddle behaviour — round,
// confined, no climbing — is exactly that system's, and the
// v4.8 audit still covers it.
//
// SNOW was rebuilt in v4.11 as a physically-based granular
// height-field simulation (Sean, 2026-11-01: "Snow is now a
// simulation... affected by physics and gravity... very very
// very accurate to how real snow works"; the v4.2-v4.10.4
// rule-stack — closed-form ceilings as shapes, exposure clocks,
// the plow layer, drag caps, relaxation patches — is RETIRED).
// The model (design record + Python mirror proof in the goal
// record, hidden_files/v411-snow-audit.py — PILE / SETTLE /
// DRAG / SPAWN / ROOF / LONGRUN all pass there, and the old
// kernel rules fail the same SETTLE field at 79.7 deg):
//
//   State per lattice column: MASS m (SWE metres), DENSITY
//   rho in [~0.35, 1.0], sinter AGE. Depth h = m / rho is
//   DERIVED (snowWritebackKernel refreshes the snowDepth
//   buffer the skin and water read). Per substep (1/30 s):
//     1. snowSourcesKernel — deposition as an input FLUX
//        (base rate x drift factor x fresh density, softly
//        gated by the per-vertex gate = the old ceiling value
//        demoted to an input gate), melt as mass removal at
//        the session sink rate, overburden + gate + intruder
//        compaction, footprint press events (CPU-stamped),
//        ageing. This is the ONLY place snow mass is created
//        or destroyed (plus the sync-time initial condition).
//     2. snowFluxKernel — gravity flux: pairwise transfers
//        over the snow SURFACE (support + depth) across ALL
//        EIGHT neighbours (diagonals included: axial-only
//        constraints let corner gradients reach sqrt(2) x
//        repose — measured 43.7 deg at box corners in the
//        mirror). ONE synchronous (Jacobi) sweep per substep:
//        snowSnapKernel first copies the mass, and every
//        transfer is decided from that snapshot — chained
//        sweeps ratcheted piles below repose over minutes
//        (mirror PILE-LONG). transfer = 0.25 x mobility x
//        excess over the repose plane, the giver's total
//        outflow scaled to never exceed its snapshot mass
//        (the limiter makes conservation exact — without it
//        parallel giving at support steps CREATED mass in
//        the mirror).
//        Mobility falls with density and sinter age; the
//        repose angle is temperature-driven (34 deg cold to
//        38 deg at 0 C, + up to 4 deg of sinter bonus). No
//        diffusion term. Support steps are NOT skipped: snow
//        sheds off edges and wall-adjacent fins relax, which
//        is the v4.10.4 device failure this replaces.
//     3. snowDisplaceCountKernel + snowDisplaceApplyKernel —
//        objects as volume boundaries: the CPU rasterises the
//        live intruder set into occupancy buffers every tick;
//        intruded columns vacate a rate-limited fraction of
//        their overlap mass each substep (compaction absorbs
//        first, in step 1), distributed to non-intruded
//        neighbours lowest-surface-first. Berms and trenches
//        EMERGE from this + flux; nothing is recorded.
//   simSourceSinkKernel's snow half is gated OFF by
//   sp.snowMode == 1 (the v4.11 path); it still runs water.
//   simFlowKernel now runs kind==1 (water) only.
//   simApplyEditsKernel is dead (kept compiled; no dispatch).
//
// The closed-form height function survives ONLY as (a) the
// per-vertex deposition GATE + drift-rate field (inputs), and
// (b) the load-time initial condition (converted to mass via
// a density profile at sync). The rendered sheet is a skin
// over this state, as before.

struct SimParams {
    uint vx;          // lattice vertices per row
    uint vz;          // lattice rows
    uint parity;      // red-black pass parity (flow kernel)
    uint kind;        // flow kernel: 0 = snow, 1 = water, 2 = both (v4.9 merged dispatch)
    float dt;         // seconds this substep
    float cell;       // lattice cell size (m)
    float snowSink;   // uniform snow sink rate (m/s): temp melt or legacy shrink
    float compactRate;// m/s toward ceiling when snow exceeds it
    float waterDry;   // water drying rate (m/s)
    float waterDrain; // extra water loss at support edges (m/s)
    float freezeFrac; // 0..1; water flow scales by (1 - freezeFrac)
    float snowBase;   // v4.8: live snow base rate (m/s at drift factor 1)
    float waterBase;  // v4.8: live water fill scale (surge x rate scale)
    uint snowMode;    // v4.11: 1 = snow owned by the physics kernels
                      // (source/sink skips its snow half; flow runs water only)
    float reposeTan;  // v4.11: tan(angle of repose) at the current temperature
    float freshRho;   // v4.11: fresh-snow density at the current temperature
    float snowTemp;   // v4.11: air temperature (C); >= 100 = unset (legacy)
    float gateUniform;// v4.11: the user's Max Depth (uniform, NOT
                      // drift-factored): the gate-compaction threshold
};

kernel void simSourceSinkKernel(
    device float* snowDepth [[buffer(0)]],
    device float* waterDepth [[buffer(1)]],
    device const float* supportY [[buffer(2)]],
    device const float* snowCeil [[buffer(3)]],
    device const float* snowRate [[buffer(4)]],
    device const float* waterCeil [[buffer(5)]],
    device const float* waterRate [[buffer(6)]],
    device const uint* hasSupport [[buffer(7)]],
    constant SimParams& sp [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    uint count = sp.vx * sp.vz;
    if (idx >= count) return;
    if (hasSupport[idx] == 0u) {
        // v4.11: in snowMode the snow depth buffer is owned by
        // snowWritebackKernel (it zeroes unsupported columns
        // itself); only water is still this kernel's to clear.
        if (sp.snowMode == 0u) { snowDepth[idx] = 0.0f; }
        waterDepth[idx] = 0.0f;
        return;
    }
    // Snow (LEGACY path only — v4.11 sets snowMode == 1 and
    // snowSourcesKernel owns snow): deposit toward the ceiling
    // at the drift rate; above the ceiling compact toward it;
    // uniform sinks (temperature melt / legacy terms).
    if (sp.snowMode == 0u) {
        float sn = snowDepth[idx];
        float ceil = snowCeil[idx];
        if (sn < ceil) {
            sn = min(ceil, sn + snowRate[idx] * sp.snowBase * sp.dt);
        } else if (sn > ceil) {
            sn = max(ceil, sn - sp.compactRate * sp.dt);
        }
        if (sp.snowSink > 0.0f) sn = max(0.0f, sn - sp.snowSink * sp.dt);
        snowDepth[idx] = sn;
    }
    // Water: rain fills toward the pooling ceiling (the baked
    // puddle profile for this vertex -- zero where the field
    // never pools, so rain alone cannot flood a flat plane
    // edge-to-edge); drying removes at the ~10 min rate.
    float wa = waterDepth[idx];
    if (wa < waterCeil[idx]) {
        wa = min(waterCeil[idx], wa + waterRate[idx] * sp.waterBase * sp.dt);
    }
    if (sp.waterDry > 0.0f) wa = max(0.0f, wa - sp.waterDry * sp.dt);
    // Edge drainage: any missing 4-neighbour means this vertex
    // sits on a support boundary -- water there drains off the
    // edge instead of standing in a ledge against thin air.
    uint x = idx % sp.vx;
    uint z = idx / sp.vx;
    bool edge = false;
    if (x == 0u || z == 0u || x + 1u >= sp.vx || z + 1u >= sp.vz) {
        edge = true;
    } else {
        if (hasSupport[idx - 1u] == 0u || hasSupport[idx + 1u] == 0u
            || hasSupport[idx - sp.vx] == 0u || hasSupport[idx + sp.vx] == 0u) {
            edge = true;
        }
    }
    if (edge && sp.waterDrain > 0.0f) {
        wa = max(0.0f, wa - sp.waterDrain * sp.dt);
    }
    waterDepth[idx] = wa;
}

kernel void simFlowKernel(
    device float* snowDepth [[buffer(0)]],
    device float* waterDepth [[buffer(1)]],
    device const float* supportY [[buffer(2)]],
    device const float* waterCeil [[buffer(5)]],
    device const uint* hasSupport [[buffer(7)]],
    device const uint* pinned [[buffer(9)]],
    constant SimParams& sp [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    uint count = sp.vx * sp.vz;
    if (idx >= count) return;
    if (hasSupport[idx] == 0u) return;
    uint x = idx % sp.vx;
    uint z = idx / sp.vx;
    if (((x + z) & 1u) != sp.parity) return;
    const float maxStep = 0.753554f * sp.cell; // tan(37 deg) * cell
    // Four neighbours; opposite colour by construction, so a
    // neighbour is never written by another thread this pass.
    const int dxs[4] = {1, -1, 0, 0};
    const int dzs[4] = {0, 0, 1, -1};
    // v4.9: kind == 2 runs BOTH kinds in this one dispatch.
    // The two kinds touch DISJOINT writable state (snow writes
    // snowDepth only, water writes waterDepth only; supportY,
    // waterCeil, hasSupport and pinned are read-only here), so
    // running them sequentially in this thread, snow first then
    // water, performs exactly the same transfers per buffer in
    // the same neighbour order as the two separate dispatches
    // did -- every transfer's arithmetic is unchanged. Parity
    // passes stay separate dispatches: red and black within one
    // dispatch would race (there is no grid-wide sync).
    const uint kindCount = (sp.kind >= 2u) ? 2u : 1u;
    for (uint ki = 0u; ki < kindCount; ++ki) {
        const uint kind = (sp.kind >= 2u) ? ki : sp.kind;
    for (uint k = 0u; k < 4u; ++k) {
        int nx = int(x) + dxs[k];
        int nz = int(z) + dzs[k];
        if (nx < 0 || nz < 0 || nx >= int(sp.vx) || nz >= int(sp.vz)) continue;
        uint n = uint(nz) * sp.vx + uint(nx);
        // v4.8 (steering 3): water at a support boundary
        // DRAINS in the flow pass itself -- a missing neighbour
        // (support edge) or a lower support across a step takes
        // half of the cell's water per visit and removes
        // it. A source-sink drain rate alone loses to the
        // levelling refill from the pool behind it (host audit
        // v4.8: edge ring held 9.4 mm against the rate drain);
        // draining in the flow is what guarantees water never
        // stands past an edge or piles against a step.
        if (kind == 1u
            && (hasSupport[n] == 0u
                || (abs(supportY[idx] - supportY[n]) > 0.05f
                    && supportY[n] < supportY[idx]))) {
            float fs = 1.0f - clamp(sp.freezeFrac, 0.0f, 1.0f);
            if (fs > 0.001f) {
                waterDepth[idx] -= waterDepth[idx] * 0.5f * fs;
            }
            continue;
        }
        if (hasSupport[n] == 0u) continue;
        // Same-support edges only: a genuine support step is
        // the object's silhouette, not snow/water holding a
        // cliff (the v4.5 rule, kept verbatim).
        if (abs(supportY[idx] - supportY[n]) > 0.05f) continue;
        if (kind == 0u) {
            // Snow. Pinned columns (resting intruder, carved
            // trail floor) neither give nor receive.
            if (pinned[idx] != 0u || pinned[n] != 0u) continue;
            float ha = supportY[idx] + snowDepth[idx];
            float hb = supportY[n] + snowDepth[n];
            float diff = ha - hb;
            float move = 0.0f;
            if (abs(diff) > maxStep) {
                float excess = abs(diff) - maxStep;
                // v4.10.4: 0.25, not 0.5. Within one parity
                // dispatch up to 4 same-parity givers can move
                // snow into a shared receiver simultaneously,
                // each reading the receiver's stale depth; at
                // 0.5 the receiver could gain ~2x the pairwise
                // excess and become a NEW local maximum above
                // its own ceiling (the at-rest degeneration).
                // At 1/4 per giver the worst-case receiver gain
                // is the largest single pair excess, so the
                // maximum principle holds for the 4-neighbour
                // stencil and the repose fixed point (the
                // maxStep angle) is unchanged.
                move = excess * 0.25f;
            }
            // Rounding diffusion: 4% of the depth difference
            // per visit, always on -- piles relax into soft
            // mounds instead of holding faceted steps. Small
            // enough that drift variety survives (the audit
            // measures drift contrast before/after).
            move += (snowDepth[idx] - snowDepth[n]) * 0.008f;
            // Clamp: a cell never gives more snow than it holds.
            if (move > 0.0f) move = min(move, snowDepth[idx]);
            if (move < 0.0f) move = -min(-move, snowDepth[n]);
            if (move != 0.0f) {
                snowDepth[idx] -= move;
                snowDepth[n] += move;
            }
        } else {
            // Confinement: water never flows INTO a dry cell
            // outside the pooling field (waterCeil == 0). On a
            // dead-flat plane, unconfined levelling bleeds a
            // pool into a plane-wide film (host audit v4.8) --
            // the pooling field is what makes puddles puddles.
            if (waterCeil[n] <= 0.0f && waterDepth[n] <= 0.0f
                && (supportY[idx] + waterDepth[idx])
                   > (supportY[n] + waterDepth[n])) continue;
            if (waterCeil[idx] <= 0.0f && waterDepth[idx] <= 0.0f
                && (supportY[n] + waterDepth[n])
                   > (supportY[idx] + waterDepth[idx])) continue;
            // Water levelling: move toward equal free surface.
            float flowScale = 1.0f - clamp(sp.freezeFrac, 0.0f, 1.0f);
            if (flowScale <= 0.001f) continue; // frozen: rigid
            float la = supportY[idx] + waterDepth[idx];
            float lb = supportY[n] + waterDepth[n];
            float move = (la - lb) * 0.25f * flowScale;
            if (move > 0.0f) move = min(move, waterDepth[idx]);
            if (move < 0.0f) move = -min(-move, waterDepth[n]);
            // Rounding diffusion on depth itself (surface
            // tension): blurs the perimeter into blobs.
            move += (waterDepth[idx] - waterDepth[n]) * 0.03f * flowScale;
            if (move > 0.0f) move = min(move, waterDepth[idx]);
            if (move < 0.0f) move = -min(-move, waterDepth[n]);
            if (move != 0.0f) {
                waterDepth[idx] -= move;
                waterDepth[n] += move;
            }
        }
    }
    } // kind loop (v4.9 merged dispatch)
}

kernel void simApplyEditsKernel(
    device float* snowDepth [[buffer(0)]],
    device const uint* editIndex [[buffer(10)]],
    device const float* editDelta [[buffer(11)]],
    constant uint& editCount [[buffer(12)]],
    constant uint& latticeCount [[buffer(13)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= editCount) return;
    uint v = editIndex[idx];
    // v4.10.2: this kernel currently has NO dispatch site
    // (the CPU applies edit transfers in appendSnowSheet),
    // but its write was the one unbounded GPU write in the
    // sim shader set — guard it before anyone wires it up.
    if (v >= latticeCount) return;
    snowDepth[v] = max(0.0f, snowDepth[v] + editDelta[idx]);
}

// MARK: - v4.11 snow physics kernels
//
// Constants below are ported verbatim from the Python mirror
// (hidden_files/v411-snow-audit.py), where PILE / SETTLE /
// DRAG / SPAWN / ROOF / LONGRUN pass with them. Do not retune
// one side without the other.

struct SnowEdge {
    uint idx;
    float mass;
};

// Shared per-column helpers (mirror: mobility / repose).
static float snowMobility(float rho, float age, constant SimParams& sp) {
    float mob = 1.0f / (1.0f + max(rho - sp.freshRho, 0.0f) * 3.0f);
    float agef = clamp(age / 600.0f, 0.0f, 1.0f);
    mob *= (1.0f - 0.5f * agef);
    if (sp.snowTemp < 100.0f && sp.snowTemp > 0.0f) mob *= 1.25f;
    return max(mob, 0.15f);
}
static float snowReposeTan(float age, constant SimParams& sp) {
    float agef = clamp(age / 600.0f, 0.0f, 1.0f);
    return sp.reposeTan + 0.0699268f * agef; // + tan(4 deg) sinter bonus
}

// Step 0 of each substep: clear the displacement edge counter.
kernel void snowEdgeResetKernel(
    device uint* edgeCount [[buffer(0)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx == 0u) { edgeCount[0] = 0u; }
}

// Step 1: sources / sinks / compaction / press / ageing.
// The ONLY kernel that creates or destroys snow mass
// (deposition in, melt out); press overflow is not destroyed
// — it is appended to the displacement edge buffer and lands
// on the neighbours in the apply step.
kernel void snowSourcesKernel(
    device float* snowMass [[buffer(0)]],
    device float* snowRho [[buffer(1)]],
    device float* snowAge [[buffer(2)]],
    device const float* supportY [[buffer(3)]],
    device const float* snowGate [[buffer(4)]],
    device const float* snowRate [[buffer(5)]],
    device const uint* hasSupport [[buffer(6)]],
    device const float* occCov [[buffer(7)]],
    device const float* press [[buffer(8)]],
    device SnowEdge* edges [[buffer(9)]],
    device uint* edgeCount [[buffer(10)]],
    constant uint& edgeCapacity [[buffer(11)]],
    constant SimParams& sp [[buffer(12)]],
    uint idx [[thread_position_in_grid]])
{
    uint count = sp.vx * sp.vz;
    if (idx >= count) return;
    if (hasSupport[idx] == 0u) {
        snowMass[idx] = 0.0f;
        snowAge[idx] = 0.0f;
        return;
    }
    float m = snowMass[idx];
    float rho = clamp(snowRho[idx], 0.05f, 1.0f);
    float age = snowAge[idx];
    float gate = snowGate[idx];
    // Deposition: an input flux at fresh density, softly
    // gated as depth approaches the gate (the gate is the
    // old ceiling value — an input limit, never a shape).
    float depth = m / max(rho, 0.05f);
    if (sp.snowBase > 0.0f) {
        float taper = clamp((gate - depth) / 0.05f, 0.0f, 1.0f);
        float dm = sp.snowBase * snowRate[idx] * sp.freshRho * sp.dt * taper;
        if (dm > 0.0f) {
            rho = (m * rho + dm * sp.freshRho) / max(m + dm, 1e-9f);
            m += dm;
        }
    }
    // Melt: mass out at the session sink rate (a depth rate:
    // dm = sink * rho * dt keeps dh/dt == sink). Wet snow
    // densifies slightly while it melts.
    if (sp.snowSink > 0.0f && m > 0.0f) {
        float dm = min(m, sp.snowSink * rho * sp.dt);
        m -= dm;
        if (sp.snowTemp < 100.0f && sp.snowTemp > 0.0f) {
            rho += (1.0f - rho) * 0.005f * sp.dt;
        }
    }
    // Overburden compaction: the pack densifies under its own
    // weight, faster the deeper it is. (Rate set by the
    // mirror's PILE-LONG test: at 0.001 the DIFFERENTIAL rate
    // (peak carries the most mass, compacts fastest) sank a
    // test pile's peak ~25% in 300 s and dragged its flank
    // from 33 deg to 26 deg — real densification of a fresh
    // pack acts over hours. 0.0001: e-fold ~3 h at reference
    // load; invisible over a scene session's minutes, gentle
    // subsidence over hours.)
    {
        float k = 0.0001f * (0.3f + m / 0.5f);
        rho += (1.0f - rho) * (1.0f - exp(-k * sp.dt));
    }
    // Gate compaction: snow sitting ABOVE the user's Max
    // Depth (the uniform cap, NOT the drift-factored
    // deposition gate — a displaced pile in a low-drift spot
    // must not silently densify away; the drift field shapes
    // where snowfall may accumulate, never how fast placed
    // snow vanishes) densifies — depth falls at constant
    // mass. Mass is never deleted here; v4.7's
    // delete-to-ceiling is deliberately gone (see the design
    // record).
    depth = m / max(rho, 0.05f);
    if (depth > sp.gateUniform) {
        rho += (1.0f - rho) * (1.0f - exp(-0.02f * sp.dt));
    }
    // Intruder compaction: snow actually OVERLAPPED by an
    // object is pressed toward full density fast — this is
    // the "compact first" half of displacement; the rest
    // vacates in step 3. The overlap test is the same one
    // the displacement step uses (v4.11 fix round 4: the
    // coverage mask alone also matches the SUPPORT ITSELF —
    // the ground plane is an "intruder" covering every
    // column with zero overlap — and compacting on coverage
    // alone densified ALL ground snow to rho ~ 1 within
    // seconds, collapsing pile depths by the fresh-density
    // factor; the in-app PILE audit caught it: volume fell
    // 2.41 -> 1.09 at constant mass).
    if (occCov[idx] > 0.0f) {
        float depthNow = m / max(rho, 0.05f);
        float ovTop = min(occHi[idx] - supportY[idx], depthNow);
        float ovBot = max(occLo[idx] - supportY[idx], 0.0f);
        float ovFrac = clamp((ovTop - ovBot) / max(depthNow, 1e-6f), 0.0f, 1.0f);
        if (ovFrac > 0.0f) {
            rho += (1.0f - rho) * (1.0f - exp(-3.0f * ovFrac * sp.dt));
            age = 0.0f;
        }
    }
    // Footprint press (player steps, CPU-stamped per-vertex
    // compress factors): depth scales down at constant mass
    // (rho rises); past full density the excess mass sheds
    // to the ring via the edge buffer.
    float p = press[idx];
    if (p > 0.001f && m > 0.0f) {
        depth = m / max(rho, 0.05f);
        float target = depth * (1.0f - min(p, 0.9f));
        float rhoWant = m / max(target, 1e-6f);
        if (rhoWant <= 1.0f) {
            rho = rhoWant;
        } else {
            rho = 1.0f;
            float over = m - target;
            if (over > 0.0f) {
                m = target;
                uint slot = atomic_fetch_add_explicit(
                    (device atomic_uint*)edgeCount, 1u,
                    memory_order_relaxed);
                if (slot < edgeCapacity) {
                    edges[slot].idx = idx;
                    edges[slot].mass = over;
                }
                // If the buffer were ever full the excess
                // stays in the column (m restored) rather
                // than vanishing: conservation is absolute.
                else { m += over; }
            }
        }
        age = 0.0f;
    }
    age += sp.dt;
    snowMass[idx] = m;
    snowRho[idx] = clamp(rho, 0.05f, 1.0f);
    snowAge[idx] = age;
}

// Step 2: gravity flux. One thread per column of the pass's
// colour class (sp.parity = class 0..3 of the 4-colouring
// (x&1) + 2*(z&1)); a thread owns its column (plain subtract
// of its own outflow — same-class columns never receive from
// each other, and other classes are not running) and adds to
// receivers atomically (several givers share receivers; the
// adds commute and none are lost, so mass is conserved to
// float rounding). Transfers are computed from the surfaces
// the thread reads; staleness between givers is bounded by
// the 0.25 factor and the outflow limiter, exactly as the
// mirror's snapshot passes are.
// v4.11 Jacobi snapshot copy: the flux pass reads ALL its
// decisions from this snapshot of the substep-start mass, so
// no transfer can chain off another transfer in the same
// substep (the chained colour sweeps ratcheted piles below
// repose — see the design record).
kernel void snowSnapKernel(
    device const float* snowMass [[buffer(0)]],
    device float* snowMassSnap [[buffer(1)]],
    constant SimParams& sp [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= sp.vx * sp.vz) return;
    snowMassSnap[idx] = snowMass[idx];
}

kernel void snowFluxKernel(
    device float* snowMass [[buffer(0)]],
    device const float* snowRho [[buffer(1)]],
    device float* snowAge [[buffer(2)]],
    device const float* supportY [[buffer(3)]],
    device const uint* hasSupport [[buffer(4)]],
    constant SimParams& sp [[buffer(5)]],
    device const float* snowMassSnap [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    uint count = sp.vx * sp.vz;
    if (idx >= count) return;
    if (hasSupport[idx] == 0u) return;
    uint x = idx % sp.vx;
    uint z = idx / sp.vx;
    // ONE synchronous sweep per substep: every decision below
    // reads the SNAPSHOT (buffer 6), never the live buffer, so
    // the repose set is a true equilibrium of the update.
    float m = snowMassSnap[idx];
    if (m <= 0.0f) return;
    float rho = clamp(snowRho[idx], 0.05f, 1.0f);
    float surfI = supportY[idx] + m / rho;
    float mobI = snowMobility(rho, snowAge[idx], sp);
    float rtanI = snowReposeTan(snowAge[idx], sp);
    const int dxs[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dzs[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    uint nIdx[8];
    float nMove[8];
    int nCount = 0;
    float totalOut = 0.0f;
    for (uint k = 0u; k < 8u; ++k) {
        int nx = int(x) + dxs[k];
        int nz = int(z) + dzs[k];
        if (nx < 0 || nz < 0 || nx >= int(sp.vx) || nz >= int(sp.vz)) continue;
        uint n = uint(nz) * sp.vx + uint(nx);
        if (hasSupport[n] == 0u) continue;
        float dist = sp.cell * ((dxs[k] != 0 && dzs[k] != 0) ? 1.41421356f : 1.0f);
        float rhoN = clamp(snowRho[n], 0.05f, 1.0f);
        float surfN = supportY[n] + snowMassSnap[n] / rhoN;
        float diff = surfI - surfN;
        if (diff <= 0.0f) continue;
        float maxStep = max(rtanI, snowReposeTan(snowAge[n], sp)) * dist;
        float excess = diff - maxStep;
        if (excess <= 0.0f) continue;
        float mob = min(mobI, snowMobility(rhoN, snowAge[n], sp));
        float move = min(0.25f * mob * excess * rho, m);
        if (move <= 0.0f) continue;
        nIdx[nCount] = n;
        nMove[nCount] = move;
        nCount += 1;
        totalOut += move;
    }
    if (nCount == 0 || totalOut <= 0.0f) return;
    // Outflow limiter against the snapshot mass: the giver
    // never gives more than it had at substep start.
    float scale = (totalOut > m) ? (m / totalOut) : 1.0f;
    float given = 0.0f;
    for (int k = 0; k < nCount; ++k) {
        float mv = nMove[k] * scale;
        if (mv <= 0.0f) continue;
        atomic_fetch_add_explicit(
            (device atomic_float*)&snowMass[nIdx[k]], mv,
            memory_order_relaxed);
        given += mv;
    }
    if (given > 0.0f) {
        atomic_fetch_sub_explicit(
            (device atomic_float*)&snowMass[idx], given,
            memory_order_relaxed);
    }
    if (given > 1e-6f) {
        // Disturbed snow loses its sinter age. (The write is
        // the owner's own column; receivers' ages are reset by
        // their own threads when they next give, and by the
        // sources kernel when intruded — age is a material
        // heuristic, not conserved state.)
        snowAge[idx] = 0.0f;
    }
}

// Step 3a: displacement count. One thread per column: if an
// intruder's volume overlaps this column's snow, append the
// rate-limited vacated mass to the edge buffer. The mass is
// subtracted in the apply step (exactly once per entry).
kernel void snowDisplaceCountKernel(
    device const float* snowMass [[buffer(0)]],
    device const float* snowRho [[buffer(1)]],
    device const float* supportY [[buffer(2)]],
    device const uint* hasSupport [[buffer(3)]],
    device const float* occCov [[buffer(4)]],
    device const float* occLo [[buffer(5)]],
    device const float* occHi [[buffer(6)]],
    device SnowEdge* edges [[buffer(7)]],
    device uint* edgeCount [[buffer(8)]],
    constant uint& edgeCapacity [[buffer(9)]],
    constant SimParams& sp [[buffer(10)]],
    uint idx [[thread_position_in_grid]])
{
    uint count = sp.vx * sp.vz;
    if (idx >= count) return;
    if (hasSupport[idx] == 0u) return;
    float cov = occCov[idx];
    if (cov <= 0.0f) return;
    float m = snowMass[idx];
    if (m <= 0.0f) return;
    float rho = clamp(snowRho[idx], 0.05f, 1.0f);
    float depth = m / rho;
    float ovTop = min(occHi[idx] - supportY[idx], depth);
    float ovBot = max(occLo[idx] - supportY[idx], 0.0f);
    float ovH = ovTop - ovBot;
    if (ovH <= 0.0f) return;
    float shed = m * (ovH / max(depth, 1e-6f)) * cov * 0.18f;
    if (shed <= 1e-7f) return;
    uint slot = atomic_fetch_add_explicit(
        (device atomic_uint*)edgeCount, 1u, memory_order_relaxed);
    if (slot < edgeCapacity) {
        edges[slot].idx = idx;
        edges[slot].mass = shed;
    }
    // Buffer full: the snow simply stays this substep (it
    // vacates next substep instead) — never deleted.
}

// Step 3b: displacement apply. One thread per edge entry:
// distribute the shed mass to the giver's non-intruded
// neighbours, weighted by surface headroom (lowest surfaces
// first — berms shoulder against the mover, never teleport
// bands), then subtract exactly the distributed sum from the
// giver. All writes atomic: several entries can share
// receivers, and a giver here is another entry's neighbour.
kernel void snowDisplaceApplyKernel(
    device float* snowMass [[buffer(0)]],
    device const float* snowRho [[buffer(1)]],
    device const float* supportY [[buffer(2)]],
    device const uint* hasSupport [[buffer(3)]],
    device const float* occCov [[buffer(4)]],
    device const SnowEdge* edges [[buffer(5)]],
    device const uint* edgeCount [[buffer(6)]],
    constant SimParams& sp [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= edgeCount[0]) return;
    SnowEdge e = edges[idx];
    uint g = e.idx;
    if (g >= sp.vx * sp.vz) return;
    float shed = e.mass;
    if (shed <= 0.0f) return;
    uint x = g % sp.vx;
    uint z = g / sp.vx;
    float rhoG = clamp(snowRho[g], 0.05f, 1.0f);
    float surfG = supportY[g] + snowMass[g] / rhoG;
    const int dxs[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dzs[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    uint nIdx[8];
    float nW[8];
    int nCount = 0;
    float wSum = 0.0f;
    for (uint k = 0u; k < 8u; ++k) {
        int nx = int(x) + dxs[k];
        int nz = int(z) + dzs[k];
        if (nx < 0 || nz < 0 || nx >= int(sp.vx) || nz >= int(sp.vz)) continue;
        uint n = uint(nz) * sp.vx + uint(nx);
        if (hasSupport[n] == 0u) continue;
        if (occCov[n] > 0.0f) continue; // into the mover: no
        float rhoN = clamp(snowRho[n], 0.05f, 1.0f);
        float surfN = supportY[n] + snowMass[n] / rhoN;
        float w = max(surfG - surfN, 0.0f) + 0.01f;
        nIdx[nCount] = n;
        nW[nCount] = w;
        nCount += 1;
        wSum += w;
    }
    if (nCount == 0 || wSum <= 0.0f) return; // boxed in: stays
    float given = 0.0f;
    for (int k = 0; k < nCount; ++k) {
        float share = shed * (nW[k] / wSum);
        if (share <= 0.0f) continue;
        atomic_fetch_add_explicit(
            (device atomic_float*)&snowMass[nIdx[k]], share,
            memory_order_relaxed);
        given += share;
    }
    if (given > 0.0f) {
        atomic_fetch_add_explicit(
            (device atomic_float*)&snowMass[g], -given,
            memory_order_relaxed);
    }
}

// Press stamps are one-shot events: after the first substep's
// sources kernel consumes them, this clears the stamp buffer
// so the remaining substeps of the tick do not re-apply them.
kernel void snowPressClearKernel(
    device float* press [[buffer(0)]],
    uint idx [[thread_position_in_grid]])
{
    press[idx] = 0.0f;
}

// Step 4: writeback — the derived depth the skin, the water
// ceiling and snowDepthAt read (mass / density).
kernel void snowWritebackKernel(
    device float* snowDepth [[buffer(0)]],
    device const float* snowMass [[buffer(1)]],
    device const float* snowRho [[buffer(2)]],
    device const uint* hasSupport [[buffer(3)]],
    constant SimParams& sp [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    uint count = sp.vx * sp.vz;
    if (idx >= count) return;
    if (hasSupport[idx] == 0u) { snowDepth[idx] = 0.0f; return; }
    snowDepth[idx] = snowMass[idx] / clamp(snowRho[idx], 0.05f, 1.0f);
}
