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
//     (clear -> frosted as roughness rises); the tint colours the
//     transmitted light.
//   * Textures are procedural PBR sets (albedo / tangent normal /
//     roughness+height) sampled per material slice, with one-step
//     parallax from the height channel; STEP imports (no UVs) use
//     triplanar world-space sampling instead.
//   * Everything else is Lambertian with cosine-weighted bounces
//     and Russian roulette. One sample per dispatch is accumulated
//     into a float texture; the tonemap pass divides by the sample
//     count, applies ACES and gamma.

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
};

struct Material {
    float4 baseColor;    // rgb; a unused
    float4 emissive;     // rgb colour; a = intensity
    float4 glassParams;  // x: transmission, y: roughness, z: ior, w unused
    float4 tint;         // rgb; a = textureScale
    uint texIndex;       // 0 none, 1..3 texture array slice+1 (slice = texIndex-1)
    uint flags;          // bit0 emissive on, bit1 glass on, bit2 triplanar
    uint2 pad;
};

constant uint FLAG_EMISSIVE = 1u;
constant uint FLAG_GLASS = 2u;
constant uint FLAG_TRIPLANAR = 4u;

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

// MARK: - Path trace kernel

kernel void pathTraceKernel(
    texture2d<float, access::read_write> accum [[texture(0)]],
    texture2d_array<float> texAlbedo [[texture(1)]],
    texture2d_array<float> texNormal [[texture(2)]],
    texture2d_array<float> texData [[texture(3)]],
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
    acceleration_structure<primitive_data> accel [[buffer(10)]],
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

    // Primary ray.
    float2 ndc = float2((float(pixel.x) + rng.nextFloat()) / float(width),
                        (float(pixel.y) + rng.nextFloat()) / float(height));
    float2 d = (ndc * 2.0f - 1.0f) * float2(u.tanFovAspect.y, 1.0f) * u.tanFovAspect.x;
    raytracing::ray ray;
    ray.origin = u.camOrigin;
    ray.direction = normalize(u.camForward + u.camRight * d.x + u.camUp * (-d.y));
    ray.min_distance = 0.0f;
    ray.max_distance = 1e30f;

    intersector<triangle_data, primitive_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.accept_any_intersection(false);
    intersector<triangle_data, primitive_data> shadowIsect;
    shadowIsect.assume_geometry_type(geometry_type::triangle);
    shadowIsect.accept_any_intersection(true);

    float3 throughput = float3(1);
    float3 radiance = float3(0);
    bool prevSpecular = true; // primary ray: emissive hits count directly

    for (uint bounce = 0; bounce <= u.maxBounces; ++bounce) {
        auto hit = isect.intersect(ray, accel);
        if (hit.type == intersection_type::none) {
            radiance += throughput * envColor(ray.direction, u.envIntensity);
            break;
        }

        uint tri = hit.primitive_index;
        uint i0 = indices[3 * tri], i1 = indices[3 * tri + 1], i2 = indices[3 * tri + 2];
        float3 p0 = float3(positions[3 * i0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
        float3 p1 = float3(positions[3 * i1], positions[3 * i1 + 1], positions[3 * i1 + 2]);
        float3 p2 = float3(positions[3 * i2], positions[3 * i2 + 1], positions[3 * i2 + 2]);
        float bc1 = hit.barycentric_coordinates.x;
        float bc2 = hit.barycentric_coordinates.y;
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
            } else {
                float2 tuv = vUV * scale;
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
        (void)texRough;

        // ---- Glass: reflect / refract with Fresnel.
        if ((mat.flags & FLAG_GLASS) && rng.nextFloat() < mat.glassParams.x) {
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
                }
                throughput *= mat.tint.rgb;
            }
            float3 off = dot(newDir, geoN) >= 0 ? geoN : -geoN;
            ray.origin = hitPos + off * 0.0015f;
            ray.direction = newDir;
            ray.min_distance = 0.0f;
            prevSpecular = true;
            continue;
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

        // ---- Diffuse bounce.
        float3 newDir = cosineDirection(shadingN, rng);
        throughput *= albedo;
        ray.origin = hitPos + geoN * 0.0015f;
        ray.direction = newDir;
        ray.min_distance = 0.0f;
        prevSpecular = false;

        // Russian roulette once the path is deep.
        if (bounce >= 3) {
            float q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (rng.nextFloat() > q) break;
            throughput /= q;
        }
    }

    float4 prev = accum.read(pixel);
    accum.write(float4(prev.rgb + radiance, 1.0f), pixel);
}

// MARK: - Tonemap / display

struct TonemapParams {
    uint samples;
    float exposure;
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
                                constant TonemapParams& p [[buffer(0)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float3 c = accum.sample(s, in.uv).rgb / max(1.0f, float(p.samples));
    c = aces(c * p.exposure);
    c = pow(c, float3(1.0f / 2.2f));
    return float4(c, 1.0f);
}
