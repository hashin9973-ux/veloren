#version 440 core
// #extension GL_ARB_texture_storage : require

#include <constants.glsl>

#define LIGHTING_TYPE LIGHTING_TYPE_REFLECTION

#define LIGHTING_REFLECTION_KIND LIGHTING_REFLECTION_KIND_GLOSSY

#if (FLUID_MODE == FLUID_MODE_LOW)
    #define LIGHTING_TRANSPORT_MODE LIGHTING_TRANSPORT_MODE_IMPORTANCE
#elif (FLUID_MODE >= FLUID_MODE_MEDIUM)
    #define LIGHTING_TRANSPORT_MODE LIGHTING_TRANSPORT_MODE_RADIANCE
#endif

#define LIGHTING_DISTRIBUTION_SCHEME LIGHTING_DISTRIBUTION_SCHEME_MICROFACET

#define LIGHTING_DISTRIBUTION LIGHTING_DISTRIBUTION_BECKMANN

#define HAS_SHADOW_MAPS

#include <globals.glsl>
#include <random.glsl>

layout(location = 0) in vec3 f_pos;
layout(location = 1) flat in uint f_pos_norm;
layout(location = 3) in vec2 f_uv_pos;

layout(set = 2, binding = 0)
uniform texture2D t_col_light;
layout(set = 2, binding = 1)
uniform sampler s_col_light;
layout(set = 2, binding = 2)
uniform utexture2D t_kind;
layout(set = 2, binding = 3)
uniform sampler s_kind;

// Terrain PBR arrays: layer order is defined in renderer/mod.rs.
layout(set = 2, binding = 4) uniform texture2DArray t_pbr_albedo;
layout(set = 2, binding = 5) uniform sampler s_pbr_albedo;
layout(set = 2, binding = 6) uniform texture2DArray t_pbr_normal;
layout(set = 2, binding = 7) uniform sampler s_pbr_normal;
layout(set = 2, binding = 8) uniform texture2DArray t_pbr_specular;
layout(set = 2, binding = 9) uniform sampler s_pbr_specular;

layout (std140, set = 3, binding = 0)
uniform u_locals {
    mat4 model_mat;
    ivec4 atlas_offs;
    float load_time;
};

layout(location = 0) out vec4 tgt_color;
layout(location = 1) out uvec4 tgt_mat;

#include <sky.glsl>
#include <light.glsl>
#include <lod.glsl>

float vmin(vec2 v) {
    return min(v.x, v.y);
}

// Select the closest pack equivalents for Veloren's native material kinds.
// The complete 147-layer pack remains available in the arrays; Veloren does
// not store Minecraft's per-block identifier, so only kinds with a clear
// native counterpart are selected here.
int terrain_pbr_layer(uint kind, vec3 face_normal) {
    if (kind == 0x10u || kind == 0x13u) return 122; // stone
    if (kind == 0x11u || kind == 0x14u) return 16; // cobblestone
    if (kind == 0x20u) {
        return face_normal.z > 0.5 ? 58 : (face_normal.z < -0.5 ? 41 : 55);
    }
    if (kind == 0x30u) return 41; // dirt
    if (kind == 0x31u) return 109; // sand
    if (kind == 0x40u) return abs(face_normal.z) > 0.5 ? 81 : 80; // log end / bark
    if (kind == 0x21u || kind == 0x22u) return 116; // snow
    if (kind == 0x41u) return 79; // oak leaves
    if (kind == 0x43u) return 50; // glass-like ice surface
    return -1;
}

// LabPBR normal maps store tangent-space X/Y in R/G, AO in B,
// and height in A. Z is reconstructed because this pack uses the
// two-channel normal format.
vec3 decode_labpbr_normal(vec4 encoded, out float material_ao) {
    vec2 xy = encoded.rg * 2.0 - 1.0;
    float z = sqrt(max(1.0 - dot(xy, xy), 0.0));
    material_ao = encoded.b;
    return normalize(vec3(xy, z));
}

float sample_pbr_height(int layer, vec2 uv, vec2 uv_dx, vec2 uv_dy) {
    return textureGrad(
        sampler2DArray(t_pbr_normal, s_pbr_normal),
        vec3(fract(uv), float(layer)),
        uv_dx,
        uv_dy
    ).a;
}

// Fixed 32-layer POM march, with the pack's 0.25 block-relative depth,
// view-distance fade, blue-noise-style per-fragment phase, and a
// depth-crossing interpolation used for the final trace depth.
vec4 trace_pbr_parallax(
    int layer,
    vec2 source_uv,
    vec2 uv_dx,
    vec2 uv_dy,
    vec3 view_dir_ts,
    float fade
) {
    float first_height = sample_pbr_height(layer, source_uv, uv_dx, uv_dy);
    vec2 hit_uv = source_uv;
    float hit_height = first_height;
    float hit_trace_depth = 1.0;

    if (fade <= 0.0 || view_dir_ts.z >= 0.0 || first_height >= (254.0 / 255.0)) {
        return vec4(source_uv, first_height, 1.0);
    }

    const float POM_DEPTH = 0.25;
    const float POM_SAMPLES = 32.0;
    const float TRACE_STEP = 1.0 / POM_SAMPLES;
    vec2 ray_step = view_dir_ts.xy * POM_DEPTH
        / (1.0 + (-view_dir_ts.z) * POM_SAMPLES);

    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    hit_uv += ray_step * jitter;

    float previous_height = first_height;
    float previous_trace_depth = 1.0;
    int sample_count = max(1, int(POM_SAMPLES * fade));

    for (int i = 1; i <= 32; ++i) {
        if (i > sample_count) {
            break;
        }

        float trace_depth = 1.0 - float(i) * TRACE_STEP;
        float sampled_height = max(
            sample_pbr_height(layer, hit_uv, uv_dx, uv_dy),
            1.0 - fade
        );

        previous_trace_depth = 1.0 - float(i - 1) * TRACE_STEP;
        hit_trace_depth = trace_depth;
        hit_height = sampled_height;

        if (sampled_height >= trace_depth) {
            float crossing = (previous_trace_depth - previous_height)
                / max(sampled_height - previous_height
                    + previous_trace_depth - trace_depth, 0.00001);
            hit_trace_depth = previous_trace_depth - clamp(crossing, 0.0, 1.0) * TRACE_STEP;
            break;
        }

        previous_height = sampled_height;
        hit_uv += ray_step;
    }

    return vec4(hit_uv, hit_height, hit_trace_depth);
}

// Parallax self-shadowing uses the same height field as the camera ray.
// The caller folds this factor into the sun's existing shadow term so
// ambient and moon illumination are not incorrectly occluded.
float pbr_parallax_shadow(
    int layer,
    vec2 hit_uv,
    vec2 uv_dx,
    vec2 uv_dy,
    vec3 light_dir_ts,
    float surface_height,
    float depth_offset,
    float fade
) {
    float ndotl = dot(vec3(0.0, 0.0, 1.0), light_dir_ts);
    if (fade <= 0.01 || ndotl <= 0.01) {
        return 1.0;
    }

    light_dir_ts.xy *= 0.3;
    const float SHADOW_SAMPLES = 16.0;
    float step_size = 1.28 / SHADOW_SAMPLES;
    float jitter = fract(sin(dot(gl_FragCoord.yx, vec2(39.3468, 11.135))) * 24634.6345);
    vec3 start_offset = step_size * light_dir_ts * jitter;
    start_offset.z -= depth_offset;

    float visibility = 1.0;
    for (int i = 0; i < 16; ++i) {
        float ray_height = surface_height
            + light_dir_ts.z * step_size * float(i)
            + start_offset.z;
        if (ray_height > 1.0) {
            break;
        }

        vec2 shadow_uv = hit_uv
            + light_dir_ts.xy * (step_size * float(i))
            + start_offset.xy;
        float sampled_height = sample_pbr_height(layer, shadow_uv, uv_dx, uv_dy);
        visibility *= clamp(1.0 - (sampled_height - ray_height) * 40.0, 0.0, 1.0);
        if (visibility < 0.01) {
            break;
        }
    }

    return mix(1.0, visibility, fade);
}

void main() {
    // First 3 normals are negative, next 3 are positive
    const vec3 normals[8] = vec3[](vec3(-1,0,0), vec3(1,0,0), vec3(0,-1,0), vec3(0,1,0), vec3(0,0,-1), vec3(0,0,1), vec3(0,0,0), vec3(0,0,0));

    vec2 atlas_uv = f_uv_pos + vec2(atlas_offs.xy);
    float f_light, f_glow, f_ao, f_sky_exposure;
    uint f_kind;
    vec3 f_col = greedy_extract_col_light_kind_terrain(t_col_light, s_col_light, t_kind, atlas_uv, f_light, f_glow, f_ao, f_sky_exposure, f_kind);

    uint f_mat = MAT_BLOCK;

#ifdef EXPERIMENTAL_BAREMINIMUM
    tgt_color = vec4(simple_lighting(f_pos.xyz, f_col, f_light), 1);
#else

    vec3 face_norm = normals[(f_pos_norm >> 29) & 0x7u];
    vec3 f_norm = face_norm;

    #ifdef EXPERIMENTAL_BRICKLOREN
        vec3 pos = f_pos + focus_off.xyz;
        const vec3 bk_sz = vec3(2, 2, 2);
        vec3 sz = vec3(1.0 + mod(floor(pos.z * bk_sz.z + floor(pos.x) + floor(pos.y) - 0.01), 2.0) * (bk_sz.x - 1), 1.0 + mod(floor(pos.z * bk_sz.z + floor(pos.x) + floor(pos.y) + 0.99), 2.0) * (bk_sz.y - 1), bk_sz.z);
        vec3 fp = pos * sz;
        vec3 clamped = min(floor(fp.xyz) + 1.0 - 0.07 * sz, max(floor(fp.xyz) - 0.07 * sz, fp.xyz));
        f_norm.xyz += (fp.xyz - clamped) * 5.0 * sign(1.0 - f_norm) * max(1.0 - length(f_pos - cam_pos.xyz) / 64.0, 0);
        f_norm = normalize(f_norm);
        f_col /= 1.0 + length((fp - clamped) * sign(1.0 - f_norm)) * 2;
    #endif

    // Whether this face is facing fluid or not.
    bool faces_fluid = bool((f_pos_norm >> 28) & 0x1u);

    vec3 cam_to_frag = normalize(f_pos - cam_pos.xyz);
    vec3 view_dir = -cam_to_frag;

    int pbr_layer = faces_fluid ? -1 : terrain_pbr_layer(f_kind, face_norm);
    vec3 pbr_albedo = vec3(1.0);
    vec3 pbr_normal_ts = vec3(0.0, 0.0, 1.0);
    vec4 pbr_specular = vec4(1.0, 0.04, 0.0, 1.0);
    float pbr_material_ao = 1.0;
    float pbr_emission = 0.0;
    float pbr_shadow = 1.0;
    mat3 pbr_tbn = mat3(1.0);

    if (pbr_layer >= 0) {
        vec3 tangent = abs(face_norm.x) > 0.5
            ? vec3(0.0, 1.0, 0.0)
            : vec3(1.0, 0.0, 0.0);
        vec3 bitangent = normalize(cross(face_norm, tangent));
        pbr_tbn = mat3(tangent, bitangent, face_norm);

        vec3 world_pos = f_pos + focus_off.xyz;
        vec2 base_uv = vec2(dot(world_pos, tangent), dot(world_pos, bitangent));
        vec2 uv_dx = dFdx(base_uv);
        vec2 uv_dy = dFdy(base_uv);
        float view_distance = distance(cam_pos.xyz, f_pos);
        float parallax_fade = 1.0 - smoothstep(24.0, 48.0, view_distance);
        vec3 view_dir_ts = transpose(pbr_tbn) * cam_to_frag;

        // POM returns the parallax hit position, sampled height, and
        // interpolated trace depth. Texture derivatives remain those of
        // the original planar UV so the march does not introduce unstable
        // derivative discontinuities.
        vec4 parallax_hit = trace_pbr_parallax(
            pbr_layer, base_uv, uv_dx, uv_dy, view_dir_ts, parallax_fade
        );
        vec2 material_uv = parallax_hit.xy;
        vec3 layer_uv = vec3(fract(material_uv), float(pbr_layer));

        pbr_albedo = textureGrad(
            sampler2DArray(t_pbr_albedo, s_pbr_albedo), layer_uv, uv_dx, uv_dy
        ).rgb;
        vec4 normal_sample = textureGrad(
            sampler2DArray(t_pbr_normal, s_pbr_normal), layer_uv, uv_dx, uv_dy
        );
        pbr_normal_ts = decode_labpbr_normal(normal_sample, pbr_material_ao);
        pbr_specular = textureGrad(
            sampler2DArray(t_pbr_specular, s_pbr_specular), layer_uv, uv_dx, uv_dy
        );

        float depth_offset = max(parallax_hit.z - parallax_hit.w, 0.0);
        vec3 light_dir_ts = transpose(pbr_tbn) * (-sun_dir.xyz);
        pbr_shadow = pbr_parallax_shadow(
            pbr_layer,
            material_uv,
            uv_dx,
            uv_dy,
            light_dir_ts,
            normal_sample.a,
            depth_offset,
            parallax_fade
        );
    }

#if (SHADOW_MODE == SHADOW_MODE_CHEAP || SHADOW_MODE == SHADOW_MODE_MAP || FLUID_MODE >= FLUID_MODE_MEDIUM)
    float f_alt = alt_at(f_pos.xy);
#elif (SHADOW_MODE == SHADOW_MODE_NONE || FLUID_MODE == FLUID_MODE_LOW)
    float f_alt = f_pos.z;
#endif

    float alpha = 1.0;
    // TODO: Possibly angle with water surface into account?  Since we can basically assume it's horizontal.
    const float n2 = 1.5;
    const float R_s2s0 = pow(abs((1.0 - n2) / (1.0 + n2)), 2);
    const float R_s1s0 = pow(abs((1.3325 - n2) / (1.3325 + n2)), 2);
    const float R_s2s1 = pow(abs((1.0 - 1.3325) / (1.0 + 1.3325)), 2);
    const float R_s1s2 = pow(abs((1.3325 - 1.0) / (1.3325 + 1.0)), 2);
    float fluid_alt = max(f_pos.z + 1, floor(f_alt + 1));
    float R_s = faces_fluid ? mix(R_s2s1 * R_s1s0, R_s1s0, medium.x) : mix(R_s2s0, R_s1s2 * R_s2s0, medium.x);

    vec3 k_a = vec3(1.0);
    vec3 k_d = vec3(1.0);
    vec3 k_s = vec3(R_s);

    if (pbr_layer >= 0) {
        // LabPBR: red is inverted perceptual roughness; green is
        // dielectric F0 or the conductor/material code. Values >= 230
        // select the conductor path.
        float roughness = max(pow(1.0 - pbr_specular.r, 2.0), 0.0002);
        float encoded_f0 = floor(pbr_specular.g * 255.0 + 0.5);
        bool conductor = encoded_f0 >= 230.0;
        vec3 base_color = pbr_albedo * f_col;

        f_col = base_color;
        if (conductor) {
            // LabPBR conductor records (230-237) have a metal-specific
            // complex IOR. Use the corresponding normal-incidence F0
            // triplet in Veloren's existing Fresnel input.
            if (encoded_f0 <= 237.0) {
                int metal_id = clamp(int(encoded_f0) - 230, 0, 7);
                const vec3 eta_table[8] = vec3[8](
                    vec3(2.9114, 2.9497, 2.5845),
                    vec3(0.18299, 0.42108, 1.3734),
                    vec3(1.3456, 0.96521, 0.61722),
                    vec3(3.1071, 3.1812, 2.3230),
                    vec3(0.27105, 0.67693, 1.3164),
                    vec3(1.9100, 1.8300, 1.4400),
                    vec3(2.3757, 2.0847, 1.8453),
                    vec3(0.15943, 0.14512, 0.13547)
                );
                const vec3 etak_table[8] = vec3[8](
                    vec3(3.0893, 2.9318, 2.7670),
                    vec3(3.4242, 2.3459, 1.7704),
                    vec3(7.4746, 6.3995, 5.3031),
                    vec3(3.3314, 3.3291, 3.1350),
                    vec3(3.6092, 2.6248, 2.2921),
                    vec3(3.5100, 3.4000, 3.1800),
                    vec3(4.2655, 3.7153, 3.1365),
                    vec3(3.9291, 3.1900, 2.3808)
                );
                vec3 eta = eta_table[metal_id];
                vec3 etak = etak_table[metal_id];
                vec3 eta2 = eta * eta;
                vec3 etak2 = etak * etak;
                k_s = ((eta - 1.0) * (eta - 1.0) + etak2)
                    / ((eta + 1.0) * (eta + 1.0) + etak2);
            } else {
                // Non-complex conductors use the texture albedo as the
                // colored Fresnel reflectance, as specified by LabPBR.
                k_s = clamp(base_color, 0.0, 0.99);
            }
            k_d = vec3(0.0);
        } else {
            float f0 = clamp(pbr_specular.g, 0.02, 0.9);
            k_d = base_color;
            k_s = vec3(f0);
        }

        alpha = roughness;
        f_norm = normalize(pbr_tbn * pbr_normal_ts);
        float emission_code = pbr_specular.a * 255.0;
        pbr_emission = emission_code <= 254.5
            ? clamp((emission_code - 0.5) / 254.0, 0.0, 1.0)
            : 0.0;
    }

    float f_alpha = 1.0;
    #ifdef RAIN_ENABLED
        #if (REFLECTION_MODE >= REFLECTION_MODE_MEDIUM)
            if (rain_density > 0 && !faces_fluid && f_norm.z > 0.5) {
                vec3 pos = f_pos + focus_off.xyz;
                vec3 drop_density = vec3(2, 2, 2);
                vec3 drop_pos = pos + vec3(pos.zz, 0) + vec3(0, 0, -tick.x * 1.0);
                drop_pos.z += noise_2d(floor(drop_pos.xy * drop_density.xy) * 13.1) * 10;
                vec2 cell2d = floor(drop_pos.xy * drop_density.xy);
                drop_pos.z *= 0.5 + hash_fast(uvec3(cell2d, 0));
                vec3 cell = vec3(cell2d, floor(drop_pos.z * drop_density.z));

                #if (REFLECTION_MODE >= REFLECTION_MODE_HIGH)
                    float puddle = clamp((noise_2d((f_pos.xy + focus_off.xy + vec2(0.1, 0)) * 0.02) - 0.5) * 20.0, 0.0, 1.0)
                        * min(rain_density * 10.0, 1.0)
                        * clamp((f_sky_exposure - 0.95) * 50.0, 0.0, 1.0);
                #else
                    const float puddle = 1.0;
                #endif

                #if (REFLECTION_MODE >= REFLECTION_MODE_HIGH)
                    if (puddle > 0.0) {
                        f_alpha = puddle * 0.2 * max(1.0 + cam_to_frag.z, 0.3);
                        #ifdef EXPERIMENTAL_PUDDLEDETAILS
                            float t0 = sin(tick_loop(2.0 * PI, 8.0, f_pos.x * 3));
                            float t1 = sin(tick_loop(2.0 * PI, 3.5, -f_pos.x * 6));
                            float h = (noise_2d((f_pos.xy + focus_off.xy) * 0.3) - 0.5) * t0
                                + (noise_2d((f_pos.xy + focus_off.xy) * 0.6) - 0.5) * t1;
                            float hx = (noise_2d((f_pos.xy + focus_off.xy + vec2(0.1, 0)) * 0.3) - 0.5) * t0
                                + (noise_2d((f_pos.xy + focus_off.xy + vec2(0.1, 0)) * 0.6) - 0.5) * t1;
                            float hy = (noise_2d((f_pos.xy + focus_off.xy + vec2(0, 0.1)) * 0.3) - 0.5) * t0
                                + (noise_2d((f_pos.xy + focus_off.xy + vec2(0, 0.1)) * 0.6) - 0.5) * t1;
                            f_norm.xy += mix(vec2(0), vec2(h - hx, h - hy) / 0.1 * 0.03, puddle);
                        #endif
                        alpha = mix(1.0, 0.2, puddle);
                        f_col.rgb *= mix(1.0, 0.7, puddle);
                        k_s = mix(k_s, vec3(0.7, 0.7, 1.0), puddle);
                        f_mat = MAT_PUDDLE;
                    }
                #endif

                if (rain_occlusion_at(f_pos.xyz + vec3(0, 0, 0.25)) > 0.5) {
                    if (fract(hash(fract(vec4(cell, 0) * 0.01))) < rain_density * 2.0) {
                        vec3 off = vec3(hash_fast(uvec3(cell * 13)), hash_fast(uvec3(cell * 5)), 0);
                        vec3 near_cell = (cell + 0.5 + (off - 0.5) * 0.5) / drop_density;

                        float dist = length((drop_pos - near_cell) * vec3(1, 1, 0.5));
                        float drop_rad = 0.075 + puddle * 0.05;
                        float distort = max(1.0 - abs(dist - drop_rad) * 100, 0) * 1.5 * max(drop_pos.z - near_cell.z, 0);
                        k_a += distort;
                        k_d += distort;
                        k_s += distort;

                        f_norm.xy += (drop_pos - near_cell).xy
                            * max(1.0 - abs(dist - drop_rad) * 30, 0)
                            * 500.0
                            * max(drop_pos.z - near_cell.z, 0)
                            * sign(dist - drop_rad)
                            * max(drop_pos.z - near_cell.z, 0);
                    }
                }
            }
        #endif
    #endif

    #if (REFLECTION_MODE >= REFLECTION_MODE_MEDIUM)
    // Reflections on ice
    if (f_kind == BLOCK_ICE && f_norm.z == 1.0) {
        f_alpha = min(f_alpha, 0.3);
        k_s = mix(k_s, vec3(0.7, 0.7, 1.0), 0.5);
    }
    #endif

#if (SHADOW_MODE == SHADOW_MODE_CHEAP || SHADOW_MODE == SHADOW_MODE_MAP)
    vec4 f_shadow = textureMaybeBicubic(t_horizon, s_horizon, pos_to_tex(f_pos.xy));
    float sun_shade_frac = horizon_at2(f_shadow, f_alt, f_pos, sun_dir);
#elif (SHADOW_MODE == SHADOW_MODE_NONE)
    float sun_shade_frac = 1.0;
#endif
    float moon_shade_frac = 1.0;

    DirectionalLight sun_info = get_sun_info(sun_dir, sun_shade_frac, /*sun_pos*/f_pos);
    // Only the directional sun contribution receives the height-field
    // self-shadow; sky fill and moonlight retain their existing visibility.
    sun_info.shadow *= pbr_shadow;
    DirectionalLight moon_info = get_moon_info(moon_dir, moon_shade_frac/*, light_pos*/);

    #ifdef EXPERIMENTAL_DIRECTIONALSHADOWMAPTEXELGRID
        float offset_scale = 0.5;
        vec3 offset_one = dFdx(f_pos) * offset_scale;
        vec3 offset_two = dFdy(f_pos) * offset_scale;
        vec3 one_up = f_pos + offset_one;
        vec3 one_down = f_pos - offset_one;
        vec3 two_up = f_pos + offset_two;
        vec3 two_down = f_pos - offset_two;

        // Adjust this to change the size of the grid cells relative to the
        // number of shadow texels
        float grid_cell_to_texel_ratio = 32.0;

        vec2 shadowTexSize = textureSize(sampler2DShadow(t_directed_shadow_maps, s_directed_shadow_maps), 0) / grid_cell_to_texel_ratio;

        vec4 one_up_shadow_tex = texture_mat * vec4(one_up, 1.0);
        vec2 oust_snap = floor(one_up_shadow_tex.xy * shadowTexSize / one_up_shadow_tex.w);
        vec4 one_down_shadow_tex = texture_mat * vec4(one_down, 1.0);
        vec2 odst_snap = floor(one_down_shadow_tex.xy * shadowTexSize / one_down_shadow_tex.w);
        vec4 two_up_shadow_tex = texture_mat * vec4(two_up, 1.0);
        vec2 tust_snap = floor(two_up_shadow_tex.xy * shadowTexSize / two_up_shadow_tex.w);
        vec4 two_down_shadow_tex = texture_mat * vec4(two_down, 1.0);
        vec2 tdst_snap = floor(two_down_shadow_tex.xy * shadowTexSize / two_down_shadow_tex.w);
        float border = length(max(abs(oust_snap - odst_snap), abs(tust_snap - tdst_snap)));

        if (border != 0.0) {
            tgt_color = vec4(vec3(0.0, 0.7, 0.2), 1.0);
            return;
        }
    #endif

    float max_light = 0.0;

    // Compute attenuation due to water from the camera.
    vec3 mu = faces_fluid ? MU_WATER : vec3(0.0);
    // NOTE: Default intersection point is camera position, meaning if we fail to intersect we assume the whole camera is in water.
    // Computing light attenuation from water.
    vec3 cam_attenuation = compute_attenuation_point(f_pos, -view_dir, mu, fluid_alt, cam_pos.xyz);

    // Prevent the sky affecting light when underground
    float not_underground = clamp((f_pos.z - f_alt) / 128.0 + 1.0, 0.0, 1.0);

    // To account for prior saturation
    #if (FLUID_MODE == FLUID_MODE_LOW)
        f_light = f_light * sqrt(f_light);
    #else
        f_light = faces_fluid ? not_underground : f_light * sqrt(f_light);
    #endif

    vec3 emitted_light = vec3(1.0);
    vec3 reflected_light = vec3(1.0);

    float sun_diffuse = get_sun_diffuse2(sun_info, moon_info, f_norm, view_dir, f_pos, mu, cam_attenuation, fluid_alt, k_a/* * (shade_frac * 0.5 + light_frac * 0.5)*/, k_d, k_s, alpha, f_norm, 1.0, emitted_light, reflected_light);
    max_light += sun_diffuse;

    emitted_light *= f_light;
    reflected_light *= f_light;
    max_light *= f_light;

    // TODO: Apply AO after this
    vec3 glow = glow_light(f_pos) * (pow(f_glow, 3) * 5 + pow(f_glow, 2.0) * 2) * pow(max(dot(face_norm, f_norm), 0), 2);
    reflected_light += glow * cam_attenuation;

    max_light += lights_at(f_pos, f_norm, view_dir, mu, cam_attenuation, fluid_alt, k_a, k_d, k_s, alpha, f_norm, 1.0, emitted_light, reflected_light);

    emitted_light *= mix(1.0, f_ao, 0.5) * pbr_material_ao;
    reflected_light *= mix(1.0, f_ao, 0.5) * pbr_material_ao;
    emitted_light += pbr_albedo * pbr_emission;

    float point_shadow = shadow_at(f_pos, f_norm);
    reflected_light *= point_shadow;
    emitted_light *= point_shadow;

    #ifndef EXPERIMENTAL_NOCAUSTICS
        #if (FLUID_MODE >= FLUID_MODE_MEDIUM)
            if (faces_fluid) {
                vec3 wpos = f_pos + vec3(focus_off.xy, 0);
                vec3 spos = (wpos + (fluid_alt - wpos.z) * vec3(sun_dir.xy, 0)) * 0.25;
                reflected_light += caustics(spos.xy * 1.0, tick.x * 0.5)
                    * 3
                    / (1.0 + pow(abs(fluid_alt - wpos.z) * 0.075, 2))
                    * cam_attenuation
                    * max(dot(f_norm, -sun_dir.xyz), 0)
                    * sun_diffuse
                    * sun_info.shadow
                    * f_light;
            }
        #endif
    #endif

    vec3 f_chunk_pos = f_pos - (model_mat[3].xyz - focus_off.xyz);
    #ifdef EXPERIMENTAL_NONOISE
        float noise = 0.0;
    #else
        #ifdef EXPERIMENTAL_BRICKLOREN
            float noise = hash(vec4(floor(clamped), 0)) * 2 + hash(vec4(floor(clamped * 27 / sz), 0)) * 0.5;
        #else
            float noise = hash(vec4(floor(f_chunk_pos * 3.0 - f_norm * 0.5), 0));//0.005/* - 0.01*/;
        #endif
    #endif

    const float A = 0.055;
    const float W_INV = 1 / (1 + A);
    const float W_2 = W_INV * W_INV;
    const float NOISE_FACTOR = 0.015;
    vec3 noise_delta = (sqrt(f_col) * W_INV + noise * NOISE_FACTOR);
    vec3 col = noise_delta * noise_delta * W_2;
    vec3 surf_color = illuminate(max_light, view_dir, col * emitted_light, col * reflected_light);
    #ifdef EXPERIMENTAL_SNOWGLITTER
    if (f_kind == BLOCK_SNOW || f_kind == BLOCK_ART_SNOW) {
        float cam_distance = distance(cam_pos.xyz, f_pos);
        vec3 pos = f_pos + focus_off.xyz;

        float map = max(noise_3d(pos), 0.0);

        vec4 lpos = vec4(floor(pos * 35.0), 0.0);

        vec3 n = normalize(vec3(hash(lpos + 128), hash(lpos - 435), hash(lpos + 982)));

        float s = pow(abs(dot(n, view_dir)), 4.0);

        surf_color += pow(map * s, 10.0) * 5.0 / max(1.0, cam_distance * 0.5);
    }
    #endif

    float f_select = (select_pos.w > 0 && select_pos.xyz == floor(f_pos - f_norm * 0.5)) ? 1.0 : 0.0;
    surf_color += f_select * (surf_color + 0.1) * vec3(0.5, 0.5, 0.5);
    
    #ifdef EXPERIMENTAL_SHOWCHUNKBORDERS
    float border_scale = 0.0001 * distance(cam_pos.xyz, f_pos);
    if (vmin(fract((f_pos.xy + focus_off.xy) / 32.0 + 1024) - border_scale * 0.5) < border_scale && f_norm.z > 0.5) {
        surf_color = vec3(1.0, 0.0, 0.0);
    }
    #endif

    tgt_color = vec4(surf_color, f_alpha);
    tgt_mat = uvec4(uvec3((f_norm + 1.0) * 127.0), f_mat);
#endif
}
