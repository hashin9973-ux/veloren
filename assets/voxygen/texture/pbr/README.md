# Terrain PBR materials

This directory contains the initial terrain-only material set used by Voxygen's
terrain shader. The diffuse, normal, and specular maps are loaded as matching
2D-array layers. Keep the three files for a material together and preserve the
normal-map alpha channel: it stores the height field used by parallax occlusion
mapping (POM).

All source maps in this first set are 64x64 RGBA PNGs.

## Layer order

The wood material uses the bark map on vertical faces and the top/end-grain maps on horizontal faces. The order below must match `pbr_material_names` in
`voxygen/src/render/renderer/mod.rs` and `terrain_pbr_layer` in
`assets/voxygen/shaders/terrain-frag.glsl`.

| Layer | Material | Veloren block kinds |
| ---: | --- | --- |
| 0 | stone | Rock, GlowingRock |
| 1 | cobblestone | WeakRock, GlowingWeakRock |
| 2 | grass_block_top | Grass top faces |
| 3 | dirt | Earth |
| 4 | sand | Sand |
| 5 | oak_log | Wood side faces |
| 6 | snow | Snow, ArtSnow |
| 7 | grass_block_side | Grass side faces |
| 8 | oak_log_top | Wood end faces |

Files use the suffixes `_albedo.png`, `_normal.png`, and `_specular.png`.

## Material handling

- The normal map is sampled in tangent space; its alpha channel is the POM
  height field. Do not strip or premultiply that alpha.
- The specular map is read as linear data using the LabPBR channel conventions:
  red provides inverted roughness and green provides F0/conductor identification.
- Diffuse maps use an sRGB GPU format. Normal and specular maps use linear UNORM.
- POM is faded with distance and uses world-aligned planar UVs so adjacent terrain
  faces do not depend on mesh atlas placement.

This initial set is deliberately limited to opaque terrain block kinds. Leaves,
ArtLeaves, plants, flowers, sprites, fluids, and miscellaneous decorative blocks
are not mapped and continue through Veloren's existing material path.
