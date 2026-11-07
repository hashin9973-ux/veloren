# Parallax PBR block materials

This directory stores the static material maps from the pack's
`assets/minecraft/textures/block` directory, converted to Veloren's asset
naming convention.

The complete block-folder inventory has **147 static materials** and
**441 RGBA PNGs**:

- `<material>_albedo.png`: sRGB base color
- `<material>_normal.png`: linear tangent-space normal data; blue is ambient
  occlusion and alpha is the parallax height field
- `<material>_specular.png`: linear LabPBR material data

Animated fire sheets are excluded because they have no normal/specular map
triplet. Files outside the block texture directory (entity, armor, boat,
painting, and other non-block textures) are also excluded.

All 147 layers are loaded into matching 2D texture arrays. The layer order is
the alphabetical order in `pbr_material_names` in
`voxygen/src/render/renderer/mod.rs`; the corresponding native block-kind
selection is in `terrain_pbr_layer` in
`assets/voxygen/shaders/terrain-frag.glsl`.

## Renderer mapping constraint

Veloren's current `Block` representation stores a `BlockKind` and an RGB
color, not a Minecraft block identifier for every one of these 147 names.
This integration therefore preserves the existing world/save/network block
format and maps the native terrain kinds to the closest supported material
layers (rock, weak rock, grass faces, earth, sand, wood faces, snow, leaves,
and ice-like surfaces). The other layers are present and GPU-uploaded, but
cannot be selected individually until Veloren gains per-block material IDs or
equivalent explicit material metadata.

The shader ports the pack's POM behavior using the normal-map alpha height
field, the 0.25 depth scale, 32 view-ray samples, 16 self-shadow samples, and
the 24–48 world-unit distance fade. LabPBR roughness/F0 and conductor codes,
normal reconstruction, and AO are decoded in the terrain fragment path.
