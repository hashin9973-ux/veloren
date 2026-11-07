use super::{
    super::{
        Bound, Consts, ExperimentalShader, GlobalsLayouts, PipelineModes, Vertex as VertexTrait,
        texture::Texture,
    },
    AtlasData,
};
use bytemuck::{Pod, Zeroable};
use common::figure::CellSurface;
use std::mem;
use vek::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
pub struct Vertex {
    pos_norm: u32,
    atlas_pos: u32,
}

impl Vertex {
    /// NOTE: meta is true when the terrain vertex is touching water.
    pub fn new(atlas_pos: Vec2<u16>, pos: Vec3<f32>, norm: Vec3<f32>, meta: bool) -> Self {
        const EXTRA_NEG_Z: f32 = 32768.0;

        #[expect(clippy::bool_to_int_with_if)]
        let norm_bits = if norm.x != 0.0 {
            if norm.x < 0.0 { 0 } else { 1 }
        } else if norm.y != 0.0 {
            if norm.y < 0.0 { 2 } else { 3 }
        } else if norm.z < 0.0 {
            4
        } else {
            5
        };
        Self {
            pos_norm: (((pos.x as u32) & 0x003F) << 0)
                | (((pos.y as u32) & 0x003F) << 6)
                | ((((pos + EXTRA_NEG_Z).z.clamp(0.0, (1 << 16) as f32) as u32) & 0xFFFF) << 12)
                | (u32::from(meta) << 28)
                | ((norm_bits & 0x7) << 29),
            atlas_pos: (((atlas_pos.x as u32) & 0xFFFF) << 0)
                | (((atlas_pos.y as u32) & 0xFFFF) << 16),
        }
    }

    pub fn new_figure(atlas_pos: Vec2<u16>, pos: Vec3<f32>, norm: Vec3<f32>, bone_idx: u8) -> Self {
        let norm_bits = u32::from(norm.x.min(norm.y).min(norm.z) >= 0.0);
        let axis_bits = if norm.x != 0.0 {
            0
        } else if norm.y != 0.0 {
            1
        } else {
            2
        };
        Self {
            pos_norm: pos
                .map2(Vec3::new(0, 9, 18), |e, shift| {
                    (((e * 2.0 + 256.0) as u32) & 0x1FF) << shift
                })
                .reduce_bitor()
                | (((bone_idx & 0xF) as u32) << 27)
                | (norm_bits << 31),
            atlas_pos: (((atlas_pos.x as u32) & 0x7FFF) << 2)
                | (((atlas_pos.y as u32) & 0x7FFF) << 17)
                | axis_bits & 3,
        }
    }

    pub fn make_col_light(
        // 0 to 31
        light: u8,
        // 0 to 31
        glow: u8,
        col: Rgb<u8>,
        ao: bool,
    ) -> [u8; 4] {
        //[col.r, col.g, col.b, light]
        // It would be nice for this to be cleaner, but we want to squeeze 5 fields into
        // 4. We can do this because both `light` and `glow` go from 0 to 31,
        // meaning that they can both fit into 5 bits. If we steal a bit from
        // red and blue each (not green, human eyes are more sensitive to
        // changes in green) then we get just enough to expand the nibbles of
        // the alpha field enough to fit both `light` and `glow`.
        //
        // However, we now have a problem. In the shader code with use hardware
        // filtering to get at the `light` and `glow` attributes (but not
        // colour, that remains constant across a block). How do we resolve this
        // if we're twiddling bits? The answer is to very carefully manipulate
        // the bit pattern such that the fields we want to filter (`light` and
        // `glow`) always sit as the higher bits of the fields. Then, we can do
        // some modulation magic to extract them from the filtering unharmed and use
        // unfiltered texture access (i.e: `texelFetch`) to access the colours, plus a
        // little bit-fiddling.
        //
        // TODO: This isn't currently working (no idea why). See `srgb.glsl` for current
        // impl that intead does manual bit-twiddling and filtering.
        [
            (light.min(31) << 3) | ((col.r >> 1) & 0b111),
            (glow.min(31) << 3) | ((col.b >> 1) & 0b111),
            (col.r & 0b11110000) | (col.b >> 4),
            (col.g & 0xFE) | ao as u8,
        ]
    }

    pub fn make_col_light_figure(
        // 0 to 31
        light: u8,
        col: Rgb<u8>,
        surf: CellSurface,
    ) -> [u8; 4] {
        debug_assert!((surf as u8) < 32);
        [
            (light.min(31) << 3) | ((col.r >> 1) & 0b111),
            ((surf as u8) << 3) | ((col.b >> 1) & 0b111),
            (col.r & 0b11110000) | (col.b >> 4),
            col.g, // Green is lucky, it remains unscathed
        ]
    }

    /// Set the bone_idx for an existing figure vertex.
    pub fn set_bone_idx(&mut self, bone_idx: u8) {
        self.pos_norm = (self.pos_norm & !(0xF << 27)) | ((bone_idx as u32 & 0xF) << 27);
    }

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Uint32,1 => Uint32];
        wgpu::VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

impl VertexTrait for Vertex {
    // Note: I think it's u32 due to figures??
    // potentiall optimize by splitting
    const QUADS_INDEX: Option<wgpu::IndexFormat> = Some(wgpu::IndexFormat::Uint32);
    const STRIDE: wgpu::BufferAddress = mem::size_of::<Self>() as wgpu::BufferAddress;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
// TODO: new function and private fields??
pub struct Locals {
    model_mat: [f32; 16],
    atlas_offs: [i32; 4],
    load_time: f32,
    _dummy: [f32; 3],
}

impl Locals {
    pub fn new(
        model_offs: Vec3<f32>,
        ori: Quaternion<f32>,
        atlas_offs: Vec2<u32>,
        load_time: f32,
    ) -> Self {
        let mat = Mat4::from(ori).translated_3d(model_offs);
        Self {
            model_mat: mat.into_col_array(),
            load_time,
            atlas_offs: Vec4::new(atlas_offs.x as i32, atlas_offs.y as i32, 0, 0).into_array(),
            _dummy: [0.0; 3],
        }
    }
}

impl Default for Locals {
    fn default() -> Self {
        Self {
            model_mat: Mat4::identity().into_col_array(),
            load_time: 0.0,
            atlas_offs: [0; 4],
            _dummy: [0.0; 3],
        }
    }
}

pub type BoundLocals = Bound<Consts<Locals>>;

pub struct TerrainLayout {
    pub locals: wgpu::BindGroupLayout,
}

impl TerrainLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            locals: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Terrain locals layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }),
        }
    }

    pub fn bind_locals(&self, device: &wgpu::Device, locals: Consts<Locals>) -> BoundLocals {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.locals,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: locals.buf().as_entire_binding(),
            }],
        });

        BoundLocals {
            bind_group,
            with: locals,
        }
    }
}

/// GPU-resident diffuse, normal, and LabPBR specular texture arrays used by
/// the terrain fragment shader. Each array has matching material layer indices.
pub struct TerrainPbrTextures {
    pub(super) albedo: Texture,
    pub(super) normal: Texture,
    pub(super) specular: Texture,
}

impl TerrainPbrTextures {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dimensions: Vec2<u32>,
        albedo_layers: &[Vec<u8>],
        normal_layers: &[Vec<u8>],
        specular_layers: &[Vec<u8>],
    ) -> Self {
        assert!(
            !albedo_layers.is_empty(),
            "PBR material arrays cannot be empty"
        );
        assert_eq!(albedo_layers.len(), normal_layers.len());
        assert_eq!(albedo_layers.len(), specular_layers.len());

        let create_array =
            |label: &'static str, format: wgpu::TextureFormat, layers: &[Vec<u8>]| {
                let layer_bytes = dimensions.x as usize * dimensions.y as usize * 4;
                assert!(
                    layers.iter().all(|layer| layer.len() == layer_bytes),
                    "{label} PBR layers must all be RGBA8 images with matching dimensions"
                );
                let mut data = Vec::with_capacity(layer_bytes * layers.len());
                for layer in layers {
                    data.extend_from_slice(layer);
                }

                let texture_info = wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: dimensions.x,
                        height: dimensions.y,
                        depth_or_array_layers: layers.len() as u32,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                };
                let view_info = wgpu::TextureViewDescriptor {
                    label: Some(label),
                    format: Some(format),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    usage: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                };
                let sampler_info = wgpu::SamplerDescriptor {
                    label: Some(label),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                };
                let texture = Texture::new_raw(device, &texture_info, &view_info, &sampler_info);
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(dimensions.x * 4),
                        rows_per_image: Some(dimensions.y),
                    },
                    texture_info.size,
                );
                texture
            };

        let albedo = create_array(
            "Terrain PBR albedo array",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            albedo_layers,
        );
        let normal = create_array(
            "Terrain PBR normal array",
            wgpu::TextureFormat::Rgba8Unorm,
            normal_layers,
        );
        let specular = create_array(
            "Terrain PBR specular array",
            wgpu::TextureFormat::Rgba8Unorm,
            specular_layers,
        );

        Self {
            albedo,
            normal,
            specular,
        }
    }
}

pub struct TerrainPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

impl TerrainPipeline {
    pub fn new(
        device: &wgpu::Device,
        vs_module: &wgpu::ShaderModule,
        fs_module: &wgpu::ShaderModule,
        global_layout: &GlobalsLayouts,
        layout: &TerrainLayout,
        format: wgpu::TextureFormat,
        pipeline_modes: &PipelineModes,
    ) -> Self {
        common_base::span!(_guard, "TerrainPipeline::new");
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Terrain pipeline layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[
                    &global_layout.globals,
                    &global_layout.shadow_textures,
                    global_layout.terrain_atlas_layout.layout(),
                    &layout.locals,
                ],
            });

        let samples = pipeline_modes.aa.samples();

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: vs_module,
                entry_point: Some("main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: if pipeline_modes
                    .experimental_shaders
                    .contains(&ExperimentalShader::Wireframe)
                {
                    wgpu::PolygonMode::Line
                } else {
                    wgpu::PolygonMode::Fill
                },
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState::IGNORE,
                    back: wgpu::StencilFaceState::IGNORE,
                    read_mask: !0,
                    write_mask: 0,
                },
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: fs_module,
                entry_point: Some("main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline: render_pipeline,
        }
    }
}

/// Represents texture that can be converted into texture atlases for terrain.
pub struct TerrainAtlasData {
    pub col_lights: Vec<[u8; 4]>,
    pub kinds: Vec<u8>,
}

impl AtlasData for TerrainAtlasData {
    type SliceMut<'a> =
        std::iter::Zip<std::slice::IterMut<'a, [u8; 4]>, std::slice::IterMut<'a, u8>>;

    const TEXTURES: usize = 2;

    fn blank_with_size(sz: Vec2<u16>) -> Self {
        let col_lights =
            vec![Vertex::make_col_light(254, 0, Rgb::broadcast(254), true); sz.as_().product()];
        let kinds = vec![0; sz.as_().product()];
        Self { col_lights, kinds }
    }

    fn as_texture_data(&self) -> [(wgpu::TextureFormat, &[u8]); Self::TEXTURES] {
        [
            (
                wgpu::TextureFormat::Rgba8Unorm,
                bytemuck::cast_slice(&self.col_lights),
            ),
            (wgpu::TextureFormat::R8Uint, &self.kinds),
        ]
    }

    fn layout() -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            // col lights
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // kind
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            // Static PBR material arrays share this terrain atlas bind group so
            // the terrain pipeline stays within WebGPU's four-group minimum.
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ]
    }

    fn slice_mut(&mut self, range: std::ops::Range<usize>) -> Self::SliceMut<'_> {
        self.col_lights[range.clone()]
            .iter_mut()
            .zip(self.kinds[range].iter_mut())
    }
}
