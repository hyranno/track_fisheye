struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uvw: vec3<f32>,
};
struct Vec2Container_f32 {
  val: vec2<f32>,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> direction: Vec2Container_f32;
@group(0) @binding(4) var<storage, read> scales_px: array<u32>;

@stage(vertex)
fn vs_main(in: VertexInput) -> VertexOutput {
    var positions = array<vec3<f32>, 8>(
      vec3<f32>(-1.0, -1.0, -1.0),
      vec3<f32>(-1.0, -1.0, 1.0),
      vec3<f32>(-1.0, 1.0, -1.0),
      vec3<f32>(-1.0, 1.0, 1.0),
      vec3<f32>(1.0, -1.0, -1.0),
      vec3<f32>(1.0, -1.0, 1.0),
      vec3<f32>(1.0, 1.0, -1.0),
      vec3<f32>(1.0, 1.0, 1.0),
    );
    let index = i32(in.vertex_index);
    let p: vec3<f32> = positions[index];

    var out: VertexOutput;
    out.position = vec4<f32>(p, 1.0);
    out.uvw = (vec3<f32>(p.x, -p.y, p.z) + 1.0) / 2.0;
    return out;
}


fn convolution_1d(start: vec2<f32>, end: vec2<f32>, num_sample: u32) -> f32 {
  var result = 0.0;
  let normalize_factor = 1.0 / f32(num_sample);
  for (var i = 0; u32(i) < num_sample; i += 1) {
    let coord_ratio = (0.5 + f32(i)) / f32(num_sample);
    let kernel_index = i32( floor(coord_ratio * f32(arrayLength(&kernel))) );
    let sample_coord = start + (end - start) * coord_ratio;
    let tex_val = (textureSample(tex, samp, sample_coord).r - 0.5) * 2.0;
    let ker_val = kernel[kernel_index];
    result += tex_val * ker_val;
  }
  return result * normalize_factor;
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let scale_px = scales_px[i32(in.position.z)];
  let scale_uv = f32(scale_px) / vec2<f32>(textureDimensions(tex));
  let half_range = direction.val * scale_uv / 2.0;
  let start: vec2<f32> = in.uvw.xy - half_range;
  let end: vec2<f32> = in.uvw.xy + half_range;

  let res = convolution_1d(start, end, scale_px);
  return vec4<f32>(res, res, res, 1.0);
}
