
struct F32Vec2Array {
  val: array<vec2<f32>>,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_samp: sampler;

@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<storage, read> positions: F32Vec2Array;
@group(0) @binding(4) var<storage, read> rotations: F32Vec2Array;

fn convolution_1d(start: vec2<f32>, end: vec2<f32>, num_sample: u32) -> f32 {
  var result = 0.0;
  let normalize_factor = 1.0 / f32(num_sample);
  for (var i = 0; u32(i) < num_sample; i += 1) {
    let coord_ratio = (0.5 + f32(i)) / f32(num_sample);
    let kernel_index = i32( floor(coord_ratio * f32(arrayLength(&kernel))) );
    let sample_coord = start + (end - start) * coord_ratio;
    let tex_val = (textureSample(tex, tex_samp, sample_coord).r - 0.5) * 2.0;
    let ker_val = kernel[kernel_index];
    result += tex_val * ker_val;
  }
  return result * normalize_factor;
}


@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let resolution: vec2<f32> = vec2<f32>(textureDimensions(tex));
  let i = u32(floor(in.position.y));
  let j = u32(floor(in.position.x));
  let point_position_px = positions.val[i];
  let point_rotation_px = rotations.val[j];

  let center = (point_position_px + point_rotation_px) / 2.0;
  let start_px: vec2<f32> = center + (point_position_px - center) * 2.0;
  let end_px :  vec2<f32> = center + (point_rotation_px - center) * 2.0;
  let num_sample: u32 = u32(floor(distance( start_px, end_px )) + 1.0);

  let start_uv: vec2<f32> = start_px / resolution;
  let end_uv: vec2<f32> = end_px / resolution;

  let res = convolution_1d(start_uv, end_uv, num_sample);
  return vec4<f32>(res, res, res, 1.0);
}
