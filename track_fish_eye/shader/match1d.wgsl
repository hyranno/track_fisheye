@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> range: Vec2Container_f32;

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
  let half_range: vec2<f32> = range.val / 2.0;
  let start: vec2<f32> = in.uv - half_range;
  let end: vec2<f32> = in.uv + half_range;
  let start_px: vec2<f32> = start * vec2<f32>(textureDimensions(tex));
  let end_px: vec2<f32> = end * vec2<f32>(textureDimensions(tex));
  let num_sample: u32 = u32(floor(distance( start_px, end_px )) + 1.0);

  let res = convolution_1d(start, end, num_sample);
  return vec4<f32>(res, res, res, 1.0);
}
