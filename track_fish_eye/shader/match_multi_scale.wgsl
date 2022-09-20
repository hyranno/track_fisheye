@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_samp: sampler;
@group(0) @binding(2) var<uniform> resolution: Vec2Container_u32;
@group(0) @binding(3) var kernel: texture_1d<f32>;
@group(0) @binding(4) var kernel_samp: sampler;
@group(0) @binding(5) var<uniform> range: Vec2Container_f32;
@group(0) @binding(6) var<uniform> threshold: F32Container;
@group(0) @binding(7) var<uniform> scale_to: F32Container;
@group(0) @binding(8) var<uniform> scale_step: F32Container;

fn convolution_1d(start: vec2<f32>, end: vec2<f32>, num_sample: u32) -> f32 {
  var result = 0.0;
  let normalize_factor = 1.0 / f32(num_sample);
  for (var i = 0; u32(i) < num_sample; i += 1) {
    let kernel_coord = (0.5 + f32(i)) / f32(num_sample);
    let sample_coord = start + (end - start) * kernel_coord;
    let tex_val = (textureSample(tex, tex_samp, sample_coord).r - 0.5) * 2.0;
    let ker_val = textureSample(kernel, kernel_samp, kernel_coord).r;
    result += tex_val * ker_val;
  }
  return result * normalize_factor;
}

fn match_scale(scale: f32, uv: vec2<f32>) -> i32 {
  let half_range: vec2<f32> = scale * range.val / 2.0;
  let start: vec2<f32> = uv - half_range;
  let end: vec2<f32> = uv + half_range;
  let start_px: vec2<f32> = start * vec2<f32>(resolution.val);
  let end_px: vec2<f32> = end * vec2<f32>(resolution.val);
  let num_sample: u32 = u32(floor(distance( start_px, end_px )) + 1.0);
  let val = convolution_1d(start, end, num_sample);
  var res: i32 = 0;
  if (val < -threshold.val) {
    res = -1;
  }
  if (threshold.val < val) {
    res = 1;
  }
  return res;
}

/*
  0: match_negative,
  0.5: match_nothing,
  1: match_positive
*/
@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  var matching: i32 = 0;
  var scale = 1.0;
  for (var scale = 1.0; scale < scale_to.val; scale *= scale_step.val) {
    matching += match_scale(scale, in.uv);
  }
  if (matching != 0) {
    matching /= abs(matching);
  }
  var res = f32(matching + 1) * 0.5;
  return vec4<f32>(res, res, res, 1.0);
}
