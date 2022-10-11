@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_samp: sampler;

@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> threshold: F32Container;
@group(0) @binding(4) var<uniform> range: Vec2Container_f32;
@group(0) @binding(5) var<uniform> scale_to: F32Container;
@group(0) @binding(6) var<uniform> scale_step: F32Container;


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

fn match_scale(scale: f32, uv: vec2<f32>) -> f32 {
  let half_range: vec2<f32> = scale * range.val / 2.0;
  let start: vec2<f32> = uv - half_range;
  let end: vec2<f32> = uv + half_range;
  let start_px: vec2<f32> = start * vec2<f32>(textureDimensions(tex));
  let end_px: vec2<f32> = end * vec2<f32>(textureDimensions(tex));
  let num_sample: u32 = u32(floor(distance( start_px, end_px )) + 1.0);
  let val = convolution_1d(start, end, num_sample);
  return val;
}

/*
  0: match_negative,
  0.5: match_nothing,
  1: match_positive
*/
@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  var matching_max = 0.0;
  var matching_min = 0.0;
  var scale = 1.0;
  for (var scale = 1.0; scale < scale_to.val; scale *= scale_step.val) {
    matching_max = max(matching_max, match_scale(scale, in.uv));
    matching_min = min(matching_min, match_scale(scale, in.uv));
  }

  var matching = matching_max;
  if (abs(matching_max) < abs(matching_min)) {
    matching = matching_min;
  }
  var res = 0.5;
  if (threshold.val < abs(matching)) {
    res = matching / abs(matching);
  }
  return vec4<f32>(res, res, res, 1.0);
}
