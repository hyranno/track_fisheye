@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_samp: sampler;
@group(0) @binding(2) var<uniform> resolution: Vec2Container_u32;
@group(0) @binding(3) var kernel: texture_1d<f32>;
@group(0) @binding(4) var kernel_samp: sampler;
@group(0) @binding(5) var<uniform> range: Vec2Container_f32;
@group(0) @binding(6) var<uniform> normalizing: U32Container;

fn normalizer(num_sample: u32) -> f32 {  // mode is enum
  var result = 1.0;
  switch normalizing.val {
    case 1: {  // num_sample
      result = 1.0 / f32(num_sample);
    }
    case 2: {  // one sum
      var sum = 0.0;
      for (var i = 0; u32(i) < num_sample; i += 1) {
        let kernel_coord = (0.5 + f32(i)) / f32(num_sample);
        sum += textureSample(kernel, kernel_samp, kernel_coord).r;
      }
      result = 1.0 / sum;
    }
    default: {  // None
      result = 1.0;
    }
  }
  return result;
}

fn weighted_sum_1d(start: vec2<f32>, end: vec2<f32>, num_sample: u32) -> vec4<f32> {
  var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let normalize_factor = normalizer(num_sample);
  for (var i = 0; u32(i) < num_sample; i += 1) {
    let kernel_coord = (0.5 + f32(i)) / f32(num_sample);
    let sample_coord = start + (end - start) * kernel_coord;
    result += textureSample(tex, tex_samp, sample_coord) * textureSample(kernel, kernel_samp, kernel_coord).r * normalize_factor;
  }
  return result;
}

fn filter_1d(uv: vec2<f32>) -> vec4<f32> {
  let half_range: vec2<f32> = range.val / 2.0;
  let start: vec2<f32> = uv - half_range;
  let end: vec2<f32> = uv + half_range;
  let start_px: vec2<f32> = start * vec2<f32>(resolution.val);
  let end_px: vec2<f32> = end * vec2<f32>(resolution.val);
  let num_sample: u32 = u32(floor(distance( start_px, end_px )) + 1.0);
  return weighted_sum_1d(start, end, num_sample);
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  // return textureSample(tex, tex_samp, in.uv);
  return filter_1d(in.uv);
}
