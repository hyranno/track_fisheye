@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> direction: Vec2Container_i32;

fn filter(start: vec2<f32>, end: vec2<f32>) -> f32 {
  let num_sample = i32(arrayLength(&kernel));
  var result = 0.0;
  for (var i = 0; i < num_sample; i += 1) {
    let coord_ratio = (0.5 + f32(i)) / f32(num_sample);
    let sample_coord = start + (end - start) * coord_ratio;
    let tex_val = textureSample(tex, samp, sample_coord).r;
    let ker_val = kernel[i];
    result += tex_val * ker_val;
  }
  return result;
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let half_range = f32(arrayLength(&kernel)) / 2.0 * vec2<f32>(direction.val) / vec2<f32>(textureDimensions(tex));
  let start: vec2<f32> = in.uv - half_range;
  let end: vec2<f32> = in.uv + half_range;
  let res = filter(start, end);
  return vec4<f32>(res, res, res, 1.0);
}
