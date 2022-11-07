@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var mean: texture_2d<f32>;
@group(0) @binding(3) var<uniform> edge: Vec2Container_f32;

fn smoothstep(low: f32, high: f32, x: f32) -> f32 {
  let t = clamp((x - low) / (high - low), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let low: f32 = edge.val.x;
  let high: f32 = edge.val.y;
  let val: f32 = textureSample(tex, samp, in.uv).r - textureSample(mean, samp, in.uv).r;
  let res: f32 = smoothstep(low, high, val);
  return vec4<f32>(res, res, res, 1.0);
}
