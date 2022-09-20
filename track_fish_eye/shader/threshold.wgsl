@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_samp: sampler;
@group(0) @binding(2) var<uniform> edge: F32Container;
@group(0) @binding(3) var<uniform> negative: U32Container;

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let val = step(edge.val, textureSample(tex, tex_samp, in.uv).r);
  var res = val;
  if (negative.val > u32(0)) {
    res = 1.0 - val;
  }
  return vec4<f32>(res, res, res, 1.0);
}
