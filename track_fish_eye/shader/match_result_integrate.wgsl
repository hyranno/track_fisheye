@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> threshold: F32Container;


@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let val0_min = textureSample(tex0, samp, in.uv).g;
  let val0_max = textureSample(tex0, samp, in.uv).r;
  let val1_min = textureSample(tex1, samp, in.uv).g;
  let val1_max = textureSample(tex1, samp, in.uv).r;

  var res_min = 0.0;
  if (val0_min < -threshold.val && val1_min < -threshold.val) {
    res_min = 1.0;
  }
  var res_max = 0.0;
  if (threshold.val < val0_max && threshold.val < val1_max) {
    res_max = 1.0;
  }

  return vec4<f32>(res_max, res_min, 0.0, 1.0);
}
