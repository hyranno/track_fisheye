@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex0_samp: sampler;
@group(0) @binding(2) var tex1: texture_2d<f32>;
@group(0) @binding(3) var tex1_samp: sampler;

fn color_to_val(c: f32) -> i32 {
  var res = 0;
  if (c < 0.4) {
    res = -1;
  }
  if (0.6 < c) {
    res = 1;
  }
  return res;
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let val0: i32 = color_to_val(textureSample(tex0, tex0_samp, in.uv).r);
  let val1: i32 = color_to_val(textureSample(tex1, tex1_samp, in.uv).r);

  var res = 0.5;
  if (val0 < 0 && val1 < 0) {
    res = 0.0;
  }
  if (0 < val0 && 0 < val1) {
    res = 1.0;
  }
  return vec4<f32>(res, res, res, 1.0);
}
