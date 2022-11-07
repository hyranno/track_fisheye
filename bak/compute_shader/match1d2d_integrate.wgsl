@group(0) @binding(0) var src0: texture_2d<f32>;
@group(0) @binding(1) var src1: texture_2d<f32>;
@group(0) @binding(2) var<uniform> threshold: F32Container;
@group(0) @binding(3) var dest: texture_storage_2d<rgba8unorm, write>;

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let position = vec2<i32>(global_invocation_id.xy);
  let val0 = loadPixel(src0, position);
  let val1 = loadPixel(src1, position);
  var res = vec4<f32>(0.0, 0.0, 0.0, 1.0);

  if (threshold.val < val0.r && threshold.val < val1.r) {
    res.r = 1.0;
  }
  if (threshold.val < -val0.g && threshold.val < -val1.g) {
    res.g = 1.0;
  }

  /*
  res.r = min(val0.r, val1.r);
  res.g = max(-val0.g, -val1.g);
  */

  textureStore(dest, position, res);
}
