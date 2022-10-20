@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read> starts: F32Vec2ArrayContainer;
@group(0) @binding(3) var<storage, read> ends: F32Vec2ArrayContainer;
@group(0) @binding(4) var dest: texture_storage_2d<rgba8unorm, write>;

fn convolution_1d(start: vec2<f32>, end: vec2<f32>) -> f32 {
  let num_sample = i32(distance(start, end));
  var result = 0.0;
  for (var i = 0; i < num_sample; i += 1) {
    let coord_ratio = (0.5 + f32(i)) / f32(num_sample);
    let tex_coord = start + (end - start) * coord_ratio;
    let src_val = (loadPixelLinear(src, tex_coord).r - 0.5) * 2.0;  /* [0,1] to [-1,1] */
    let kernel_index = i32( coord_ratio * f32(arrayLength(&kernel)) );
    let ker_val = kernel[kernel_index];
    result += src_val * ker_val;
  }
  return result / f32(num_sample);
}

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let position = vec2<i32>(global_invocation_id.xy);
  let i = i32(global_invocation_id.y);
  let j = i32(global_invocation_id.x);
  let center = (starts.val[i] + ends.val[j]) / 2.0;
  let start = center + (starts.val[i] - center) * 2.0;
  let end = center + (ends.val[j] - center) * 2.0;
  let res = convolution_1d(start, end);
  textureStore(dest, position, vec4<f32>(res, res, res, 1.0));
}
