@group(0) @binding(0) var src: texture_2d<f32>; /* rgba8snorm */
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<uniform> direction: Vec2Container_i32;
@group(0) @binding(3) var dest: texture_storage_2d<rgba8snorm, write>;

fn convolution_1d(start: vec2<i32>, num_sample: u32) -> vec4<f32> {
  var result = vec4<f32>(0.0);
  for (var i = 0; u32(i) < num_sample; i += 1) {
    let tex_index = start + i * direction.val;
    let tex_val = loadPixel(src, tex_index);
    let ker_val = kernel[i];
    result += tex_val * ker_val;
  }
  return result;
}

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let position = vec2<i32>(global_invocation_id.xy);
  let scale_px = arrayLength(&kernel);
  let half_range = vec2<i32>(direction.val * i32(scale_px) / 2);
  let start = position - half_range;
  let res = convolution_1d(start, scale_px);
  textureStore(dest, position, vec4<f32>(res.xyz, 1.0));
}
