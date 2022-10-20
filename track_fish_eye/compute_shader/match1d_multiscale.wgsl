@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<uniform> direction: Vec2Container_i32;
@group(0) @binding(3) var<storage, read> scales_px: array<u32>;
@group(0) @binding(4) var dest: texture_storage_3d<rgba8snorm, write>;

fn scaled_convolution_1d(start: vec2<i32>, num_sample: u32) -> f32 {
  var result = 0.0;
  for (var i = 0; u32(i) < num_sample; i += 1) {
    let tex_index = start + i * direction.val;
    let tex_val = (loadPixel(src, tex_index).r - 0.5) * 2.0;  /* rgba8unorm[0,1] -> rgba8snorm[-1,1] */
    let coord_ratio = (0.5 + f32(i)) / f32(num_sample);
    let kernel_index = i32( coord_ratio * f32(arrayLength(&kernel)) );
    let ker_val = kernel[kernel_index];
    result += tex_val * ker_val;
  }
  return result / f32(num_sample);
}

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let position = vec2<i32>(global_invocation_id.xy);
  let scale_px = scales_px[global_invocation_id.z];
  let half_range = vec2<i32>(direction.val * i32(scale_px) / 2);
  let start = position - half_range;
  let res = scaled_convolution_1d(start, scale_px);
  textureStore(dest, vec3<i32>(global_invocation_id), vec4<f32>(res, res, res, 1.0));
}
