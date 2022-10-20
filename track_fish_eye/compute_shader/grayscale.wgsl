@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dest: texture_storage_2d<rgba8unorm, write>;

fn grayscale(color: vec4<f32>) -> vec4<f32> {
  let val = dot(vec3<f32>(0.299, 0.578, 0.114), color.rgb);
  return vec4<f32>(val, val, val, 1.0);
}

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let position = vec2<i32>(global_invocation_id.xy);
  textureStore(dest, position, grayscale(loadPixel(src, position)));
}
