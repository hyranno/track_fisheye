@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var mean: texture_2d<f32>;
@group(0) @binding(2) var<uniform> edge: Vec2Container_f32;
@group(0) @binding(3) var dest: texture_storage_2d<rgba8unorm, write>;

fn smoothstep(low: f32, high: f32, x: f32) -> f32 {
  let t = clamp((x - low) / (high - low), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let position = vec2<i32>(global_invocation_id.xy);
  let val = loadPixel(src, position) - loadPixel(mean, position);
  let res = smoothstep(edge.val.x, edge.val.y, val.r);
  textureStore(dest, position, vec4<f32>(res, res, res, 1.0));
}
