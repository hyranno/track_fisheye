@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dest: texture_storage_2d<rgba8snorm, write>;

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let position = vec2<i32>(global_invocation_id.xy);
  let num_scales = i32(textureDimensions(src).z);
  var vmin = textureLoad(src, vec3<i32>(position, 0), 0).r;
  var vmax = vmin;
  var vsum = vmin;
  for (var s=1; s < num_scales; s += 1) {
    let v = textureLoad(src, vec3<i32>(position, s), 0).r;
    vmin = min(vmin, v);
    vmax = max(vmax, v);
    vsum += v;
  }
  let res = vec4<f32>(vmax, vmin, vsum / f32(num_scales), 1.0);
  textureStore(dest, position, res);
}
