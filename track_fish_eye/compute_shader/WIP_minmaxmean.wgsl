@group(0) @binding(0) var tex: texture_3d<f32>;
@group(0) @binding(1) var samp: sampler;

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let num_scales = textureDimensions(tex).z;
  var vmin = textureSample(tex, samp, vec3<f32>(in.uv, 0.0));
  var vmax = vmin;
  var vsum = 0.0;
  for (var i=0; i < num_scales; i++) {
    let w = f32(i) / f32(num_scales - 1);
    let v = textureSample(tex, samp, vec3<f32>(in.uv, w);
    vmin = min(vmin, v);
    vmax = max(vmax, v);
    vsum += v;
  }

  return vec4<f32>(vmax, vmin, vsum / f32(num_scales), 1.0);
}
