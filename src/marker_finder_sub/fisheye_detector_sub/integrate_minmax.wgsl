@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let v0 = textureSample(tex0, samp, in.uv);
  let v1 = textureSample(tex1, samp, in.uv);
  return vec4<f32>(max(v0.r, v1.r), min(v0.g, v1.g), mix(v0.b, v1.b, 0.5), 1.0);
}
