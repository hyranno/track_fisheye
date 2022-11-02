@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, in.uv);
}
