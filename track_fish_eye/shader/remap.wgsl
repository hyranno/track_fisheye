@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_samp: sampler;
@group(0) @binding(2) var<uniform> bias: Vec4Container_f32;
@group(0) @binding(3) var<uniform> amp: Vec4Container_f32;

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return (textureSample(tex, tex_samp, in.uv) + bias.val) * amp.val;
}
