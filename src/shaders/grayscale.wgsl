fn grayscale(color: vec4<f32>) -> vec4<f32> {
  let val = dot(vec3<f32>(0.299, 0.578, 0.114), color.rgb);
  return vec4<f32>(val, val, val, 1.0);
}

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return grayscale(textureSample(tex, samp, in.uv));
}
