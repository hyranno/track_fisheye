@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> affine: Mat3x3Container_f32;

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let p: vec2<f32> = (affine.val * vec3<f32>(in.uv, 1.0)).xy;
  return textureSample(src, samp, p);
}
