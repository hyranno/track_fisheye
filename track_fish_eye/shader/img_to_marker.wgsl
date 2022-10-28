@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> affine: Mat3x3Container_f32;
@group(0) @binding(3) var<uniform> dest_len: U32Container;

fn resample(A: mat3x3<f32>, uv: vec2<f32>) -> vec4<f32> {
  let s = 0.0; /* 0.3 / f32(dest_len.val); */
  let pc: vec2<f32> = (A * vec3<f32>(uv.x, uv.y, 1.0)).xy;
  let pu: vec2<f32> = (A * vec3<f32>(uv.x, uv.y+s, 1.0)).xy;
  let pd: vec2<f32> = (A * vec3<f32>(uv.x, uv.y-s, 1.0)).xy;
  let pl: vec2<f32> = (A * vec3<f32>(uv.x+s, uv.y, 1.0)).xy;
  let pr: vec2<f32> = (A * vec3<f32>(uv.x-s, uv.y, 1.0)).xy;
  let vc = textureSample(src, samp, pc);
  let vu = textureSample(src, samp, pu);
  let vd = textureSample(src, samp, pd);
  let vl = textureSample(src, samp, pl);
  let vr = textureSample(src, samp, pr);
  let v = 0.5*vc + 0.5*(vu + vd + vl + vr)/4.0;
  return v;
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let res = step(0.5, resample(affine.val, in.uv).r);
  return vec4<f32>(res, res, res, 1.0);
}
