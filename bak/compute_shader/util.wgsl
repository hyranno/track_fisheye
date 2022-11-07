struct Vec4Container_f32 {
  val: vec4<f32>,
};
struct Vec2Container_u32 {
  val: vec2<u32>,
};
struct Vec2Container_i32 {
  val: vec2<i32>,
};
struct Vec2Container_f32 {
  val: vec2<f32>,
};
struct F32Vec2ArrayContainer {
  val: array<vec2<f32>>,
};
struct U32Container {
  val: u32,
};
struct F32Container {
  val: f32,
};

fn loadPixel(tex: texture_2d<f32>, index: vec2<i32>) -> vec4<f32> {
  let low = vec2<i32>(0, 0);
  let high = vec2<i32>(textureDimensions(tex)) - vec2<i32>(1, 1);
  let i = clamp(index, low, high);
  return textureLoad(tex, i, 0);
}

fn loadPixelLinear(tex: texture_2d<f32>, index: vec2<f32>) -> vec4<f32> {
  let low = vec2<f32>(0.0, 0.0);
  let high = vec2<f32>(textureDimensions(tex)) - vec2<f32>(1.0, 1.0);
  let i = clamp(index, low, high);
  let fi = vec2<i32>(floor(i));
  let ci = vec2<i32>(ceil(i));
  let r = i - vec2<f32>(fi);
  let val_ff = textureLoad( tex, fi, 0 );
  let val_fc = textureLoad( tex, vec2<i32>(fi.x, ci.y), 0 );
  let val_cf = textureLoad( tex, vec2<i32>(ci.x, fi.y), 0 );
  let val_cc = textureLoad( tex, ci, 0 );
  return (1.0-r.x)*(1.0-r.y)*val_ff + (1.0-r.x)*r.y*val_fc + r.x*(1.0-r.y)*val_cf + r.x*r.y*val_cc;
}
