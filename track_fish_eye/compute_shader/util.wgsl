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
  return textureLoad(tex, index, 0);
}
