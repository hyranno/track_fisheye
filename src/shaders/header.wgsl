struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct Mat3x3Container_f32 {
  val: mat3x3<f32>,
};
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
