struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct Vec2Container_u32 {
  val: vec2<u32>,
};
struct Vec2Container_f32 {
  val: vec2<f32>,
};
struct U32Container {
  val: u32,
};
