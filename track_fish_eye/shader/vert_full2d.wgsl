@stage(vertex)
fn vs_main(in: VertexInput) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let index = i32(in.vertex_index);
    let p: vec2<f32> = positions[index];

    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = (-p + 1.0) / 2.0;
    return out;
}
