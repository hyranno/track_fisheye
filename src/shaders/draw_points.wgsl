
struct F32Vec2Array {
  val: array<vec2<f32>>,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_samp: sampler;

@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<storage, read> positions: F32Vec2Array;
@group(0) @binding(4) var<storage, read> rotations: F32Vec2Array;

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let val = textureSample(tex, tex_samp, in.uv).r;
  var res = vec4<f32>(0.0, 0.0, val, 1.0);
  let threshold = 0.002;
  let resolution: vec2<f32> = vec2<f32>(textureDimensions(tex));

  for (var i=0; i < i32(arrayLength(&positions.val)); i += 1) {
    let point_px = positions.val[i];
    let point_uv: vec2<f32> = point_px / resolution;
    if (distance(in.uv, point_uv) < threshold) {
      res.x = 1.0;
    }
  }
  for (var i=0; i < i32(arrayLength(&rotations.val)); i += 1) {
    let point_px = rotations.val[i];
    let point_uv: vec2<f32> = point_px / resolution;
    if (distance(in.uv, point_uv) < threshold) {
      res.y = 1.0;
    }
  }

  return vec4<f32>(res);
}
