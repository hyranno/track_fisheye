@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex0_samp: sampler;
@group(0) @binding(2) var<uniform> threshold0: F32Container;
@group(0) @binding(3) var tex1: texture_2d<f32>;
@group(0) @binding(4) var tex1_samp: sampler;
@group(0) @binding(5) var<uniform> threshold1: F32Container;
@group(0) @binding(6) var<uniform> op: U32Container;

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let val0: bool = threshold0.val < textureSample(tex0, tex0_samp, in.uv).r;
  let val1: bool = threshold1.val < textureSample(tex1, tex1_samp, in.uv).r;
  let op_core = op.val >> u32(1);
  let op_not: bool = (op.val & u32(1)) > u32(0);
  var val = false;
  switch op_core {
    case 0: {  // OR
      val = val0 || val1;
    }
    case 1: {  // AND
      val = val0 && val1;
    }
    case 2: {  // XOR
      val = (val0 && !val1) || (!val0 && val1);
    }
    default: {
      val = false;
    }
  }
  if op_not {
    val = !val;
  }
  var res = 0.0;
  if val {
    res = 1.0;
  }
  return vec4<f32>(res, res, res, 1.0);
}
