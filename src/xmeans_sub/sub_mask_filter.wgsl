type Point = vec2<f32>;

let WORK_GROUP_SIZE: i32 = 1;
let MAX_DATA_NUM: i32 = 256;

type array_i32 = array<i32, 256>;
type array_f32 = array<f32, 256>;
type array_Point = array<Point, 256>;

struct ContainerI32 {
  value: i32,
};
struct ContainerF32 {
  value: f32,
};
struct ArrayVec2U32 {
  value: array<vec2<u32>>,
};
struct Cluster {
  count: i32,
  _: i32,
  mean: Point,
  variance: Point,
  BIC: f32,
  _: i32,
};
type array_Cluster = array<Cluster, 256>;


@group(0) @binding(0) var<storage, read> sub_clusters: array_Cluster;
@group(0) @binding(1) var<storage, read> parent_clusters: array_Cluster;
@group(0) @binding(2) var<storage, read> cluster_pairs: ArrayVec2U32;
@group(0) @binding(3) var<uniform> depth: ContainerI32;
@group(0) @binding(4) var<uniform> min_distance: ContainerF32;
@group(0) @binding(5) var<storage, read_write> sub_mask: array_i32;
/*
  0: no apply
  1: apply, done
  2: apply, then recurse
*/


fn cumulative_norm(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-1.7*x));  //0.5 * (1 + erf(x/sqrt(2)))
}

fn calc_sub_BIC(cids: vec2<u32>) -> f32 {
  let c0 = sub_clusters[cids[0]];
  let c1 = sub_clusters[cids[1]];
  let d = distance(c0.mean, c1.mean);
  let vprods = (c0.variance[0] * c0.variance[1]) + (c1.variance[0] * c1.variance[1]);
  var k = 1.0;
  if (0.0 < vprods) {
    k = cumulative_norm(sqrt(d*d/vprods));
  }
  let q = f32(8);  /* 2*2*DIM */
  let cc = c0.count + c1.count;
  let lc = log(f32(cc * cc) / f32(c0.count * c1.count));
  return c0.BIC + c1.BIC - 2.0*log(0.5/k) + q * lc;
}

fn is_target(cids: vec2<u32>) -> bool {
  let sub_dictance = distance(sub_clusters[cids[0]].mean, sub_clusters[cids[1]].mean);
  let is_distance_ok = min_distance.value < sub_dictance;
  let sub_BIC = calc_sub_BIC(cids);
  let parent_BIC = parent_clusters[cids[0]].BIC;
  let is_BIC_ok = (depth.value == 0) || (sub_BIC < parent_BIC);
  return is_distance_ok && is_BIC_ok;
}

@stage(compute)
@workgroup_size(1)  //WORK_GROUP_SIZE
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid : u32
) {
  let cids = cluster_pairs.value[wid.x];
  let is_apply_target = is_target(cids);
  sub_mask[cids[0]] = select(0, select(1, 2, 2 <= sub_clusters[cids[0]].count), is_apply_target);
  sub_mask[cids[1]] = select(0, select(1, 2, 2 <= sub_clusters[cids[1]].count), is_apply_target);
}
