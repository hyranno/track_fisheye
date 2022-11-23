type Point = vec2<f32>;

let WORK_GROUP_SIZE: i32 = 256;
let MAX_DATA_NUM: i32 = 256;

type array_i32 = array<i32, 256>;
type array_f32 = array<f32, 256>;
type array_Point = array<Point, 256>;

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

@group(0) @binding(0) var<storage, read> src_datas: array_Point;
@group(0) @binding(1) var<storage, read> src_clusters: array_Cluster;

@group(0) @binding(2) var<storage, read_write> dest_datas: array_Point;
@group(0) @binding(3) var<storage, read_write> dest_clusters: array_Cluster;

@group(0) @binding(4) var<storage, read> data_ranges: ArrayVec2U32;
@group(0) @binding(5) var<storage, read> cluster_pairs: ArrayVec2U32;

struct IndexRange {
  offset: u32,
  length: u32,
};

@stage(compute)
@workgroup_size(256)  //WORK_GROUP_SIZE
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid : u32
) {
  let data_range = IndexRange(data_ranges.value[wid.x][0], data_ranges.value[wid.x][1]);
  let data_id = data_range.offset + lid;
  if (lid < data_range.length) {
    dest_datas[data_id] = src_datas[data_id];
  }
  let cluster_ids = cluster_pairs.value[wid.x];
  if (lid < 2u) {
    let cid = cluster_ids[lid];
    dest_clusters[cid].count = src_clusters[cid].count;
    dest_clusters[cid].mean = src_clusters[cid].mean;
    dest_clusters[cid].variance = src_clusters[cid].variance;
    dest_clusters[cid].BIC = src_clusters[cid].BIC;
  }
}
