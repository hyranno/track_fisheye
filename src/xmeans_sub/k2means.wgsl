
type Point = vec2<f32>;

let WORK_GROUP_SIZE: i32 = 256;
let MAX_DATA_NUM: i32 = WORK_GROUP_SIZE;

type array_i32 = array<i32, 256>;
type array_bool = array<bool, 256>;
type array_Point = array<Point, 256>;

struct IndexRange {
  offset: u32,
  length: u32,
};

@group(0) @binding(0) var<storage, read> data_range: IndexRange;
@group(0) @binding(1) var<storage, read> datas: array_Point;
@group(0) @binding(2) var<storage, read_write> assignments: array_i32;
@group(0) @binding(3) var<storage, read> cluster_range: IndexRange;
@group(0) @binding(4) var<storage, read_write> centers: array_Point;
@group(0) @binding(5) var<storage, read_write> counts: array_i32;

var<workgroup> changes: array_bool;


fn assign(lid: u32) {
  let did = data_range.offset + lid;
  let d0 = distance(datas[did], centers[cluster_range.offset + 0u]);
  let d1 = distance(datas[did], centers[cluster_range.offset + 1u]);
  let cluster = select(0, 1, d1 < d0);
  changes[lid] = (assignments[did] != cluster);
  assignments[did] = cluster;
  workgroupBarrier();
}

fn accum_change(lid: u32) {
  for (var stride = u32(MAX_DATA_NUM >> 1u); 0u < stride; stride >>= 1u) {
    let v0 = changes[lid];
    let v1 = changes[lid + stride];
    changes[lid] = select(v0 || v1, v0, data_range.length <= lid + stride);
    workgroupBarrier();
  }
  // result is changes[0]
}

var<workgroup> buffer_center: array_Point;
var<workgroup> buffer_count: array_i32;
fn calc_cluster(lid: u32, cluster: i32) {
  let did = data_range.offset + lid;
  let cid = cluster_range.offset + u32(cluster);
  buffer_center[lid] = select(Point(0.0), datas[did], cluster == assignments[did]);
  buffer_count[lid] = select(1, 0, cluster == assignments[did]);
  workgroupBarrier();
  for (var stride = u32(MAX_DATA_NUM >> 1u); 0u < stride; stride >>= 1u) {
    let p0 = buffer_center[lid];
    let p1 = buffer_center[lid + stride];
    buffer_center[lid] = select(p0 + p1, p0, data_range.length <= lid + stride);
    let n0 = buffer_count[lid];
    let n1 = buffer_count[lid + stride];
    buffer_count[lid] = select(n0 + n1, n0, data_range.length <= lid + stride);
    workgroupBarrier();
  }
  if (lid == 0u) {
    centers[cid] = buffer_center[0] / f32(buffer_count[0]);
    counts[cid] = buffer_count[0];
  }
  workgroupBarrier();
}

fn init(lid: u32) {
  let did = data_range.offset + lid;
  if (lid == 0u) {
    centers[cluster_range.offset + 0u] = datas[data_range.offset + 0u];
    centers[cluster_range.offset + 1u] = datas[data_range.offset + data_range.length - 1u];
  }
  workgroupBarrier();
  assign(lid);
  changes[lid] = true;
  workgroupBarrier();
}

@stage(compute)
@workgroup_size(16, 16)  //WORK_GROUP_SIZE
fn main(@builtin(local_invocation_index) lid : u32) {
  init(lid);
  loop {
    if (!changes[0]) { break; }
    calc_cluster(lid, 0);
    calc_cluster(lid, 1);
    assign(lid);
    accum_change(lid);
  }
}
