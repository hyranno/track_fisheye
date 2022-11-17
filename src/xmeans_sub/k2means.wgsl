let DIM = 2;
type Point = vec2<f32>;

let WORK_GROUP_SIZE: i32 = 256;
let MAX_DATA_NUM: i32 = WORK_GROUP_SIZE;

type array_i32 = array<i32, 256>;
type array_f32 = array<f32, 256>;
type array_Point = array<Point, 256>;

struct IndexRange {
  offset: u32,
  length: u32,
};
struct ClusterIds {
  id: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> data_range: IndexRange;
@group(0) @binding(1) var<storage, read> datas: array_Point;
@group(0) @binding(2) var<storage, read_write> assignments: array_i32;
@group(0) @binding(3) var<storage, read> clusters: ClusterIds;
@group(0) @binding(4) var<storage, read_write> counts: array_i32;
@group(0) @binding(5) var<storage, read_write> means: array_Point;
@group(0) @binding(6) var<storage, read_write> BICs: array_f32;

var<workgroup> changes: array_i32;


var<workgroup> buffer_sum_i32: array_i32;
fn sum_i32(lid: u32, src: ptr<workgroup, array_i32>) -> i32 {
  buffer_sum_i32[lid] = (*src)[lid];
  let dlen2 = 1u << u32(ceil(log2(f32(MAX_DATA_NUM))) - 1.0);
  for (var stride = dlen2; 0u < stride; stride >>= 1u) {
    if (lid < stride) {
      buffer_sum_i32[lid] = buffer_sum_i32[lid] + buffer_sum_i32[lid + stride];
    }
    workgroupBarrier();
    storageBarrier();
  }
  return buffer_sum_i32[0];
}

var<workgroup> buffer_sum_Point: array_Point;
fn sum_Point(lid: u32, src: ptr<workgroup, array_Point>) -> Point {
  buffer_sum_Point[lid] = (*src)[lid];
  let dlen2 = 1u << u32(ceil(log2(f32(MAX_DATA_NUM))) - 1.0);
  for (var stride = dlen2; 0u < stride; stride >>= 1u) {
    if (lid < stride) {
      buffer_sum_Point[lid] = buffer_sum_Point[lid] + buffer_sum_Point[lid + stride];
    }
    workgroupBarrier();
    storageBarrier();
  }
  return buffer_sum_Point[0];
}


fn select_cluster(data: Point, cluster0: u32, cluster1: u32) -> i32 {
  let d0 = distance(data, means[cluster0]);
  let d1 = distance(data, means[cluster1]);
  return select(0, 1, d1 < d0);
}

fn assign(lid: u32) {
  if (lid < data_range.length) {
    let data_id = data_range.offset + lid;
    let cluster = select_cluster(datas[data_id], clusters.id[0], clusters.id[1]);
    changes[lid] = abs(cluster - assignments[data_id]);
    assignments[data_id] = cluster;
  }
  workgroupBarrier();
  storageBarrier();
}

var<workgroup> buffer_count: array_i32;
fn calc_count(lid: u32, cluster: i32) -> i32 {
  let data_id = data_range.offset + lid;
  let is_match_cluster = (lid < data_range.length) && (cluster == assignments[data_id]);
  buffer_count[lid] = select(0, 1, is_match_cluster);
  workgroupBarrier();
  let count = sum_i32(lid, &buffer_count);
  if (lid == 0u) {
    let cid = clusters.id[cluster];
    counts[cid] = count;
  }
  storageBarrier();
  return count;
}

var<workgroup> buffer_mean: array_Point;
fn calc_mean(lid: u32, cluster: i32) -> Point {
  let data_id = data_range.offset + lid;
  let is_match_cluster = (lid < data_range.length) && (cluster == assignments[data_id]);
  if (lid < data_range.length) {
    buffer_mean[lid] = select(Point(0.0), datas[data_id], is_match_cluster);
  } else {
    buffer_mean[lid] = Point(0.0);
  }
  workgroupBarrier();
  let cid = clusters.id[cluster];
  let mean = sum_Point(lid, &buffer_mean) / f32(counts[cid]);
  if (lid == 0u) {
    means[cid] = mean;
  }
  storageBarrier();
  return mean;
}

var<workgroup> buffer_variance: array_Point;
fn calc_variance(lid: u32, cluster: i32) -> Point {
  let data_id = data_range.offset + lid;
  let cid = clusters.id[cluster];
  let is_match_cluster = (lid < data_range.length) && (cluster == assignments[data_id]);
  if (is_match_cluster) {
    let d = (datas[data_id] - means[cid]);
    buffer_variance[lid] = d * d;
  } else {
    buffer_variance[lid] = Point(0.0);
  }
  workgroupBarrier();
  return sum_Point(lid, &buffer_variance) / f32(counts[cid]);
}

fn calc_BIC(lid: u32, cluster: i32) -> f32 {
  let cid = clusters.id[cluster];
  let num_data = f32(counts[cid]);
  let num_dimension = f32(DIM);
  let variance = calc_variance(lid, cluster);
  let v: Point = num_data * (Point(1.0) + log(variance));
  let logL: f32 = -0.5 * (num_data * num_dimension * log(radians(360.0)) + dot(v, Point(1.0)));
  let q = 2.0 * num_dimension;
  let bic = -2.0 * logL + q * log(num_data);
  if (lid == 0u) {
    BICs[cid] = bic;
  }
  storageBarrier();
  return bic;
}


@stage(compute)
@workgroup_size(16, 16)  //WORK_GROUP_SIZE
fn main(@builtin(local_invocation_index) lid : u32) {
  changes[lid] = 0;
  if (lid == 0u) {
    means[clusters.id[0]] = datas[data_range.offset + 0u];
    means[clusters.id[1]] = datas[data_range.offset + data_range.length - 1u];
  }
  if (lid < data_range.length) {
    let data_id = data_range.offset + lid;
    assignments[data_id] = 0;
  }
  loop {
    workgroupBarrier();
    storageBarrier();
    assign(lid);
    if (0 == sum_i32(lid, &changes)) { break; }
    calc_count(lid, 0);
    calc_count(lid, 1);
    calc_mean(lid, 0);
    calc_mean(lid, 1);
  }
  calc_BIC(lid, 0);
  calc_BIC(lid, 1);
}
