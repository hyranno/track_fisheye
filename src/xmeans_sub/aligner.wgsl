type Point = vec2<f32>;

let WORK_GROUP_SIZE: i32 = 1;
let MAX_DATA_NUM: i32 = 256;

type array_i32 = array<i32, 256>;
type array_Point = array<Point, 256>;

struct ArrayVec2U32 {
  value: array<vec2<u32>>,
};

@group(0) @binding(0) var<storage, read> data_ranges: ArrayVec2U32;
@group(0) @binding(1) var<storage, read_write> datas: array_Point;
@group(0) @binding(2) var<storage, read_write> assignments: array_i32;

struct IndexRange {
  offset: u32,
  length: u32,
};
var<workgroup> data_range: IndexRange;

fn swap_data(i: i32, j: i32) {
  let data_t = datas[i];
  datas[i] = datas[j];
  datas[j] = data_t;
  let assignments_t = assignments[i];
  assignments[i] = assignments[j];
  assignments[j] = assignments_t;
}

@stage(compute)
@workgroup_size(1)  //WORK_GROUP_SIZE
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid : u32
) {
  data_range = IndexRange(data_ranges.value[wid.x][0], data_ranges.value[wid.x][1]);
  let start = i32(data_range.offset);
  let end = i32(data_range.offset + data_range.length);
  var i = start;
  var j = end - 1;
  loop {
    for (; assignments[i]==0 && i < end; i+=1) {}
    for (; assignments[j]==1 && start < j; j-=1) {}
    if !(i < j) { break; }
    swap_data(i,j);
  }
}
