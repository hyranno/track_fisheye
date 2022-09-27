
let DIMENSION = 2;

struct Data {
  point: array<f32, DIMENSION>,
  cluster: i32,
}
struct Cluster {
  offset: i32,
  num_data: i32,
  mean: array<f32, DIMENSION>,
  variance: array<f32, DIMENSION>,
  bic: f32,  //Bayesian Information Criterion
}

@group(0) @binding(0) var<storage,read> src: array<f32>;
@group(0) @binding(1) var<storage,read_write> datas: array<Data>;
@group(0) @binding(2) var<storage,read_write> clusters: array<Cluster>;
@group(0) @binding(3) var<storage,read_write> datas_div: array<Data>;
@group(0) @binding(4) var<storage,read_write> clusters_div: array<Cluster>;

fn count(
  cluster_index: i32,
  datas: ptr<storage, array<Data>, read_write>,
  clusters: ptr<storage, array<Cluster>, read_write>,
){}
fn calc_mean(
  cluster_index: i32,
  datas: ptr<storage, array<Data>, read_write>,
  clusters: ptr<storage, array<Cluster>, read_write>,
){}
fn calc_variance(  // requires calculated mean
  cluster_index: i32,
  datas: ptr<storage, array<Data>, read_write>,
  clusters: ptr<storage, array<Cluster>, read_write>,
){}
fn calc_bic(  // requires calculated mean, variance
  cluster_index: i32,
  datas: ptr<storage, array<Data>, read_write>,
  clusters: ptr<storage, array<Cluster>, read_write>,
){}

fn pick_first_centers(src_index: i32, target_indices: array<i32, 2>){}
fn reassign_to_cluster(target_indices: array<i32, 2>){}
fn kmeans(src_index: i32, target_indices: array<i32, 2>){
}

fn calc_bic_sub(){}
fn xmeans(){}
