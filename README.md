# Background
マーカーの姿勢推定は、トラッキングや自己位置推定に有用

マーカーを用いた姿勢推定はARToolKitやArUco等で提案されているが、マーカー検出処理のラベリングや輪郭検出あたりが重い
これに対して、QRコードの位置検出パターンのようなfisheyeパターンを用いて高速化したものも提案されている
  (https://cir.nii.ac.jp/crid/1050574047078214528)
このプログラムはFisheyeパターンを用いた、より高速化した実装の試み


marker_finder
  本体
marker_format
  マーカーのフォーマット
example_track_camera
  カメラ入力からマーカー検出と姿勢推定
マーカー画像出力するやつ ... salvage


コンセプト
  fisheyeパターンを利用する
    誤マッチしにくい
    一発の検出で全てのマーカーを見つける
  wgpuで並列化する
  状態を持たない
    起動時やエラーがあった際に初期化する必要がない
    フィルタを後付けしてもよい