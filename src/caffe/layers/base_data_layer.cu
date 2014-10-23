#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  if (this->multi_blob_mode_) {
    // Copy the data in each prefetch_blob_ to the corresponding top blob
    for (int blob_id = 0; blob_id < this->prefetch_blobs_.size(); ++blob_id) {
      caffe_copy(this->prefetch_blobs_[blob_id]->count(),
                 this->prefetch_blobs_[blob_id]->cpu_data(),
                 top[blob_id]->mutable_gpu_data());
    }
  } else {
    caffe_copy(prefetch_blobs_[0]->count(), prefetch_blobs_[0]->cpu_data(),
               top[0]->mutable_gpu_data());
    if (this->output_labels_) {
      caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
                 top[1]->mutable_gpu_data());
    }
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

// template <typename Dtype>
// void JBY_BasePrefetchingMultiDataLayer<Dtype>::Forward_gpu(
//     const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//   // First, join the thread
//   JoinPrefetchThread();
//   DLOG(INFO) << "Thread joined";

//   // Copy the data in each prefetch_blob_ to the corresponding top blob
//   for (int blob_id = 0; blob_id < this->prefetch_blobs_.size(); ++blob_id) {
//     caffe_copy(this->prefetch_blobs_[blob_id].count(),
//                this->prefetch_blobs_[blob_id].cpu_data(),
//                top[blob_id]->mutable_gpu_data());
//   }
//   // Start a new prefetch thread
//   CreatePrefetchThread();
// }

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
// INSTANTIATE_LAYER_GPU_FORWARD(JBY_BasePrefetchingMultiDataLayer);

}  // namespace caffe
