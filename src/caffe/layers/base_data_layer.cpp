#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.InitRand();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  if (this->multi_blob_mode_) {
    // At this point, top is already the correct size, so we just resize
    // the prefetch vector to match
    LOG(INFO) << "JBY top vector is of size " << top.size();

    for (int blob_id = 0; blob_id < top.size(); ++blob_id) {
      this->prefetch_blobs_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
    }
    //this->prefetch_blobs_.resize(top.size());

    //this->transform_blob_.resize(top.size(), false);
    //this->transform_blob_[0] = true; // Hard coded for now: always transform the first blob and none of the others. See note in data_layers.hpp
    CHECK_EQ(this->output_labels_, false) << "output_labels_ should not be used with multi_blob_mode_ (just use another blob for labels).";
  }

  // Now, start the prefetch thread. Before calling prefetch, we make a few
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  if (this->multi_blob_mode_) {
    for (int blob_id = 0; blob_id < this->prefetch_blobs_.size(); ++blob_id) {
      this->prefetch_blobs_[blob_id]->mutable_cpu_data();
    }
  } else {
    this->prefetch_blobs_[0]->mutable_cpu_data();
    if (this->output_labels_) {
      this->prefetch_label_.mutable_cpu_data();
    }
  }

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";

  // CHECK_EQ(this->prefetch_data_.size(), 0) << "prefetch_data_ should not be used by JBY_BasePrefetchingMultiDataLayer";
  // CHECK_EQ(this->prefetch_label_.size(), 0) << "prefetch_label_ should not be used by JBY_BasePrefetchingMultiDataLayer";
  // CHECK_EQ(this->transformed_data_.size(), 0) << "transformed_data_ should not be used by JBY_BasePrefetchingMultiDataLayer";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Copy the data
  if (this->multi_blob_mode_) {
    for (int blob_id = 0; blob_id < this->prefetch_blobs_.size(); ++blob_id) {
      caffe_copy(this->prefetch_blobs_[blob_id]->count(),
                 this->prefetch_blobs_[blob_id]->cpu_data(),
                 top[blob_id]->mutable_cpu_data());
    }
  } else {
    caffe_copy(prefetch_blobs_[0]->count(), prefetch_blobs_[0]->cpu_data(),
               top[0]->mutable_cpu_data());
    if (this->output_labels_) {
      caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
                 top[1]->mutable_cpu_data());
    }
  }
  DLOG(INFO) << "Prefetch copied";
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

// template <typename Dtype>
// void JBY_BasePrefetchingMultiDataLayer<Dtype>::LayerSetUp(
//     const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//   BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

//   // At this point, top is already the correct size, so we just resize
//   // the prefetch vector to match
//   LOG(INFO) << "JBY top vector is of size " << top.size();
//   this->prefetch_blobs_.resize(top.size());
//   CHECK_EQ(this->prefetch_data_.size(), 0) << "prefetch_data_ should not be used by JBY_BasePrefetchingMultiDataLayer";
//   CHECK_EQ(this->prefetch_label_.size(), 0) << "prefetch_label_ should not be used by JBY_BasePrefetchingMultiDataLayer";
//   CHECK_EQ(this->transformed_data_.size(), 0) << "transformed_data_ should not be used by JBY_BasePrefetchingMultiDataLayer";

//   // Now, start the prefetch thread. Before calling prefetch, we make a few
//   // cpu_data calls so that the prefetch thread does not accidentally make
//   // simultaneous cudaMalloc calls when the main thread is running. In some
//   // GPUs this seems to cause failures if we do not so.
//   for (int blob_id = 0; blob_id < this->prefetch_blobs_.size(); ++blob_id) {
//     this->prefetch_blobs_.mutable_cpu_data()
//   }

//   DLOG(INFO) << "Initializing prefetch";
//   this->CreatePrefetchThread();
//   DLOG(INFO) << "Prefetch initialized.";
// }

// template <typename Dtype>
// void JBY_BasePrefetchingMultiDataLayer<Dtype>::Forward_cpu(
//     const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//   // First, join the thread
//   JoinPrefetchThread();
//   DLOG(INFO) << "Thread joined";

//   // Copy the data in each prefetch_blob_ to the corresponding top blob
//   for (int blob_id = 0; blob_id < this->prefetch_blobs_.size(); ++blob_id) {
//     caffe_copy(this->prefetch_blobs_[blob_id].count(),
//                this->prefetch_blobs_[blob_id].cpu_data(),
//                top[blob_id]->mutable_cpu_data());
//   }

//   DLOG(INFO) << "Prefetch copied";
//   // Start a new prefetch thread
//   DLOG(INFO) << "CreatePrefetchThread";
//   CreatePrefetchThread();
// }

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
  //STUB_GPU_FORWARD(JBY_BasePrefetchingMultiDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);
  //INSTANTIATE_CLASS(JBY_BasePrefetchingMultiDataLayer);

}  // namespace caffe
