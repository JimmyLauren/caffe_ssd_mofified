#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionEvaluateParameter& detection_evaluate_param =
      this->layer_param_.detection_evaluate_param();
  CHECK(detection_evaluate_param.has_num_classes())
      << "Must provide num_classes.";
  num_classes_ = detection_evaluate_param.num_classes();


  //*******************************************************
  // added by Jimmy
  // ��Ҫ�Ǹ�label_project��Old2NewLabel_��ֵ
  label_project = detection_evaluate_param.label_project();
  if (label_project) {
	  if (detection_evaluate_param.has_label_project_file()) {
		  //if there exists a label project file in model.prototxt
		  string label_project_file = detection_evaluate_param.label_project_file();
		  std::ifstream labelProjectInfile(label_project_file.c_str());
		  CHECK(labelProjectInfile.good())
			  << "Failed to open label project file: " << label_project_file;
		  // The file is in the following format:
		  //     old_label  new_label
		  //     ......
		  int old_label, new_label;
		  while (labelProjectInfile >> old_label >> new_label) {
			  Old2NewLabel_.push_back(std::make_pair(old_label, new_label));
			  std::cout << old_label << ' ' << new_label << std::endl;
		  }
		  labelProjectInfile.close();
	  }
  }
  //*********************************************************

  background_label_id_ = detection_evaluate_param.background_label_id();
  overlap_threshold_ = detection_evaluate_param.overlap_threshold();
  CHECK_GT(overlap_threshold_, 0.) << "overlap_threshold must be non negative.";
  evaluate_difficult_gt_ = detection_evaluate_param.evaluate_difficult_gt();
  if (detection_evaluate_param.has_name_size_file()) {
    string name_size_file = detection_evaluate_param.name_size_file();
    std::ifstream infile(name_size_file.c_str());
    CHECK(infile.good())
        << "Failed to open name size file: " << name_size_file;
    // The file is in the following format:
    //    name height width
    //    ...
    string name;
    int height, width;
    while (infile >> name >> height >> width) {
      sizes_.push_back(std::make_pair(height, width));
    }
    infile.close();
  }
  count_ = 0;
  // If there is no name_size_file provided, use normalized bbox to evaluate.
  use_normalized_bbox_ = sizes_.size() == 0;

  // Retrieve resize parameter if there is any provided.
  has_resize_ = detection_evaluate_param.has_resize_param();
  if (has_resize_) {
    resize_param_ = detection_evaluate_param.resize_param();
  }
}

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(count_, sizes_.size());
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->width(), 7);
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->width(), 8);

  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  int num_pos_classes = background_label_id_ == -1 ?
      num_classes_ : num_classes_ - 1;
  int num_valid_det = 0;
  const Dtype* det_data = bottom[0]->cpu_data();
  for (int i = 0; i < bottom[0]->height(); ++i) {
    if (det_data[1] != -1) {
      ++num_valid_det;
    }
    det_data += 7;
  }
  top_shape.push_back(num_pos_classes + num_valid_det);
  // Each row is a 5 dimension vector, which stores
  // [image_id, label, confidence, true_pos, false_pos]
  top_shape.push_back(5);
  top[0]->Reshape(top_shape);
}

//д���������������������ӳ��
/*
bool label_project; // �Ƿ�Ҫ���б�ǩӳ��
string label_project_file; //��ǩӳ���ļ��ĵ�ַ
�����������������������Ʋ����Ƿ���б�ǩӳ��
*/
template<typename Dtype>
int DetectionEvaluateLayer<Dtype>::Label_Project(int & label){
	if (label_project) {  //�����ж��Ƿ�Ҫ���б�ǩӳ��
		//��Ϊ���ٽ��б�ǩӳ��
		//std::vector<std::pair<int, int>>::iterator it = find(Old2NewLabel_.begin(), Old2NewLabel_.end(), label); //�ҵ���label��ǩ����λ�õı��
		std::vector<std::pair<int, int>>::iterator it = Old2NewLabel_.begin();
		for (;it != Old2NewLabel_.end();it++) {
			int tempOldLabel = it->first; //old label
			if (tempOldLabel == label) {
				break;
			}
		}
		if (it == Old2NewLabel_.end()) { //labelû����label_project_file�����׳��쳣
			std::cout << "There is no " << label << " in label_project_file!!" << std::endl;
			throw - 1;
		}
		return (int)it->second;//�õ��ɵ�it->first��Ӧ���µ�it->second��ǩ
	}
	else
		return label;
}


template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data(); //detection data;
  const Dtype* gt_data = bottom[1]->cpu_data(); //ground truth data; 

  /*
  map<int,LabelBBox> �Ľṹ������
	��һ��intΪimage id����ͼƬ�ı�־
	�ڶ���LabelBBoxΪ��ǩ-�����
		LabelBBox�ĵ�һ��Ϊ��ǩ
		LabelBBox�ĵڶ���Ϊ�ñ�ǩ��Ӧ��λ�÷�����Ϣ��predict������ground truth��
  */

  // Retrieve all detection results. 
  //�ڵõ����е�Ԥ����֮����Ҫͳһ�����е�Ԥ�����еľɱ�ǩ���ĵ��±�ǩ     1111111
  map<int, LabelBBox> all_detections;  //image_id, label, boxes

  //GetDetectionResults(det_data, bottom[0]->height(), background_label_id_,
  //                    &all_detections); //�õ�ģ��Ԥ������еĽ��

  //added by JimmyLau
  /*(
	const Dtype* det_data, 
	const int num_det,
    const int background_label_id,
	map<int, map<int, vector<NormalizedBBox> > >* all_detections) */


  //const Dtype* det_data = det_data;
  const int tempNum_det = bottom[0]->height();
  const int tempBackground_label_id = background_label_id_;
  //map<int, map<int, vector<NormalizedBBox> > >* all_detections = &all_detections;
  {
	  all_detections.clear();
	  for (int i = 0; i < tempNum_det; ++i) {
		  int start_idx = i * 7;
		  int item_id = det_data[start_idx];
		  if (item_id == -1) {
			  continue;
		  }
		  int label = det_data[start_idx + 1];

		  int tempLabel = Label_Project(label);

		  std::cout << "Predict old label: " << label << "; Predict new label: " << tempLabel << std::endl;

		  label = tempLabel;  //changing old labels to new labels
		  CHECK_NE(tempBackground_label_id, label)
			  << "Found background label in the detection results.";
		  NormalizedBBox bbox;
		  bbox.set_score(det_data[start_idx + 2]);
		  bbox.set_xmin(det_data[start_idx + 3]);
		  bbox.set_ymin(det_data[start_idx + 4]);
		  bbox.set_xmax(det_data[start_idx + 5]);
		  bbox.set_ymax(det_data[start_idx + 6]);
		  float bbox_size = BBoxSize(bbox);
		  bbox.set_size(bbox_size);
		  all_detections[item_id][label].push_back(bbox);
	  }
  }


  // Retrieve all ground truth (including difficult ones).
  map<int, LabelBBox> all_gt_bboxes; // all ground truth boxes
  GetGroundTruth(gt_data, bottom[1]->height(), background_label_id_,
                 true, &all_gt_bboxes);

  Dtype* top_data = top[0]->mutable_cpu_data(); //output data
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  int num_det = 0;

  all_detections, all_gt_bboxes;
  // Insert number of ground truth for each label. �õ�ÿ����ǩ������ground truth��Ŀ
  map<int, int> num_pos; //map�ĵ�һ����Ϊ����ǩ���ڶ�����Ϊ������ǩ��ground truth����Ŀ
  for (map<int, LabelBBox>::iterator it = all_gt_bboxes.begin();it != all_gt_bboxes.end(); ++it) {  
		//������е�ground truth boxes����
		for (LabelBBox::iterator iit = it->second.begin(); iit != it->second.end();++iit) { 
			  int count = 0;
			  if (evaluate_difficult_gt_) {
					//if evaluate_difficult_gt_ is true, then using all ground truth
					count = iit->second.size();
			  } 
			  else {
					// Get number of non difficult ground truth.
					for (int i = 0; i < iit->second.size(); ++i) {
						  if (!iit->second[i].difficult()) {
								++count;
						  }
					}
			  }
			  if (num_pos.find(iit->first) == num_pos.end()) {
					num_pos[iit->first] = count;
			  } 
			  else {
					num_pos[iit->first] += count;
			  }
		}
  }

  //num_pos�ĵ�һ����Ϊ����ǩ���ڶ�����Ϊ������ǩ��ground truth����Ŀ
  // this is for the first part of concated data. the concat data is made up of location data and confidence data
  for (int c = 0; c < num_classes_; ++c) {  //�������е����
		if (c == background_label_id_) {  //�����ų�
			continue;
		}
		top_data[num_det * 5] = -1;
		top_data[num_det * 5 + 1] = c;  //�˴������⣬��������prototxt�ļ��а����Ĺ��������ڴ˴��������ӳ�䣿�� �ú���һ��  2222
		//top_data[num_det * 5 + 1] = Label_Project(c);
		if (num_pos.find(c) == num_pos.end()) { //��c����ground truth
			top_data[num_det * 5 + 2] = 0;
		} 
		else {
			top_data[num_det * 5 + 2] = num_pos.find(c)->second;//���c��ground truth��Ŀ
		}
		top_data[num_det * 5 + 3] = -1;
		top_data[num_det * 5 + 4] = -1;
		++num_det;
  }

  // Insert detection evaluate status. For all detections:
  for (map<int, LabelBBox>::iterator it = all_detections.begin();it != all_detections.end(); ++it) 
  {  
	  // for all detections produced by model
		int image_id = it->first;
		LabelBBox& detections = it->second; //image_id, detection boxes
		if (all_gt_bboxes.find(image_id) == all_gt_bboxes.end()) {  // No ground truth for current image. All detections become false_pos.
				for (LabelBBox::iterator iit = detections.begin();iit != detections.end(); ++iit) {
					int label = iit->first;
					if (label == -1) {
						continue;
					}
					const vector<NormalizedBBox>& bboxes = iit->second;
					for (int i = 0; i < bboxes.size(); ++i) {
							top_data[num_det * 5] = image_id; 
							top_data[num_det * 5 + 1] = label;  //******
							//top_data[num_det * 5 + 1] = Label_Project(label);
							//std::cout << "old label: " << label << "; new label: " << Label_Project(label) << std::endl;

							top_data[num_det * 5 + 2] = bboxes[i].score();
							top_data[num_det * 5 + 3] = 0;  // false_pos��01��ʾ������01�ֱ�Ϊtop_data�ĵ��ĵ���λ
							top_data[num_det * 5 + 4] = 1;
							++num_det;
					}
				}
		} 
		else {  // ��������ȷ��Ԥ��
				LabelBBox& label_bboxes = all_gt_bboxes.find(image_id)->second;
				for (LabelBBox::iterator iit = detections.begin();iit != detections.end(); ++iit) { //for all detections 
					int label = iit->first;
					if (label == -1) {
						continue;
					}
					vector<NormalizedBBox>& bboxes = iit->second;
					if (label_bboxes.find(label) == label_bboxes.end()) {
						// No ground truth for current label. All detections become false_pos.
						for (int i = 0; i < bboxes.size(); ++i) {
							top_data[num_det * 5] = image_id;
							top_data[num_det * 5 + 1] = label;  //******
							//top_data[num_det * 5 + 1] = Label_Project(label);
							//std::cout << "Predict old label: " << label << "; Predict new label: " << Label_Project(label) << std::endl;

							top_data[num_det * 5 + 2] = bboxes[i].score();

							//ͨ��top_data��num_det * 5 + 3λ��num_det * 5 + 4λ����ʾtp,fp���࣬tp:1 0; fp:0 1.
							top_data[num_det * 5 + 3] = 0;
							top_data[num_det * 5 + 4] = 1;
							++num_det;
						}
					} 
					else {
						vector<NormalizedBBox>& gt_bboxes = label_bboxes.find(label)->second;
						// Scale ground truth if needed.
						if (!use_normalized_bbox_) {
						CHECK_LT(count_, sizes_.size());
						for (int i = 0; i < gt_bboxes.size(); ++i) {
							OutputBBox(gt_bboxes[i], sizes_[count_], has_resize_,
										resize_param_, &(gt_bboxes[i]));
						}
						}
						vector<bool> visited(gt_bboxes.size(), false);
						// Sort detections in descend order based on scores.
						std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
						for (int i = 0; i < bboxes.size(); ++i) {
						top_data[num_det * 5] = image_id;
						top_data[num_det * 5 + 1] = label;  //******
						//top_data[num_det * 5 + 1] = Label_Project(label);
						//std::cout << "old label: " << label << "; new label: " << Label_Project(label) << std::endl;

						top_data[num_det * 5 + 2] = bboxes[i].score();

						if (!use_normalized_bbox_) {
							OutputBBox(bboxes[i], sizes_[count_], has_resize_,
										resize_param_, &(bboxes[i]));
						}
						// Compare with each ground truth bbox.
						float overlap_max = -1;
						int jmax = -1;
						for (int j = 0; j < gt_bboxes.size(); ++j) {
							float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j],
															use_normalized_bbox_);
							if (overlap > overlap_max) {
							overlap_max = overlap;
							jmax = j;
							}
						}
						if (overlap_max >= overlap_threshold_) {
							if (evaluate_difficult_gt_ ||
								(!evaluate_difficult_gt_ && !gt_bboxes[jmax].difficult())) {
							if (!visited[jmax]) {
								// true positive.
								top_data[num_det * 5 + 3] = 1;
								top_data[num_det * 5 + 4] = 0;
								visited[jmax] = true;
							} else {
								// false positive (multiple detection).
								top_data[num_det * 5 + 3] = 0;
								top_data[num_det * 5 + 4] = 1;
							}
							}
						} 
						else {
							// false positive.
							top_data[num_det * 5 + 3] = 0;
							top_data[num_det * 5 + 4] = 1;
						}
						++num_det;
						}
					}
				}
		}
		if (sizes_.size() > 0) {
			++count_;
			if (count_ == sizes_.size()) {
			// reset count after a full iterations through the DB.
			count_ = 0;
			}
		}
	}
}

INSTANTIATE_CLASS(DetectionEvaluateLayer);
//REGISTER_LAYER_CLASS(DetectionEvaluate);

}  // namespace caffe
