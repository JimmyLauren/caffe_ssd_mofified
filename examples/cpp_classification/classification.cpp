#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//added by Jimmy
#include<sstream>
#include<iostream>
#include<fstream>
#include<time.h>
#include <ctime>
#include<stdio.h>



#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace cv;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;  //��¼ÿһ����������Լ�����  

/*
���룺�ļ�������Ŀ¼��ַ(��Ŀ¼�󲻼�б��)
���������ͼƬ�ľ���·�������Ӧ��ǩ��ɵĶԵ�����
*/
vector<pair<string, int>> ReturnImagePathAndLabel(const string &img_file, const string & root_path) {
	vector<pair<string, int>> Imgpath_Label;

	std::ifstream infile;
	infile.open(img_file, ios::in);
	if (!infile) { //��ȡ�����ļ����˳�
		std::cout << "��ȡ�ļ�����" << std::endl;
		system("exit");
	}
	string temp;
	string relative_path;
	int label_per_img;
	while (getline(infile, temp)) { //��ȡһ�У�ֱ����ȡ�����е���
		std::istringstream LineBand(temp); //ת����������
		LineBand >> relative_path;
		string full_path = root_path + "/" + relative_path;
		LineBand >> label_per_img;
		//cout << full_path <<" ------------ "<<label_per_img<< endl;

		//ѹ���
		pair<string, int> kk(full_path, label_per_img);
		Imgpath_Label.push_back(kk); //ѹ��vector
	}
	return Imgpath_Label;
}

//ClassifierΪ���캯������Ҫ����ģ�ͳ�ʼ��������ѵ����ϵ�ģ�Ͳ�������ֵ�ļ��ͱ�ǩ�ļ� 
class Classifier {
public:
	Classifier(const string& model_file, //model_fileΪ����ģ��ʱ��¼����ṹ��prototxt�ļ�·��
		const string& trained_file, //trained_fileΪѵ����ϵ�caffemodel�ļ�·��  
		const string& mean_file, //mean_fileΪ��¼���ݼ���ֵ���ļ�·�������ݼ���ֵ���ļ��ĸ�ʽͨ��Ϊbinaryproto  
		const string& label_file); //label_fileΪ��¼����ǩ���ļ�·������ǩͨ����¼��һ��txt�ļ��У�һ��һ��

								   //Classify����ȥ��������ǰ�����õ�img���ڸ�����ĸ���
	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5); //����Ԥ�����ͼƬ���������

private:
	//SetMean������Ҫ���о�ֵ�趨��ÿ�ż��ͼ��������м�ȥ��ֵ�Ĳ����������ֵ������ģ��ʹ�õ����ݼ�ͼ��ľ�ֵ
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);//Predict������Classify��������Ҫ��ɲ��֣���img�����������ǰ�򴫲����õ�������� 

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);//WrapInputLayer������img��ͨ��(input_channels)�������������blob��

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels); //Preprocess����������ͼ��img��ͨ���ֿ�(input_channels) 

private:
	shared_ptr<Net<float> > net_; //net_��ʾcaffe�е����� ����ָ��
	cv::Size input_geometry_; //input_geometry_��ʾ������ͼ��ĸ߿�ͬʱҲ���������ݲ��е�ͨ��ͼ��ĸ߿�
	int num_channels_; //num_channels_��ʾ������ͼ���ͨ���� 
	cv::Mat mean_; //mean_��ʾ�����ݼ��ľ�ֵ����ʽΪMat  
	std::vector<string> labels_; //�ַ�������labels_��ʾ�˸�����ǩ  
};

//���캯��Classifier�����˸��ָ����ĳ�ʼ����������������İ�ȫ�����˼���  
Classifier::Classifier(const string& model_file, //model_fileΪ����ģ��ʱ��¼����ṹ��prototxt�ļ�·��
	const string& trained_file, //trained_fileΪѵ����ϵ�caffemodel�ļ�·��
	const string& mean_file, //mean_fileΪ��¼���ݼ���ֵ���ļ�·�������ݼ���ֵ���ļ��ĸ�ʽͨ��Ϊbinaryproto 
	const string& label_file) { //label_fileΪ��¼����ǩ���ļ�·������ǩͨ����¼��һ��txt�ļ��У�һ��һ��
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU); //���caffe��ֻ��cpu�����еģ�������ģʽ����ΪCPU
#else
	Caffe::set_mode(Caffe::GPU); //һ�����Ƕ����õ�GPUģʽ  
#endif

								 /* Load the network. */
	net_.reset(new Net<float>(model_file, TEST)); //��model_file·���µ�prototxt��ʼ������ṹ  Ϊ����ָ�����һ���µ��ڴ�
	net_->CopyTrainedLayersFrom(trained_file); //��trained_file·���µ�caffemodel�ļ�����ѵ����ϵ��������  

											   //�����ǲ���ֻ������һ��ͼ�������blob�ṹΪ(N,C,H,W)�������Nֻ��Ϊ1
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input."; //���Ե�ʱ��ֻ��һ��һ�ŵĲ���
																				 //���������blob�ṹ�������blob�ṹͬ��Ϊ(N,C,W,H)�������Nͬ��ֻ��Ϊ1
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];//��ȡ���������blob����ʾ��������ݲ� 
	num_channels_ = input_layer->channels(); //��ȡ�����ͨ���� 
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels."; //��������ͼ���ͨ�����Ƿ�Ϊ3����1������ֻ����3ͨ����1ͨ����ͼƬ
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file); //���о�ֵ������  

						/* Load labels. */
	std::ifstream labels(label_file.c_str()); //�ӱ�ǩ�ļ�·�����붨��ı�ǩ�ļ�  
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line; //line��ȡ��ǩ�ļ��е�ÿһ��(ÿһ����ǩ)  
	while (std::getline(labels, line))
		labels_.push_back(string(line)); //�����еı�ǩ����labels_

										 /*output_layerָ����������������ٸ����ӣ����ķ���������softmax���࣬�������10�࣬��ô�������blob�ͻ���10��ͨ����ÿ��ͨ���ĳ�
										 ��Ϊ1(��Ϊ��10��������10����������������10����ÿһ��ĸ��ʣ���10����֮��Ӧ��Ϊ1)�����blob�ĽṹΪ(1,10,1,1)*/
	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels()) //���������������������ͨ�����Ƿ���ڶ���ı�ǩ��ͨ����
		<< "Number of labels is different from the output layer dimension.";
}

//PairCompare�����ȽϷ���õ�����������ĳ�������ĸ��ʵĴ�С��������lhs�ĸ��ʴ�������rhs�ĸ��ʣ������棬���򷵻ؼ�
static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
/* Argmax��������ǰN���÷ָ��ʵ���� */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i))); //���շ������洢��������ÿһ����ĸ����Լ���� 
																	/*partial_sort�������ո��ʴ�Сɸѡ��pairs�и�������N����ϣ��������ǰ��ո��ʴӴ�С����pairs��ǰN��λ��*/
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);//��ǰN���ϴ�ĸ��ʶ�Ӧ��������result��  
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);//���������ǰ���䣬�õ���������ÿһ��ĸ��ʣ��洢��output��  

	N = std::min<int>(labels_.size(), N);//�ҵ���Ҫ�õ��ĸ��ʽϴ��ǰN�࣬���NӦ��С�ڵ����ܵ������Ŀ 
	std::vector<int> maxN = Argmax(output, N);//�ҵ���������ǰN�࣬�����ǰ������ɴ�С�����洢��maxN��  
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));//��labels_�ҵ�����õ��ĸ�������N���Ӧ��ʵ�ʵ�����  
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {//�������ݼ���ƽ��ֵ  
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);//�ö���ľ�ֵ�ļ�·������ֵ�ļ�����proto�� 

																 /* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);//��proto�д洢�ľ�ֵ�ļ�ת�Ƶ�blob��  
	CHECK_EQ(mean_blob.channels(), num_channels_) //�����ֵ��ͨ�����Ƿ��������ͼ���ͨ�������������ȵĻ���Ϊ�쳣  
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels; //��mean_blob�е�����ת��ΪMatʱ�Ĵ洢����  
	float* data = mean_blob.mutable_cpu_data(); //ָ���ֵblob��ָ��  
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);//�洢��ֵ�ļ���ÿһ��ͨ��ת���õ���Mat  
		channels.push_back(channel); //����ֵ�ļ�������ͨ��ת���ɵ�Matһ��һ���ش洢��channels��  
		data += mean_blob.height() * mean_blob.width(); //�ھ�ֵ�ļ����ƶ�һ��ͨ��  
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean); //���õ�������ͨ���ϳ�Ϊһ��ͼ  

							   /* Compute the global mean pixel value and create a mean image
							   * filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean); //��þ�ֵ�ļ���ÿ��ͨ����ƽ��ֵ����¼��channel_mean�� 
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean); //��������õĸ���ͨ����ƽ��ֵ��ʼ��mean_����Ϊ���ݼ�ͼ��ľ�ֵ
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];//input_layer�����������blob
													  //��ʾ����ֻ����һ��ͼ��ͼ���ͨ������num_channels_����Ϊinput_geometry_.height����Ϊinput_geometry_.width
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape(); //��ʼ������ĸ���  

	std::vector<cv::Mat> input_channels;//�洢����ͼ��ĸ���ͨ�� 
	WrapInputLayer(&input_channels);//���洢����ͼ��ĸ���ͨ����input_channels�������������blob��

	Preprocess(img, &input_channels);//��img�ĸ�ͨ���ֿ����洢��input_channels��

	net_->Forward();//���������ǰ����  

					/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0]; //output_layerָ��������������ݣ��洢����������ݵ�blob�Ĺ����(1,c,1,1)
	const float* begin = output_layer->cpu_data(); //beginָ���������ݶ�Ӧ�ĵ�һ��ĸ���  
	const float* end = begin + output_layer->channels();//endָ���������ݶ�Ӧ�����һ��ĸ���
	return std::vector<float>(begin, end);//�����������ݾ�������ǰ����������Ķ�Ӧ�ڸ�����ķ���
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0]; //input_layerָ�����������blob

	int width = input_layer->width(); //�õ�����ָ��������ͼ��Ŀ�
	int height = input_layer->height(); //�õ�����ָ��������ͼ��ĸ�
	float* input_data = input_layer->mutable_cpu_data(); //input_dataָ�����������blob
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data); //����������blob������ͬMat��������
		input_channels->push_back(channel); //�������Matͬinput_channels��������
		input_data += width * height; //һ��һ��ͨ���ز���
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img; //if-elseǶ�ױ�ʾ��Ҫ�������imgת��Ϊnum_channels_ͨ����  

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_); //������ͼ��ĳߴ�ǿ��ת��Ϊ����涨������ߴ�
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);//������ͼ��ת����Ϊ����ǰ���Ϸ������ݹ��

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized); //��ͼ���ȥ��ֵ

														  /* This operation will write the separate BGR planes directly to the
														  * input layer of the network because it is wrapped by the cv::Mat
														  * objects in input_channels. */
														  /*����ȥ��ֵ��ͼ���ɢ��input_channels�У�������WrapInputLayer�����У�
														  input_channels�Ѿ������������blob���������ˣ����������ʵ�����ǰ�ͼ�����������������blob*/
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";//����ͼ���Ƿ�������������Ϊ����
}

main(int argc, char** argv) {
	int amount = 0, err_count = 0;
	if (argc != 7) {
		/*���������в����Ƿ�Ϊ6����6�������ֱ�Ϊ
		classification�������ɵĿ�ִ���ļ���
		����ģ��ʱ��¼����ṹ��prototxt�ļ�·����
		ѵ����ϵ�caffemodel�ļ�·����
		��¼���ݼ���ֵ���ļ�·����
		��¼����ǩ���ļ�·����

		��Ҫ���������ļ���TXT�ļ������а�����������ʵ��ǩ
		��Ŀ¼��ַ�����ļ��е���Ե�ַ���ͼƬ�����ľ��Ե�ַ��*/
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " mean.binaryproto labels.txt img.jpg" << std::endl;
		return 1;
	}

	/*
	//testing function ReturnImagePathAndLabel, added by Jimmy
	string in_file = "F:/DATASETS/CarPlate/car_plate_color/self_build_test.txt";
	string root_path = "F:/DATASETS/CarPlate/car_plate_color";
	vector<pair<string, int>> image_label = ReturnImagePathAndLabel(in_file, root_path);
	for (vector<pair<string, int>>::iterator it = image_label.begin(); it != image_label.end(); it++) {
		pair<string, int> image_label_retrive = *it;
		std::cout << image_label_retrive.first << " ---- " << image_label_retrive.second << std::endl;

	}
	system("pause");
*/

	::google::InitGoogleLogging(argv[0]); //InitGoogleLogging����һЩ��ʼ��glog�Ĺ���  classification.exe

										  //ȡ�ĸ����� 
	string model_file = argv[1]; //prototxt����Э���ļ�   deploy.prototxt
	string trained_file = argv[2]; //caffemodel����ģ���ļ�    .caffemodel
	string mean_file = argv[3]; //��ֵ�ļ�                  meanfile.binary
	string label_file = argv[4]; //��ǩ�ļ�                .txt �а���0��k-1��k�����
	Classifier classifier(model_file, trained_file, mean_file, label_file); //���м������ĳ�ʼ��

	string file_txt = argv[5]; //�洢��Ҫ����ͼƬ���·����ͼƬ��ǩ���ļ�
	string root_path = argv[6];//��Ŀ¼��ַ�����ļ��е���Ե�ַ���ͼƬ�ľ��Ե�ַ

	//added by jimmy
	std::ofstream outfile; //������¼��������ļ��������������ͼƬ��Ԥ���ǩ����ʵ��ǩ��
	outfile.open("wrong_file_recoder.txt", ios::binary | ios::app | ios::in | ios::out);

	vector<pair<string, int>> image_label = ReturnImagePathAndLabel(file_txt, root_path);
	for (vector<pair<string, int>>::iterator it = image_label.begin(); it != image_label.end(); it++) {
		amount += 1;
		pair<string, int> image_label_retrive = *it;
		std::cout << image_label_retrive.first << " ---- " << image_label_retrive.second << std::endl;
		
		string img_full_path = image_label_retrive.first;  //����ͼƬ�ľ��Ե�ַ
		int true_img_label = image_label_retrive.second;     //����ͼƬ��Ӧ����ʵ��ǩ



		std::cout << "---------- Prediction for "
			<< img_full_path << " ----------" << std::endl;

		cv::Mat img = cv::imread(img_full_path, -1); //����ͼƬ 
		CHECK(!img.empty()) << "Unable to decode image " << img_full_path;
		std::vector<Prediction> predictions = classifier.Classify(img); //���������ǰ����㣬����ȡ����������ǰN���Ӧ���������

		/* Print the top N predictions. */
		for (size_t i = 0; i < predictions.size(); ++i) { //��ӡ����������ǰN�ಢ�������� 
			Prediction p = predictions[i];
			std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
				<< p.first << "\"" << std::endl;
		}

		//added by Jimmy
		Prediction No1 = predictions[0];
		stringstream stream(No1.first);
		int predicted_label;
		stream >> predicted_label;
		std::cout << "the predicted label of this image is " << predicted_label << std::endl;

		//���Ԥ��ı�ǩ����ʵ�ı�ǩ�����ϣ���Ѵ���Ԥ���ͼƬ���ơ�Ԥ���ǩ����ʵ��ǩд��TXT�ļ����Ա���������
		if (predicted_label != true_img_label) {
			err_count += 1;
			outfile << img_full_path << " " << predicted_label << " " << true_img_label << "\n";
		}
		std::cout << "===============================" << std::endl << "Accuracy is " << (float)(amount - err_count) / amount;
	}
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
