#include <ruby.h>

#if 0
void estimateHeadPose(cv::Mat& image, CameraModel& camera, const FaceDetection::Landmark& landmark)
{
    /* reference: https://qiita.com/TaroYamada/items/e3f3d0ea4ecc0a832fac */
    /* reference: https://github.com/spmallick/learnopencv/blob/master/HeadPose/headPose.cpp */
    static const std::vector<cv::Point3f> face_object_point_list = {
        {    0.0f,    0.0f,    0.0f }, // Nose
        { -225.0f,  170.0f, -135.0f }, // Left eye
        {  225.0f,  170.0f, -135.0f }, // Right eye
        { -150.0f, -150.0f, -125.0f }, // Left lip
        {  150.0f, -150.0f, -125.0f }, // Right lip
    };
    
    static const std::vector<cv::Point3f> face_object_point_for_pnp_list = {
        face_object_point_list[0], face_object_point_list[0], // Nose
        face_object_point_list[1], face_object_point_list[1], // Left eye
        face_object_point_list[2], face_object_point_list[2], // Right eye
        face_object_point_list[3], face_object_point_list[3], // Left lip
        face_object_point_list[4], face_object_point_list[4], // Righ lip
    };

    std::vector<cv::Point2f> face_image_point_list;
    face_image_point_list.push_back(landmark[2]);  // Nose
    face_image_point_list.push_back(landmark[2]);  // Nose
    face_image_point_list.push_back(landmark[0]);  // Left eye
    face_image_point_list.push_back(landmark[0]);  // Left eye
    face_image_point_list.push_back(landmark[1]);  // Right eye
    face_image_point_list.push_back(landmark[1]);  // Right eye
    face_image_point_list.push_back(landmark[3]);  // Left lip
    face_image_point_list.push_back(landmark[3]);  // Left lip
    face_image_point_list.push_back(landmark[4]);  // Right lip
    face_image_point_list.push_back(landmark[4]);  // Right lip

    cv::Mat rvec = cv::Mat_<float>(3, 1);
    cv::Mat tvec = cv::Mat_<float>(3, 1);
    cv::solvePnP(face_object_point_for_pnp_list, face_image_point_list, camera.K, camera.dist_coeff, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    char text[128];
    snprintf(text, sizeof(text), "Pitch = %-+4.0f, Yaw = %-+4.0f, Roll = %-+4.0f", Rad2Deg(rvec.at<float>(0, 0)), Rad2Deg(rvec.at<float>(1, 0)), Rad2Deg(rvec.at<float>(2, 0)));
    CommonHelper::drawText(image, text, cv::Point(10, 10), 0.7, 3, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255), false);
    
    std::vector<cv::Point3f> nose_end_point3D = { { 0.0f, 0.0f, 500.0f } };
    std::vector<cv::Point2f> nose_end_point2D;
    cv::projectPoints(nose_end_point3D, rvec, tvec,
		      camera.K, camera.dist_coeff, nose_end_point2D);
    cv::arrowedLine(image, face_image_point_list[0], nose_end_point2D[0],
		    cv::Scalar(0, 255, 0), 5);

    
    /* Calculate Euler Angle */
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat projMat = (cv::Mat_<double>(3, 4) <<
        R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2), 0,
        R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2), 0,
        R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2), 0);

    cv::Mat K, R2, tvec2, R_x, R_y, R_z, eulerAngles;
    cv::decomposeProjectionMatrix(projMat, K, R, tvec2,
				  R_x, R_y, R_z, eulerAngles);
    double pitch = eulerAngles.at<double>(0, 0);
    double yaw   = eulerAngles.at<double>(1, 0);
    double roll  = eulerAngles.at<double>(2, 0);
    snprintf(text, sizeof(text), "X = %-+4.0f, Y = %-+4.0f, Z = %-+4.0f", pitch, yaw, roll);
    CommonHelper::drawText(image, text, cv::Point(10, 40), 0.7, 3, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255), false);
}
#endif

#if 0
    CameraModel camera;
    camera.SetIntrinsic(image_input.cols, image_input.rows, FocalLength(image_input.cols, kFovDeg));
    camera.SetDist({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
    camera.SetExtrinsic({ 0.0f, 0.0f, 0.0f },    /* rvec [deg] */
			{ 0.0f, 0.0f, 0.0f }, true);   /* tvec (in world coordinate) */

    estimateHeadPose(image_input, camera, face_list[i].second);

#endif

#define IF_UNDEF(a, b)                          \
    ((a) == Qundef) ? (b) : (a)


static VALUE cCirce       = Qundef;
static VALUE eCirceError  = Qundef;

#include <tuple>
#include <string>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

#include "yolo.h"
#include "yunet.h"

using namespace std;
using namespace cv;



// Text parameters.
const float FONT_SCALE = 0.7;
const int   FONT_FACE  = FONT_HERSHEY_SIMPLEX;
const int   THICKNESS  = 1;

// Colors.
cv::Scalar BLACK  = cv::Scalar(0,0,0);
cv::Scalar BLUE   = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED    = cv::Scalar(0,0,255);


static ID id_debug;
static ID id_face;
static ID id_classify;
static ID id_class;
static ID id_png;
static ID id_jpg;

static ID id_label;
static ID id_thickness;
static ID id_extra;
static ID id_color;

static Yolo  *yolo;
static YuNet *yunet;



static void
draw_label(cv::Mat& img, string label, Point& origin,
	   Scalar& fgcolor, Scalar& bgcolor, int thickness = 1)
{
    int baseLine;
    cv::Size label_size =
	cv::getTextSize(label, FONT_FACE, FONT_SCALE, thickness, &baseLine);
    
    Point a = { origin.x, max(origin.y, label_size.height) };
    Point b = { a.x + label_size.width,
		a.y + label_size.height + baseLine };
    Point t = { a.x, a.y + label_size.height };
    
    cv::rectangle(img, a, b, bgcolor, cv::FILLED);
    cv::putText(img, label, t, FONT_FACE, FONT_SCALE, fgcolor, THICKNESS);
}

static void
draw_box(cv::Mat& img, Rect& box,
	 Scalar& framecolor = BLUE, int thickness = 1) {

    Point a = { box.x,             box.y              };
    Point b = { box.x + box.width, box.y + box.height };

    cv::rectangle(img, a, b, framecolor, thickness);
}

static void
draw_labelbox(cv::Mat& img, string label, Rect& box,
	      Scalar& framecolor = BLUE, Scalar& textcolor = BLACK,
	      int thickness = 1) {

    Point o = { box.x, box.y };

    draw_box(img, box, framecolor, thickness);
    draw_label(img, label, o, textcolor, framecolor);    
}


VALUE
circe_annotate(Mat& img, Rect& box, VALUE v_annotation, int *state) {
    if (img.empty() || NIL_P(v_annotation))
        return Qnil;

    VALUE v_label      = Qnil;
    VALUE v_color      = ULONG2NUM(0x0000ff);
    VALUE v_thickness  = INT2NUM(1);
    VALUE v_extra      = Qtrue;
  
    VALUE s_label      = rb_id2sym(id_label);
    VALUE s_color      = rb_id2sym(id_color);
    VALUE s_thickness  = rb_id2sym(id_thickness);
    VALUE s_extra      = rb_id2sym(id_extra);
    
    switch (TYPE(v_annotation)) {
    case T_NIL:
        break;
    case T_HASH:
        v_thickness = rb_hash_lookup2(v_annotation, s_thickness, v_thickness);
	v_color     = rb_hash_lookup2(v_annotation, s_color,     v_color    );
	v_label     = rb_hash_lookup2(v_annotation, s_label,     v_label    );
	v_extra     = rb_hash_lookup2(v_annotation, s_extra,     v_extra    );
	break;
    case T_ARRAY:
        switch(RARRAY_LENINT(v_annotation)) {
	default:
	case 3: v_thickness = RARRAY_AREF(v_annotation, 2);
	case 2: v_color     = RARRAY_AREF(v_annotation, 1);
	case 1: v_label     = RARRAY_AREF(v_annotation, 0);
	case 0: break;
	}
	break;
    case T_STRING:
        v_label = v_annotation;
	break;
    }
  
    // No color, no rendering
    if (NIL_P(v_color))
        return Qnil;
  
    long   rgb   = NUM2ULONG(v_color);
    Scalar color = cv::Scalar((rgb >>  0) & 0xFF,
			      (rgb >>  8) & 0xFF,
			      (rgb >> 16) & 0xFF);
  
    if (! NIL_P(v_thickness)) {
        int thickness = NUM2INT(v_thickness);
	draw_box(img, box, color, thickness);
    }
    if (! NIL_P(v_label)) {
        string label  = StringValueCStr(v_label);
	Point  o      = { box.x, box.y };
	draw_label(img, label, o, BLACK, color);
    }

    // Return normalized parameters
    VALUE r = rb_hash_new();
    rb_hash_aset(r, s_label,     v_label    );
    rb_hash_aset(r, s_color,     v_color    );
    rb_hash_aset(r, s_thickness, v_thickness);
    rb_hash_aset(r, s_extra,     v_extra    );
    return r;
}



void
yunet_process_features(cv::Mat& faces, Mat& img, VALUE v_features, int *state)
{
    for (int i = 0; i < faces.rows; i++) {
	// Face
	int x_f   = static_cast<int>(faces.at<float>(i,  0));
        int y_f   = static_cast<int>(faces.at<float>(i,  1));
        int w_f   = static_cast<int>(faces.at<float>(i,  2));
        int h_f   = static_cast<int>(faces.at<float>(i,  3));
	// Right eye
	int x_re  = static_cast<int>(faces.at<float>(i,  4));
	int y_re  = static_cast<int>(faces.at<float>(i,  5));
	// Left eye
        int x_le  = static_cast<int>(faces.at<float>(i,  6));
	int y_le  = static_cast<int>(faces.at<float>(i,  7));
	// Nose tip
        int x_nt  = static_cast<int>(faces.at<float>(i,  8));
	int y_nt  = static_cast<int>(faces.at<float>(i,  9));
        // Right corner mouth
        int x_rcm = static_cast<int>(faces.at<float>(i, 10));
	int y_rcm = static_cast<int>(faces.at<float>(i, 11));
        // Left corner mouth
	int x_lcm = static_cast<int>(faces.at<float>(i, 12));
	int y_lcm = static_cast<int>(faces.at<float>(i, 13));
	// Confidence
	float confidence = faces.at<float>(i, 14);
	
        VALUE v_type       = ID2SYM(id_face);
	VALUE v_box        = rb_ary_new_from_args(4,
				INT2NUM(x_f), INT2NUM(y_f),
				INT2NUM(w_f), INT2NUM(h_f));
	VALUE v_landmark   = rb_ary_new_from_args(5,
			        rb_ary_new_from_args(2, INT2NUM(x_re),
						        INT2NUM(y_re)),
			        rb_ary_new_from_args(2, INT2NUM(x_le),
						        INT2NUM(y_le)),
			        rb_ary_new_from_args(2, INT2NUM(x_nt),
						        INT2NUM(y_nt)),
			        rb_ary_new_from_args(2, INT2NUM(x_rcm),
						        INT2NUM(y_rcm)),
				rb_ary_new_from_args(2, INT2NUM(x_lcm),
							INT2NUM(y_lcm)));
	VALUE v_confidence = DBL2NUM(confidence);
	VALUE v_feature    = rb_ary_new_from_args(4, v_type, v_box, v_landmark,
						  v_confidence);

	rb_ary_push(v_features, v_feature);

	
	if (!img.empty() && rb_block_given_p()) {
	    cv::Rect box       = cv::Rect(x_f, y_f, w_f, h_f);
	    VALUE v_annotation = rb_yield_splat(v_feature);
	    VALUE cfg          = circe_annotate(img, box, v_annotation, state);
	    VALUE s_extra      = rb_id2sym(id_extra);

	    if (!NIL_P(cfg) && RTEST(rb_hash_aref(cfg, s_extra))) {
		cv::Scalar color = cv::Scalar(255, 0, 0);
		cv::circle(img, cv::Point(x_le,  y_le ), 3, color, 2);
		cv::circle(img, cv::Point(x_re,  y_re ), 3, color, 2);
		cv::circle(img, cv::Point(x_nt,  y_nt ), 3, color, 2);
		cv::circle(img, cv::Point(x_rcm, y_rcm), 3, color, 2);
		cv::circle(img, cv::Point(x_lcm, y_lcm), 3, color, 2);
	    }
	}
    }
}



void
yolo_process_features(vector<Yolo::Item>& items,
		      Mat& img, VALUE v_features, int *state)
{   
    for (int i = 0; i < items.size(); i++) {
	string name        = std::get<0>(items[i]);
	float  confidence  = std::get<1>(items[i]);
	Rect   box         = std::get<2>(items[i]);
	
	VALUE v_type       = ID2SYM(id_class);
	VALUE v_box        = rb_ary_new_from_args(4,
				  INT2NUM(box.x    ), INT2NUM(box.y     ),
				  INT2NUM(box.width), INT2NUM(box.height));
	VALUE v_name       = rb_str_new(name.c_str(), name.size());
	VALUE v_confidence = DBL2NUM(confidence);
	VALUE v_feature    = rb_ary_new_from_args(4, v_type, v_box,
						     v_name, v_confidence);
	rb_ary_push(v_features, v_feature);

	if (!img.empty() && rb_block_given_p()) {
	    VALUE v_annotation = rb_yield_splat(v_feature);
	    circe_annotate(img, box, v_annotation, state);
	}
    }
}



static VALUE
circe_m_analyze(int argc, VALUE* argv, VALUE self) {
    // Retrieve arguments
    VALUE v_imgstr, v_format, v_opts;
    VALUE kwargs[3];
    rb_scan_args(argc, argv, "11:", &v_imgstr, &v_format, &v_opts);
    rb_get_kwargs(v_opts, (ID[]){ id_debug, id_face, id_classify },
		  0, 3, kwargs);
    VALUE v_debug    = IF_UNDEF(kwargs[0], Qfalse);
    VALUE v_face     = IF_UNDEF(kwargs[1], Qfalse);
    VALUE v_classify = IF_UNDEF(kwargs[2], Qfalse);

    VALUE v_features = rb_ary_new();
    VALUE v_image    = Qnil;
    
    if (! NIL_P(v_format)) {
	Check_Type(v_format, T_SYMBOL);
	ID i_format = rb_sym2id(v_format);
	if ((i_format != id_png) && (i_format != id_jpg))
	    rb_raise(rb_eArgError, "format must be :png, :jpg or nil");
    }
    
    if (!RTEST(v_face) && !RTEST(v_classify)) {
	v_face = v_classify = Qtrue;
    }


    // Load image.
    Mat i_img = cv::imdecode(cv::Mat(1, RSTRING_LEN(v_imgstr), CV_8UC1,
				     (unsigned char *)RSTRING_PTR(v_imgstr)),
			     IMREAD_UNCHANGED);	    
    Mat o_img = NIL_P(v_format) ? cv::Mat() : i_img.clone();

    // Processing
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    std::chrono::duration<double> duration;
    int state = 0;

    start_time = std::chrono::system_clock::now();
    
    if (RTEST(v_classify)) {
	vector<Yolo::Item> items;
	yolo->process(i_img, items);
	yolo_process_features(items, o_img, v_features, &state);
	if (state) goto exception;
    }

    if (RTEST(v_face)) {
	cv::Mat faces;
	yunet->process(i_img, faces);
	yunet_process_features(faces, o_img, v_features, &state);
	faces.release();
	if (state) goto exception;
    }

    end_time = std::chrono::system_clock::now();
    duration = end_time - start_time;
    

    if (! NIL_P(v_format)) {
	if (RTEST(v_debug)) {
	    double ms    = duration / 1.0ms;
	    string label = cv::format("Inference time : %0.2f ms", ms);
	    cv::putText(o_img, label, Point(20, 40),
			FONT_FACE, FONT_SCALE, RED);
	}

	ID   i_format = rb_sym2id(v_format);
	string format;
	if      (i_format == id_png) { format = ".png"; }
	else if (i_format == id_jpg) { format = ".jpg"; }

	std::vector<uchar> buf;	
	cv::imencode(format, o_img, buf);
	v_image = rb_str_new(reinterpret_cast<char*>(buf.data()), buf.size());
	buf.clear();
    }

    i_img.release();
    o_img.release();

    return rb_ary_new_from_args(2, v_features, v_image);

 exception:
    i_img.release();
    o_img.release();
    rb_jump_tag(state);
}




extern "C"
void Init_core(void) {
    /* Main classes */
    cCirce      = rb_define_class("Circe", rb_cObject);
    eCirceError = rb_define_class_under(cCirce, "Error", rb_eStandardError);
    // myclass = rb_const_get(mymodule, sym_myclass);

    VALUE v_onnx_yolo   = rb_const_get(cCirce, rb_intern("ONNX_YOLO"));
    VALUE v_yolo_path   = RARRAY_AREF(v_onnx_yolo, 0);
    VALUE v_yolo_height = RARRAY_AREF(v_onnx_yolo, 1);
    VALUE v_yolo_width  = RARRAY_AREF(v_onnx_yolo, 2);

    VALUE v_onnx_yunet  = rb_const_get(cCirce, rb_intern("ONNX_YUNET"));
    VALUE v_yunet_path  = RARRAY_AREF(v_onnx_yunet, 0);


    
    static Yolo  _yolo  = { StringValueCStr(v_yolo_path ),
			    { NUM2INT(v_yolo_width),
			      NUM2INT(v_yolo_height) }};
    static YuNet _yunet = { StringValueCStr(v_yunet_path) };

    yolo  = &_yolo;
    yunet = &_yunet;

    
    id_debug       = rb_intern_const("debug"    );
    id_face        = rb_intern_const("face"     );
    id_classify    = rb_intern_const("classify" );
    id_class       = rb_intern_const("class"    );
    id_png         = rb_intern_const("png"      );
    id_jpg         = rb_intern_const("jpg"      );
    id_label       = rb_intern_const("label"    );
    id_thickness   = rb_intern_const("thickness");
    id_extra       = rb_intern_const("extra"    );
    id_color       = rb_intern_const("color"    );
    
    
    rb_define_method(cCirce, "analyze", circe_m_analyze, -1);
}
