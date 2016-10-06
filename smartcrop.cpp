#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;
using namespace cv;

#define EXPAND_RGB(pixel) \
    uchar b = pixel.val[0]; \
    uchar g = pixel.val[1]; \
    uchar r = pixel.val[2];

const float SATURATIONBRIGHTNESSMIN = 0.05;
const float SATURATIONBRIGHTNESSMAX = 0.9;
const float SATURATIONTHRESHOLD = 0.4;
const float SATURATIONWEIGHT = 0.3;
const float SATURATIONBIAS = 0.2;

const float SKINBRIGHTNESSMIN = 0.2;
const float SKINBRIGHTNESSMAX = 1.0;
const float SKINTHRESHOLD = 0.8;
const float SKINWEIGHT = 1.8;
const float SKINBIAS = 0.01;
const float SKINCOLOR_R = 0.78;
const float SKINCOLOR_G = 0.57;
const float SKINCOLOR_B = 0.44;

const float DETAILWEIGHT = 0.2;
const float EDGERADIUS = 0.4;
const float EDGEWEIGHT = -20.0;

const char *  FACE_CASECADE_FILE = "../../opencv/data/haarcascades/haarcascade_frontalface_default.xml";

static uchar cie(Vec3b & pixel) {
    EXPAND_RGB(pixel)
    return (uchar) (0.5126 * b + 0.7152 * g + 0.0722 * r);
}

static uchar saturation(Vec3b & pixel) {
    EXPAND_RGB(pixel)
    float maximum = MAX(r, MAX(g, b)) / 255.0;
    float minumum = MIN(r, MIN(g, b)) / 255.0;

    if (maximum == minumum) {
        return 0;
    }

    float l = (maximum + minumum) / 2;
    float d = maximum - minumum;

    return (uchar) (l > 0.5 ? d / (2 - maximum - minumum) : d / (maximum + minumum));
}

static float skin_color(Vec3b & pixel) {
    EXPAND_RGB(pixel)
    float mag = sqrt(r * r + g * g + b * b);
    float rd = (r / mag - SKINCOLOR_R);
    float gd = (g / mag - SKINCOLOR_G);
    float bd = (b / mag - SKINCOLOR_B);
    float d = sqrt(rd * rd + gd * gd + bd * bd);
    return 1 - d;
}

static void gen_crop_rect(const Size& src, const Size& ts, vector<Rect>& res) {
    const int STEP = 5;
    int w = src.width, h = src.height;

    float scale_w = ts.width / (float) w;
    float scale_h = ts.height / (float) h;
    bool hor_crop = scale_h > scale_w;

    float scale = MAX(scale_w, scale_h);

    if (hor_crop) {
        int step = (w - ts.width / scale) / STEP;
        for (int s = 0; s < STEP; s ++) {
            res.push_back(Rect(s * step, 0, ts.width / scale, ts.height / scale));
        }
    } else {
        int step = (h - ts.height / scale) / STEP;
        for (int s = 0; s < STEP; s ++) {
            res.push_back(Rect(0, s * step, ts.width / scale, ts.height / scale));
        }
    }
}

static float thirds(float x) {
    x = (fmod((x - (1 / 3) + 1.0), 2.0) * 0.5 - 0.5) * 16;
    return MAX(1.0 - x * x, 0.0);
}

static float importance(float x, float y, const Rect & crop) {
    x = (x - crop.x) / crop.width;
    y = (y - crop.y) / crop.height;
    float px = abs(0.5 - x) * 2;
    float py = abs(0.5 - y) * 2;
    // Distance from edge
    float dx = MAX(px - 1.0 + EDGERADIUS, 0);
    float dy = MAX(py - 1.0 + EDGERADIUS, 0);
    float d = (dx * dx + dy * dy) * EDGEWEIGHT;
    float s = 1.41 - sqrt(px * px + py * py);
    s += (MAX(0, s + d + 0.5) * 1.2) * (thirds(px) + thirds(py));
    return s + d;
}

static float score_rect(const Mat& o, const Rect& r) {
    float res_skin = 0;
    float res_detail = 0;
    float res_saturation = 0;

    for (int x = r.x; x < r.x + r.width; x ++) {
        for (int y = r.y; y < r.y + r.height; y ++) {
                const Vec3b& pixel = o.at<Vec3b>(y, x);
                float i = importance(x, y, r);
                EXPAND_RGB(pixel)
                float detail = g / 255.0;

                res_skin += b / 255.0 * (detail + SKINBIAS) * i;
                res_detail += detail * i;
                res_saturation += r / 255.0 * (detail + SATURATIONBIAS) * i;
        }
    }

    return (res_detail * DETAILWEIGHT + res_skin * SKINWEIGHT + res_saturation * SATURATIONWEIGHT) /
               (r.width * r.height);
}

void detect_face(Mat& i, vector<Rect> &faces) {
    // Boost the region if there is face.
    // Load Face cascade (.xml file)
    CascadeClassifier face_cascade;
    face_cascade.load(FACE_CASECADE_FILE);

    // Detect faces
    face_cascade.detectMultiScale(i, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(20, 20));
}

void process_(Mat& i, Mat& r) {
    size_t w = i.cols;
    size_t h = i.rows;
    Mat o(h, w, CV_8UC3);
    Size ts(r.cols, r.rows);

    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            Vec3b res;
            // Build the edges
            Vec3b p = i.at<Vec3b>(y, x);
            float lightness = cie(p);
            float lightness_ratio = lightness / 255.0;
            if (!(x == 0 || x >= w - 1 || y == 0 || y >= h - 1)){
                lightness = lightness * 4 -
                    cie(i.at<Vec3b>(y - 1 , x)) -
                    cie(i.at<Vec3b>(y, x - 1)) -
                    cie(i.at<Vec3b>(y, x + 1))-
                    cie(i.at<Vec3b>(y + 1, x));
            }
            res.val[1] = saturate_cast<uchar>(lightness);

            // Build the saturation
            uchar sat = saturation(p);
            res.val[2] = (sat > SATURATIONTHRESHOLD &&
                lightness_ratio >= SATURATIONBRIGHTNESSMIN &&
                lightness_ratio <= SATURATIONBRIGHTNESSMAX) ?
                ((sat - SATURATIONTHRESHOLD) * (255 / (1 - SATURATIONTHRESHOLD))) : 0;

            // Build the skin color
            float skin = skin_color(p);
            res.val[0] = (skin > SKINTHRESHOLD &&
                lightness_ratio >= SKINBRIGHTNESSMIN &&
                lightness_ratio <= SKINBRIGHTNESSMAX) ?
                ((skin - SKINTHRESHOLD) * (255 / (1 - SKINTHRESHOLD))) : 0;
            o.at<Vec3b>(y, x) = res;
        }
    }

    vector<Rect> faces;
    detect_face(i,faces);

    // Boost on the detected faces
    for( int i = 0; i < faces.size(); i++ ) {
        for (int x = faces[i].x; x < faces[i].x + faces[i].width; x ++) {
            for (int y = faces[i].y; y < faces[i].y + faces[i].height; y ++) {
                Vec3b& pixel = o.at<Vec3b>(y, x);
                EXPAND_RGB(pixel)
                o.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(b * 10),
                                        saturate_cast<uchar>(g * 10),
                                        saturate_cast<uchar>(r * 10));
            }
        }
    }

    // Scale and generate crop source rect
    vector<Rect> crop_rect;
    gen_crop_rect(Size(w, h), ts, crop_rect);

    Rect best_crop;
    float best_score = 0;
    for (int i = 0; i < crop_rect.size(); i ++) {
        float score = score_rect(o, crop_rect[i]);
#ifdef DEBUG
        printf ("score : %f, with { %d, %d, %d, %d} \n", score, crop_rect[i].x,
                crop_rect[i].y,
                crop_rect[i].width,
                crop_rect[i].height);
#endif // DEBUG
        if (score > best_score) {
            best_crop = crop_rect[i];
            best_score = score;
        }
    }

    resize(i(best_crop), r, ts, 0, 0, INTER_LINEAR);
}

void process(unsigned char * src_img, int src_w, int src_h,
             unsigned char * dst_img, int dst_w, int dst_h) {
    Mat i(src_h, src_w, CV_8UC3, src_img);
    Mat o(dst_h, dst_w, CV_8UC3, dst_img);
    process_(i, o);
}

