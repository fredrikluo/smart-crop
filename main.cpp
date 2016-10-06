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

const float SKINBRIGHTNESSMIN = 0.2;
const float SKINBRIGHTNESSMAX = 1.0;
const float SKINTHRESHOLD = 0.8;

const float SKINCOLOR_R = 0.78;
const float SKINCOLOR_G = 0.57;
const float SKINCOLOR_B = 0.44;


static uchar cie(Vec3b & pixel) {
    EXPAND_RGB(pixel)
    return (uchar) (0.5126 * b + 0.7152 * g + 0.0722 * r);
}

static uchar saturation(Vec3b & pixel) {
    EXPAND_RGB(pixel)
    float maximum = MAX(r / 255.0, MAX(g / 255.0, b / 255.0));
    float minumum = MIN(r / 255.0, MIN(g / 255.0, b / 255.0));

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

void process(Mat& i, Mat& o) {
    size_t w = i.cols;
    size_t h = i.rows;

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

            // Boost the region if there is face.
            o.at<Vec3b>(y, x) = res;
        }
    }
}

/*
void applyBoosts(options, output) {
    if (!options.boost) return;
    var od = output.data;
    for (var i = 0; i < output.width; i += 4) {
        od[i + 3] = 0;
    }
    for (i = 0; i < options.boost.length; i++) {
        applyBoost(options.boost[i], options, output);
    }
}

void applyBoost(boost, options, output) {
    var od = output.data;
    var w = output.width;
    var x0 = ~~boost.x;
    var x1 = ~~(boost.x + boost.width);
    var y0 = ~~boost.y;
    var y1 = ~~(boost.y + boost.height);
    var weight = boost.weight * 255;
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            var i = (y * w + x) * 4;
            od[i + 3] += weight;
        }
    }
}
*/
int main(int args, char* argv[]) {
    Mat im;
    im = imread(argv[2], CV_LOAD_IMAGE_COLOR);  

    Mat image(Size(im.cols, im.rows),CV_8UC3);
    process(im, image);

    // Load Face cascade (.xml file)
    CascadeClassifier face_cascade;
    face_cascade.load(argv[1]);
    //
    //             // Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    //                         // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( image, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    }

    namedWindow( "window1", 1 );
    imshow( "Detected Face", image );

    waitKey(0);                   
    return 0;
}

