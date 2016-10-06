#include <iostream>
#include <stdio.h>
#include <cmath>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "smartcrop.h"

using namespace std;
using namespace cv;

void process(unsigned char * src_img, int src_w, int src_h,
             unsigned char * dst_img, int dst_w, int dst_h);

int main(int args, char* argv[]) {
    Mat iImage  = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat oImage(Size(400, 300), CV_8UC3);

    process(iImage.ptr(), iImage.cols, iImage.rows, oImage.ptr(), oImage.cols, oImage.rows);

    imshow("", oImage );
    waitKey(0);
    return 0;
}

