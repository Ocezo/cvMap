// C++ program to build a handwritten character dataset
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;

void extractROIs(const Mat& image, const vector<float>& horizontal_rhos, const vector<float>& vertical_rhos, bool reject, vector<Mat>& rois);
void augmentROIs(vector<Mat>& rois, unsigned int scale);
bool isRoiEmpty(const Mat& roi, double varianceThreshold, bool info);
Mat cropRoi(const Mat& roi, int t);
Mat warpRoi(const Mat& roi, double theta_deg, double phi_deg, double l1, double l2, double tx, double ty, bool remap);
Mat resizeRoi(const Mat& img, int outSize);
Mat remapToBinary(const Mat& roi, int num);
vector<double> linspace(double start, double end, int num);
Mat binToBinary(const Mat& roi, int num);

int main(int argc, char* argv[])
{
    const int num = 32;                     // final binary image resolution
    const unsigned int scale_factor = 10;   // dataset expansion factor: 1..10
    const bool reject = true;               // reject empty samples
    const size_t char_per_line = 140;       // number of labels per line

    struct DatasetSpec {
        string filename;
        string tag;
        char even_label;
        char odd_label;
    };

    const vector<DatasetSpec> datasets = {
        {"0n1s.jpg", "0n1s", '0', '1'},
        {"2n3s.jpg", "2n3s", '2', '3'},
        {"4n5s.jpg", "4n5s", '4', '5'},
        {"6n7s.jpg", "6n7s", '6', '7'},
        {"8n9s.jpg", "8n9s", '8', '9'}
    };

    ofstream labelsFile("../img/out/labels.txt");
    if (!labelsFile.is_open())
    {
        cerr << "Failed to open labels.txt." << endl;
        return -1;
    }

    size_t global_roi_index = 0;
    size_t written_labels = 0;

    for (const DatasetSpec& dataset : datasets)
    {
        Mat src = imread("../img/in/figures/" + dataset.filename, IMREAD_GRAYSCALE);
        if (src.empty())
        {
            cerr << "Failed to load image: " << dataset.filename << endl;
            return -1;
        }

        Mat srcColor, srcColor2;

        // 1/ Harris corners
        Mat harrisResponse;
        cvtColor(src, srcColor, COLOR_GRAY2BGR);
        cornerHarris(src, harrisResponse, 2, 3, 0.04);

        Mat harrisNorm;
        normalize(harrisResponse, harrisNorm, 0, 255, NORM_MINMAX, CV_32FC1);

        for (int y = 0; y < harrisNorm.rows; y++)
        {
            for (int x = 0; x < harrisNorm.cols; x++)
            {
                if ((int)harrisNorm.at<float>(y, x) > 100) // /!\ Adjustable threshold: 80 -> 160
                {
                    circle(srcColor, Point(x, y), 1, Scalar(0, 0, 255), FILLED); // Red
                }
            }
        }
        imwrite("../img/out/harris/harris_" + dataset.tag + ".jpg", srcColor);

        // 2/ Hough lines
        double h = src.size().height;
        double w = src.size().width;
        cvtColor(src, srcColor2, COLOR_GRAY2BGR);

        Mat edges;
        Canny(src, edges, 50, 150);

        vector<Vec2f> lines;
        vector<float> horizontal_rhos, vertical_rhos;
        HoughLines(edges, lines, 1, CV_PI / 180, 500); // /!\ Adjustable threshold: 100 -> 500

        for (size_t i = 0; i < lines.size(); i++)
        {
            float rho = lines[i][0], theta = lines[i][1];
            if (abs(theta) < CV_PI / 36 || abs(theta - CV_PI) < CV_PI / 36)
            {
                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                Point pt1(cvRound(x0 + h * (-b)), cvRound(y0 + h * (a)));
                Point pt2(cvRound(x0 - h * (-b)), cvRound(y0 - h * (a)));
                line(srcColor2, pt1, pt2, Scalar(0, 255, 0), 1, LINE_AA);
                vertical_rhos.push_back(rho);
            }
            if (abs(theta - CV_PI / 2) < CV_PI / 36)
            {
                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                Point pt1(cvRound(x0 + w * (-b)), cvRound(y0 + w * (a)));
                Point pt2(cvRound(x0 - w * (-b)), cvRound(y0 - w * (a)));
                line(srcColor2, pt1, pt2, Scalar(0, 255, 255), 1, LINE_AA);
                horizontal_rhos.push_back(rho);
            }
        }
        imwrite("../img/out/lines/lines_" + dataset.tag + ".jpg", srcColor2);

        sort(horizontal_rhos.begin(), horizontal_rhos.end());
        sort(vertical_rhos.begin(), vertical_rhos.end());

        // 3/ ROI extraction
        vector<Mat> rois;
        extractROIs(src, horizontal_rhos, vertical_rhos, reject, rois);

        // 4/ ROI augmentation
        augmentROIs(rois, scale_factor); // scale factor between 1 and 10

        // 5/ Save ROI samples and labels
        for (size_t k = 0; k < rois.size(); ++k) {
            const size_t l = global_roi_index++;
            imwrite("../img/out/rois/roi_" + to_string(l) + ".jpg", rois[k]);

            // Mat binary = binToBinary(rois[k], num);
            // imwrite("../img/out/binning/roi_nxn_" + to_string(l) + ".jpg", binary);

            Mat binary = remapToBinary(rois[k], num);
            imwrite("../img/out/binning/roi_nxn_" + to_string(l) + ".jpg", binary);

            labelsFile << ((k % 2 == 0) ? dataset.even_label : dataset.odd_label);
            ++written_labels;
            if (written_labels % char_per_line == 0) {
                labelsFile << '\n';
            }
        }
    }

    if (written_labels % char_per_line != 0) {
        labelsFile << '\n';
    }
    labelsFile.close();

    waitKey(0);
    return 0;
}

void extractROIs(const Mat& image, const vector<float>& horizontal_rhos, const vector<float>& vertical_rhos, bool reject, vector<Mat>& rois)
{
    // We assume horizontal_rhos and vertical_rhos are already sorted.
    int height, width;

    // int k = 0;
    // Loop over line pairs to detect squares and extract ROIs
    for (size_t i = 0; i < horizontal_rhos.size() - 1; ++i) {
        for (size_t j = 0; j < vertical_rhos.size() - 1; ++j) {
            // Compute the corners of the square formed by two horizontal and two vertical lines
            Point topLeft(cvRound(vertical_rhos[j]), cvRound(horizontal_rhos[i]));
            Point bottomRight(cvRound(vertical_rhos[j + 1]), cvRound(horizontal_rhos[i + 1]));

            // Check image boundaries
            if (topLeft.x < 0 || topLeft.y < 0 || bottomRight.x > image.cols || bottomRight.y > image.rows) {
                continue; // Ignore squares outside the image bounds
            }

            height = bottomRight.y - topLeft.y;
            width = bottomRight.x - topLeft.x;

            // cout << "height = " << height << " - width = " << width << endl;

            // Filter squares close to 56 x 56 px
            if (height < 53 || height > 61) {
                continue; // Ignore squares with invalid height
            }

            if (width < 53 || width > 61) {
                continue; // Ignore squares with invalid width
            }

            // cout << "k = " << k << " - height = " << height << " - width = " << width << endl;
            // k++;

            // Extract the ROI corresponding to the detected square
            Rect roi(topLeft, bottomRight);
            Mat imagette = image(roi).clone();

            Mat imagetteCropped = cropRoi(imagette, 5);
            // imwrite("../img/out/current_roi.jpg", imagetteCropped);

            // Reject "empty" ROI samples when the flag is enabled
            if (reject) {
                if (isRoiEmpty(imagetteCropped, 100.0, false)) { // /!\ Adjustable threshold
                    continue;
                }
            }

            rois.push_back(imagetteCropped); // Add the ROI sample to the list
        }
    }
}

void augmentROIs(vector<Mat>& rois, unsigned int scale)
{
    // Augment ROIs by translation and warping
    scale = clamp(scale, 1u, 10u); // scale factor between 1 and 10

    vector<Mat> rois_aug;
    rois_aug.reserve(rois.size() * scale);

    if (rois.size() % 2 != 0) {
        cerr << "Warning: rois size is not even!\n";
    }

    for (size_t k = 0; k < rois.size() / 2; ++k)
    {
        const Mat& roi_even = rois[2*k];
        const Mat& roi_odd  = rois[2*k + 1];

        Mat r;

        // x1 Original
        if (scale > 0) {
            rois_aug.push_back(roi_even);
            rois_aug.push_back(roi_odd);
        }

        // x2 Right translation
        if (scale > 1) {
            rois_aug.push_back(warpRoi(roi_even, 0,0,1,1, 4,0,false));
            rois_aug.push_back(warpRoi(roi_odd , 0,0,1,1, 4,0,false));
        }

        // x3 Left translation
        if (scale > 2) {
            rois_aug.push_back(warpRoi(roi_even, 0,0,1,1,-4,0,false));
            rois_aug.push_back(warpRoi(roi_odd , 0,0,1,1,-4,0,false));
        }

        // x4 Downward translation
        if (scale > 3) {
            rois_aug.push_back(warpRoi(roi_even, 0,0,1,1,0,4,false));
            rois_aug.push_back(warpRoi(roi_odd , 0,0,1,1,0,4,false));
        }

        // x5 Upward translation
        if (scale > 4) {
            rois_aug.push_back(warpRoi(roi_even, 0,0,1,1,0,-4,false));
            rois_aug.push_back(warpRoi(roi_odd , 0,0,1,1,0,-4,false));
        }

        // x6 Strong right warp
        if (scale > 5) {
            r = warpRoi(roi_even, 20,20,1.2,0.8,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
            r = warpRoi(roi_odd, 20,20,1.2,0.8,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
        }

        // x7 Light right warp
        if (scale > 6) {
            r = warpRoi(roi_even, 10,10,1,1,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
            r = warpRoi(roi_odd, 10,10,1,1,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
        }

        // x8 Strong left warp
        if (scale > 7) {
            r = warpRoi(roi_even, -20,-70,0.8,1.2,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
            r = warpRoi(roi_odd, -20,-70,0.8,1.2,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
        }

        // x9 Light left warp
        if (scale > 8) {
            r = warpRoi(roi_even, -10,-35,1,1,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
            r = warpRoi(roi_odd, -10,-35,1,1,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
        }

        // x10 Warp central
        if (scale > 9) {
            r = warpRoi(roi_even, -5,5,1,1,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
            r = warpRoi(roi_odd, -5,5,1,1,0,0,true);
            rois_aug.push_back(resizeRoi(r,48));
        }
    }

    rois = move(rois_aug);
}

bool isRoiEmpty(const Mat& roi, double varianceThreshold, bool info) {
    // Compute the pixel intensity variance
    Mat mean, stddev;
    meanStdDev(roi, mean, stddev);
    double variance = stddev.at<double>(0) * stddev.at<double>(0);

    if (info) {
        cout << "mean = " << mean.at<double>(0) << " - variance = " << variance << endl;
    }

    // Variance threshold used to identify "empty" images
    return variance < varianceThreshold;
}

Mat cropRoi(const Mat& roi, int t) {
    // Check that the ROI is large enough to be cropped
    if (roi.rows <= 2 * t || roi.cols <= 2 * t) {
        cerr << "The ROI is too small to be cropped by 't' pixels on each side." << endl;
        return roi; // Return the original if it is too small
    }

    // Define the cropping region
    Rect croppedRegion(t, t, roi.cols - 2 * t, roi.rows - 2 * t);

    // Return the cropped ROI sample
    return roi(croppedRegion);
}

// 2x2 rotation matrix
Matx22d R(double a)
{
    double c = cos(a);
    double s = sin(a);
    return Matx22d(c, -s,
                   s,  c);
}

Mat warpRoi(const Mat& roi, double theta_deg, double phi_deg,
                            double l1, double l2,
                            double tx, double ty,
                            bool remap = true)
{
    // Deform a ROI using an affine transformation
    double theta = theta_deg * CV_PI / 180.0;   // global rotation
    double phi   = phi_deg   * CV_PI / 180.0;   // principal axis orientation

    // 2x2 affine matrix
    Matx22d A = R(theta) * R(-phi) * Matx22d(l1, 0,
                                             0,  l2) * R(phi);

    // 2x3 warp matrix with translations
    Matx23d M(A(0,0), A(0,1), tx,
              A(1,0), A(1,1), ty);

    Size dsize;

    if (!remap) {
        // No output size adjustment
        dsize = roi.size();
    }
    else {
        // Adjust the output size based on the 4 ROI corners
        vector<Point2d> corners = {
            {0.0, 0.0},
            {(double)roi.cols, 0.0},
            {(double)roi.cols, (double)roi.rows},
            {0.0, (double)roi.rows}
        };

        auto apply = [&](const Point2d& p) -> Point2d {
            return {
                M(0,0)*p.x + M(0,1)*p.y + M(0,2),
                M(1,0)*p.x + M(1,1)*p.y + M(1,2)
            };
        };

        double minX =  1e18, minY =  1e18;
        double maxX = -1e18, maxY = -1e18;
        for (const auto& c : corners) {
            Point2d q = apply(c);
            minX = min(minX, q.x);
            minY = min(minY, q.y);
            maxX = max(maxX, q.x);
            maxY = max(maxY, q.y);
        }

        int margin = 2; // safety border in px

        // Output size = bbox + margin
        int outW = (int)ceil(maxX - minX) + 2*margin;
        int outH = (int)ceil(maxY - minY) + 2*margin;
        outW = max(outW, 1);
        outH = max(outH, 1);

        // Shift so that minX/minY becomes margin
        M(0,2) += (-minX + margin);
        M(1,2) += (-minY + margin);

        // Output size:
        dsize = Size(outW, outH);
    }

    // Border: fill in with white background
    Mat dst;
    warpAffine(
        roi, dst, Mat(M), dsize,
        INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar(255)
    );

    // Return the warped ROI sample
    return dst;
}

Mat resizeRoi(const cv::Mat& img, int outSize = 48)
{
    // Crop the letter blob and resize it to outSize x outSize
    CV_Assert(img.type() == CV_8UC1);

    // Invert threshold: letter becomes white (255), background black (0)
    Mat bin;
    threshold(img, bin, 240, 255, THRESH_BINARY_INV);

    vector<Point> pts;
    findNonZero(bin, pts);

    if (pts.empty()) {
        // No blob: return a white tile
        return Mat(outSize, outSize, CV_8UC1, Scalar(255));
    }

    Rect bb = boundingRect(pts);

    // New bounding box ratio : 1/5 + 3/5 + 1/5
    Rect bb_ = bb;
    bb_.width = (int)round(1.67 * bb.width);
    bb_.height = (int)round(1.67 * bb.height);
    bb_.x = max(1, (int)round(bb.x - 0.2 * bb_.width));
    bb_.y = max(1, (int)round(bb.y - 0.2 * bb_.height));
    bb_.width = min(bb_.width, img.cols - bb_.x - 1);
    bb_.height = min(bb_.height, img.rows - bb_.y - 1);

    Mat crop = img(bb_).clone();

    Mat resized;
    resize(crop, resized, Size(outSize, outSize), 0, 0, INTER_LINEAR);

    return resized;
}

Mat remapToBinary(const Mat& roi, int num) {
    Mat resized, binary;
    
    // Resize to num x num pixels
    resize(roi, resized, Size(num, num), 0, 0, INTER_AREA);

    // Binarization
    threshold(resized, binary, 230, 255, THRESH_BINARY);

    return binary;
}

vector<double> linspace(double start, double end, int num) {
    vector<double> vec;
    if (num == 1) {
        vec.push_back(start);
    } else {
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i) {
            vec.push_back(start + i * step);
        }
    }
    return vec;
}

// Bin ROIs into a num x num grid
Mat binToBinary(const Mat& roi, int num) {
    // Mat roiColor;
    Mat binary = Mat::zeros(num, num, CV_8U);

    double h_roi = roi.size().height;
    double w_roi = roi.size().width;

    vector<double> verticals = linspace(0, w_roi, num + 1);
    vector<double> horizontals = linspace(0, h_roi, num + 1);

    // cvtColor(roi, roiColor, COLOR_GRAY2BGR);

    // for (int i = 0; i < verticals.size(); ++i) {
    //     Point pt1(verticals[i], 0.5);
    //     Point pt2(verticals[i], h_roi + 0.5);
    //     line(roiColor, pt1, pt2, Scalar(0, 255, 0), 1, LINE_AA); // green line
    // }

    // for (int i = 0; i < horizontals.size(); ++i) {
    //     Point pt1(0.5, horizontals[i]);
    //     Point pt2(w_roi + 0.5, horizontals[i]);
    //     line(roiColor, pt1, pt2, Scalar(0, 255, 0), 1, LINE_AA); // green line
    // }

    // imwrite("../img/out/rois_color.jpg", roiColor);

    for (size_t i = 0; i < horizontals.size() - 1; ++i) {
        for (size_t j = 0; j < verticals.size() - 1; ++j) {
            // Corners of the square formed by two horizontal and two vertical lines
            Point topLeft(cvRound(verticals[j]), cvRound(horizontals[i]));
            Point bottomRight(cvRound(verticals[j + 1]), cvRound(horizontals[i + 1]));

            Rect square(topLeft, bottomRight);
            Mat pixel = roi(square).clone();
            // imwrite("../img/out/current_pixel.jpg", pixel);

            if (isRoiEmpty(pixel, 100.0, false)) { // /!\ Adjustable threshold
                binary.at<uchar>(i, j) = 255;
            }
            else {
                binary.at<uchar>(i, j) = 0;
            }
        }
    }

    return binary;
}
