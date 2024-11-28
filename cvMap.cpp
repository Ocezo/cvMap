
// CPP program to Map Cs and Ds
#include <vector>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;

void extractROIs(const Mat& image, const vector<float>& horizontal_rhos, const vector<float>& vertical_rhos, vector<Mat>& rois);
bool isRoiEmpty(const Mat& roi);
Mat cropRoi(const cv::Mat& roi, int t);
Mat remapToBinary(const Mat& roi, int num);
vector<double> linspace(double start, double end, int num);
Mat binToBinary(const Mat& roi, int num);

int main(int argc, char* argv[])
{
    int num = 10; // résolution des iamges binaires finales
    Mat src, srcColor, srcColor2;

    src = imread("../CnDs.jpg", IMREAD_GRAYSCALE);
    if (src.empty())
    {
        cerr << "Erreur de chargement de l'image !" << endl;
        return -1;
    }

    // 1/ Harris corners
    Mat harrisResponse;

    // Créer une copie en couleur de l'image
    cvtColor(src, srcColor, COLOR_GRAY2BGR);

    // Détection des coins de Harris
    cornerHarris(src, harrisResponse, 2, 3, 0.04);

    // Normaliser les valeurs pour faciliter le seuillage
    Mat harrisNorm;
    normalize(harrisResponse, harrisNorm, 0, 255, NORM_MINMAX, CV_32FC1);

    // Incrustation des coins détectés en rouge
    for (int y = 0; y < harrisNorm.rows; y++)
    {
        for (int x = 0; x < harrisNorm.cols; x++)
        {
            if ((int)harrisNorm.at<float>(y, x) > 144) // /!\ threshold
            {
                circle(srcColor, Point(x, y), 1, Scalar(0, 0, 255), FILLED); // Rouge
            }
        }
    }

    imwrite("../img/harris.jpg", srcColor);

    // 2/ Hough lines
    double h = src.size().height;
    double w = src.size().width;

    // Créer une copie en couleur de l'image
    cvtColor(src, srcColor2, COLOR_GRAY2BGR);

    // Détection des bords avec Canny pour la transformation de Hough
    Mat edges;
    Canny(src, edges, 50, 150);
    // imshow("Edges", edges);

    // Détection des lignes avec Hough
    vector<Vec2f> lines;
    vector<float> horizontal_rhos, vertical_rhos;
    HoughLines(edges, lines, 1, CV_PI / 180, 350); // /!\ Paramètre ajustable : 100->350

    // Filtrage et dessin des lignes horizontales et verticales
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        // On conserve les lignes proches  de la verticale ((theta proche de 0 ou π)
        if (abs(theta) < CV_PI / 36 || abs(theta - CV_PI) < CV_PI / 36)
        {
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            Point pt1(cvRound(x0 + h * (-b)), cvRound(y0 + h * (a)));
            Point pt2(cvRound(x0 - h * (-b)), cvRound(y0 - h * (a)));
            line(srcColor2, pt1, pt2, Scalar(0, 255, 0), 1, LINE_AA); // Lignes en vert
            // cout << ". Hori - theta = " << theta << " rad - rho = " << rho << " px" << endl;
            vertical_rhos.push_back(rho);
        }
        // On conserve les lignes proches de l'horizontale (theta proche de π/2)
        if (abs(theta - CV_PI / 2) < CV_PI / 36)
        {
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            Point pt1(cvRound(x0 + w * (-b)), cvRound(y0 + w * (a)));
            Point pt2(cvRound(x0 - w * (-b)), cvRound(y0 - w * (a)));
            line(srcColor2, pt1, pt2, Scalar(0, 255, 255), 1, LINE_AA); // Lignes en jaune
            // cout << ". Vert - theta = " << theta << " rad - rho = " << rho << " px" << endl;
            horizontal_rhos.push_back(rho);
        }
    }

    imwrite("../img/lines.jpg", srcColor2);

    sort(horizontal_rhos.begin(), horizontal_rhos.end());
    sort(vertical_rhos.begin(), vertical_rhos.end());

    // 3/ Extraction des imagettes (ROIs)
    vector<Mat> rois;
    extractROIs(src, horizontal_rhos, vertical_rhos, rois);

    // Afficher les imagettes extraites
    for (size_t k = 0; k < rois.size(); ++k) {
        imwrite("../img/rois/roi_" + to_string(k) + ".jpg", rois[k]);

        // Binning des rois en num x num
        Mat binary = binToBinary(rois[k], num);
        // cout << "Image binaire " << num << "x" << num << " :\n" << binary << endl;
        imwrite("../img/binning/roi_nxn_" + to_string(k) + ".jpg", binary);

        // Resize des rois en num x num
        // Mat binary2 = remapToBinary(rois[k], num);
        // imwrite("../img/resize/roi_nxn_" + to_string(k) + ".jpg", binary2);
    }
    
    waitKey(0);
    return 0;
}

void extractROIs(const Mat& image, const vector<float>& horizontal_rhos, const vector<float>& vertical_rhos, vector<Mat>& rois)
{
    // On suppose ici que horizontal_rhos et horizontal_rhos sont déjà triés !
    int height, width;

    // Boucler sur les paires de lignes pour détecter les carrés et extraire les ROIs
    for (size_t i = 0; i < horizontal_rhos.size() - 1; ++i) {
        for (size_t j = 0; j < vertical_rhos.size() - 1; ++j) {
            // Calculer les coins du carré formé par deux lignes horizontales et deux lignes verticales
            Point topLeft(cvRound(vertical_rhos[j]), cvRound(horizontal_rhos[i]));
            Point bottomRight(cvRound(vertical_rhos[j + 1]), cvRound(horizontal_rhos[i + 1]));

            // Vérifier les limites de l'image
            if (topLeft.x < 0 || topLeft.y < 0 || bottomRight.x > image.cols || bottomRight.y > image.rows) {
                continue; // Ignorer les carrés en dehors des limites de l'image
            }

            height = bottomRight.y - topLeft.y;
            width = bottomRight.x - topLeft.x;

            if (height < 53 || height > 61) {
                continue; // Ignorer les carrés de hauteur...
            }

            if (width < 53 || width > 61) {
                continue; // Ignorer les carrés de largeur...
            }

            // cout << "height = " << height << " - width = " << width << endl;

            // Extraire la ROI (imagette) correspondant au carré détecté
            Rect roi(topLeft, bottomRight);
            Mat imagette = image(roi).clone();

            Mat imagetteCropped = cropRoi(imagette, 5);
            // imwrite("../img/current_roi.jpg", imagetteCropped);

            if (isRoiEmpty(imagetteCropped)) {
                continue;
            }

            rois.push_back(imagetteCropped); // Ajouter l'imagette à la liste des ROIs
        }
    }
}

bool isRoiEmpty(const Mat& roi) {
    // Calcul de la variance de l'intensité des pixels
    Mat mean, stddev;
    meanStdDev(roi, mean, stddev);
    double variance = stddev.at<double>(0) * stddev.at<double>(0);

    // cout << "variance = " << variance << endl;

    // Définir un seuil de variance pour identifier les images "vides"
    return variance < 100.0; // Ajuste le seuil selon les essais
}

Mat cropRoi(const Mat& roi, int t) {
    // Vérifier que la ROI est assez grande pour être croppée
    if (roi.rows <= 2 * t || roi.cols <= 2 * t) {
        cerr << "La ROI est trop petite pour être croppée de 't' pixels sur chaque bord." << endl;
        return roi; // Retourne l'original si trop petit
    }

    // Définir la région à cropper
    Rect croppedRegion(t, t, roi.cols - 2 * t, roi.rows - 2 * t);

    // Retourner l'imagette croppée
    return roi(croppedRegion);
}

Mat remapToBinary(const Mat& roi, int num) {
    Mat resized, binary;
    
    // Redimension en num x num pixels
    resize(roi, resized, Size(num, num), 0, 0, INTER_AREA);

    // Binarisation
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

// Binning des rois en num x num
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
    //     line(roiColor, pt1, pt2, Scalar(0, 255, 0), 1, LINE_AA); // ligne en vert
    // }

    // for (int i = 0; i < horizontals.size(); ++i) {
    //     Point pt1(0.5, horizontals[i]);
    //     Point pt2(w_roi + 0.5, horizontals[i]);
    //     line(roiColor, pt1, pt2, Scalar(0, 255, 0), 1, LINE_AA); // ligne en vert
    // }

    // imwrite("../img/rois_color.jpg", roiColor);

    for (size_t i = 0; i < horizontals.size() - 1; ++i) {
        for (size_t j = 0; j < verticals.size() - 1; ++j) {
            // Coins du carré formé par deux lignes horizontales et deux lignes verticales
            Point topLeft(cvRound(verticals[j]), cvRound(horizontals[i]));
            Point bottomRight(cvRound(verticals[j + 1]), cvRound(horizontals[i + 1]));

            Rect square(topLeft, bottomRight);
            Mat pixel = roi(square).clone();
            // imwrite("../img/current_pixel.jpg", pixel);

            if (isRoiEmpty(pixel)) {
                binary.at<uchar>(i, j) = 255;
            }
            else {
                binary.at<uchar>(i, j) = 0;
            }
        }
    }

    return binary;
}