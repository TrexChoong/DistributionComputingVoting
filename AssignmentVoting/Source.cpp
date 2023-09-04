#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;
bool intersect(const Vec4i& l1, const Vec4i& l2, Point& intersectionPoint) {
	// Get the parameters of the lines.
	float x1 = l1[0], y1 = l1[1], x2 = l1[2], y2 = l1[3];
	float x3 = l2[0], y3 = l2[1], x4 = l2[2], y4 = l2[3];

	// Check if the lines are parallel.
	if ((y2 - y1) * (x4 - x3) == (y4 - y3) * (x2 - x1)) {
		return false;
	}

	// Calculate the slopes of the two lines.
	float m1 = (y2 - y1) / (x2 - x1);
	float m2 = (y4 - y3) / (x4 - x3);

	// Calculate the y-intercepts of the two lines.
	float b1 = y1 - m1 * x1;
	float b2 = y3 - m2 * x3;

	// Calculate the x-coordinate of the intersection point.
	float x = (b2 - b1) / (m1 - m2);

	// Calculate the y-coordinate of the intersection point.
	float y = m1 * x + b1;

	// Check if the intersection point is within the two lines.
	if (x >= min(x1, x2) && x <= max(x1, x2) && y >= min(y1, y2) && y <= max(y1, y2) &&
		x >= min(x3, x4) && x <= max(x3, x4) && y >= min(y3, y4) && y <= max(y3, y4)) {
		printf("Confirmed:%f %f\n", x, y);
		return true;
	}
	else {
		return false;
	}
}

static inline int
computeNumangle(double min_theta, double max_theta, double theta_step)
{
    int numangle = cvFloor((max_theta - min_theta) / theta_step) + 1;
    // If the distance between the first angle and the last angle is
    // approximately equal to pi, then the last angle will be removed
    // in order to prevent a line to be detected twice.
    if (numangle > 1 && fabs(CV_PI - (numangle - 1) * theta_step) < theta_step / 2)
        --numangle;
    return numangle;
}

static void
HoughLinesProbabilistic(Mat& image,
    float rho, float theta, int threshold,
    int lineLength, int lineGap,
    std::vector<Vec4i>& lines, int linesMax)
{
    Point pt;
    float irho = 1 / rho;
    RNG rng((uint64)-1);

    CV_Assert(image.type() == CV_8UC1);

    int width = image.cols;
    int height = image.rows;

    int numangle = computeNumangle(0.0, CV_PI, theta);
    int numrho = cvRound(((width + height) * 2 + 1) / rho);

#if defined HAVE_IPP && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_HOUGH
    CV_IPP_CHECK()
    {
        IppiSize srcSize = { width, height };
        IppPointPolar delta = { rho, theta };
        IppiHoughProbSpec* pSpec;
        int bufferSize, specSize;
        int ipp_linesMax = std::min(linesMax, numangle * numrho);
        int linesCount = 0;
        lines.resize(ipp_linesMax);
        IppStatus ok = ippiHoughProbLineGetSize_8u_C1R(srcSize, delta, &specSize, &bufferSize);
        Ipp8u* buffer = ippsMalloc_8u_L(bufferSize);
        pSpec = (IppiHoughProbSpec*)ippsMalloc_8u_L(specSize);
        if (ok >= 0) ok = ippiHoughProbLineInit_8u32f_C1R(srcSize, delta, ippAlgHintNone, pSpec);
        if (ok >= 0) { ok = CV_INSTRUMENT_FUN_IPP(ippiHoughProbLine_8u32f_C1R, image.data, (int)image.step, srcSize, threshold, lineLength, lineGap, (IppiPoint*)&lines[0], ipp_linesMax, &linesCount, buffer, pSpec); };

        ippsFree(pSpec);
        ippsFree(buffer);
        if (ok >= 0)
        {
            lines.resize(linesCount);
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        lines.clear();
        setIppErrorStatus();
    }
#endif

    Mat accum = Mat::zeros(numangle, numrho, CV_32SC1);
    Mat mask(height, width, CV_8UC1);
    std::vector<float> trigtab(numangle * 2);

    for (int n = 0; n < numangle; n++)
    {
        trigtab[n * 2] = (float)(cos((double)n * theta) * irho);
        trigtab[n * 2 + 1] = (float)(sin((double)n * theta) * irho);
    }
    const float* ttab = &trigtab[0];
    uchar* mdata0 = mask.ptr();
    std::vector<Point> nzloc;

    // stage 1. collect non-zero image points
    for (pt.y = 0; pt.y < height; pt.y++)
    {
        const uchar* data = image.ptr(pt.y);
        uchar* mdata = mask.ptr(pt.y);
        for (pt.x = 0; pt.x < width; pt.x++)
        {
            if (data[pt.x])
            {
                mdata[pt.x] = (uchar)1;
                nzloc.push_back(pt);
            }
            else
                mdata[pt.x] = 0;
        }
    }

    int count = (int)nzloc.size();

    // stage 2. process all the points in random order
    for (; count > 0; count--)
    {
        // choose random point out of the remaining ones
        int idx = rng.uniform(0, count);
        int max_val = threshold - 1, max_n = 0;
        Point point = nzloc[idx];
        Point line_end[2];
        float a, b;
        int* adata = accum.ptr<int>();
        int i = point.y, j = point.x, k, x0, y0, dx0, dy0, xflag;
        int good_line;
        const int shift = 16;

        // "remove" it by overriding it with the last element
        nzloc[idx] = nzloc[count - 1];

        // check if it has been excluded already (i.e. belongs to some other line)
        if (!mdata0[i * width + j])
            continue;

        // update accumulator, find the most probable line
        for (int n = 0; n < numangle; n++, adata += numrho)
        {
            int r = cvRound(j * ttab[n * 2] + i * ttab[n * 2 + 1]);
            r += (numrho - 1) / 2;
            int val = ++adata[r];
            if (max_val < val)
            {
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if (max_val < threshold)
            continue;

        // from the current point walk in each direction
        // along the found line and extract the line segment
        a = -ttab[max_n * 2 + 1];
        b = ttab[max_n * 2];
        x0 = j;
        y0 = i;
        if (fabs(a) > fabs(b))
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound(b * (1 << shift) / fabs(a));
            y0 = (y0 << shift) + (1 << (shift - 1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound(a * (1 << shift) / fabs(b));
            x0 = (x0 << shift) + (1 << (shift - 1));
        }

        for (k = 0; k < 2; k++)
        {
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if (k > 0)
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetic,
            // stop at the image border or in case of too big gap
            for (;; x += dx, y += dy)
            {
                uchar* mdata;
                int i1, j1;

                if (xflag)
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
                    break;

                mdata = mdata0 + i1 * width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if (*mdata)
                {
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if (++gap > lineGap)
                    break;
            }
        }

        good_line = std::abs(line_end[1].x - line_end[0].x) >= lineLength ||
            std::abs(line_end[1].y - line_end[0].y) >= lineLength;

        for (k = 0; k < 2; k++)
        {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if (k > 0)
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetic,
            // stop at the image border or in case of too big gap
            for (;; x += dx, y += dy)
            {
                uchar* mdata;
                int i1, j1;

                if (xflag)
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1 * width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if (*mdata)
                {
                    if (good_line)
                    {
                        adata = accum.ptr<int>();
                        for (int n = 0; n < numangle; n++, adata += numrho)
                        {
                            int r = cvRound(j1 * ttab[n * 2] + i1 * ttab[n * 2 + 1]);
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if (i1 == line_end[k].y && j1 == line_end[k].x)
                    break;
            }
        }

        if (good_line)
        {
            Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
            lines.push_back(lr);
            if ((int)lines.size() >= linesMax)
                return;
        }
    }
}

void HoughLinesPLocal(InputArray _image, OutputArray _lines,
    double rho, double theta, int threshold,
    double minLineLength, double maxGap)
{
    // CV_OCL_RUN(_image.isUMat() && _lines.isUMat(),
     //    ocl_HoughLinesP(_image, _lines, rho, theta, threshold, minLineLength, maxGap));

    Mat image = _image.getMat();
    std::vector<Vec4i> lines;
    HoughLinesProbabilistic(image, (float)rho, (float)theta, threshold, cvRound(minLineLength), cvRound(maxGap), lines, INT_MAX);
    Mat(lines).copyTo(_lines);
}

int main(int argc, char** argv)
{
	// Declare the output variables
	Mat dst, cdst, cdstP, cdstX;
	const char* default_file = "test2.png";
	const char* filename = argc >= 2 ? argv[1] : default_file;
	// Loads an image
	Mat src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		printf(" Program Arguments: [image_name -- default %s] \n", default_file);
		return -1;
	}
	// Edge detection
	Canny(src, dst, 50, 200, 3);
	// Copy edges to the images that will display the results in BGR
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	cdstP = cdst.clone();
	cdstX = cdst.clone();
	//cdstX = Mat::zeros(cdst.size(), cdst.type());
	// Standard Hough Line Transform
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
	// Probabilistic Line Transform
	vector<Vec4i> linesP; // will hold the results of the detection
    //HoughLinesProbabilistic(dst, 1, CV_PI / 180, 50, 50, 10, linesP, 10); // runs the actual detection
    //HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
    HoughLinesPLocal(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}

	//// Find the X patterns
	vector<Vec4i> x_patterns;
	  for (size_t i = 0; i < linesP.size(); i++) {
		for (size_t j = i + 1; j < linesP.size(); j++) {
		  Vec4i l1 = linesP[i];
		  Vec4i l2 = linesP[j];

		  // Check if the two lines intersect.
		  Point intersectionPoint;
		  printf("Intersect\n");
		  if (intersect(l1, l2, intersectionPoint)) {
			  x_patterns.push_back(cv::Vec4i(linesP[i][0], linesP[i][1], linesP[i][2], linesP[i][3]));
			  x_patterns.push_back(cv::Vec4i(linesP[j][0], linesP[j][1], linesP[j][2], linesP[j][3]));
		  }
		}
	  }

	// Draw the X patterns on the image.
	for (size_t i = 0; i < x_patterns.size(); i++) {
		//cv::line(cdst, cv::Point(x_patterns[i][0], x_patterns[i][1]), cv::Point(x_patterns[i][2], x_patterns[i][3]), cv::Scalar(0, 0, 255), 3, LINE_AA);
		cv::line(cdstX, cv::Point(x_patterns[i][0], x_patterns[i][1]), cv::Point(x_patterns[i][2], x_patterns[i][3]), cv::Scalar(0, 0, 255), 3, LINE_AA);
	}

	// Show results
	imshow("Source", src);
	imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
	imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
	imshow("Detected Lines (in red) - X Pattern Line Transform", cdstX);
	// Wait and Exit
	waitKey();
	return 0;
}