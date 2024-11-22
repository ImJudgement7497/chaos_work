#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>

int main()
{
    cv::Mat image = cv::imread("test.png"); // Read the image
    if (image.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cv::imshow("Display Window", image); // Show the image in a window
    cv::waitKey(0);                      // Wait for a key press
    return 0;
}
