#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <lapacke.h>

#define TO1D(row, col, num) ((row) * (num) + (col))

// Function to sort eigenvalues and reorder eigenvectors
void sort_eigenvalues_and_vectors(std::vector<double> &wr, std::vector<double> &vr, int N)
{
    // Create a vector of indices (0, 1, 2, ..., N-1)
    std::vector<int> indices(N);
    for (int i = 0; i < N; ++i)
    {
        indices[i] = i;
    }

    // Sort the indices based on the eigenvalues in descending order
    std::sort(indices.begin(), indices.end(), [&wr](int i1, int i2)
              {
                  return wr[i1] > wr[i2]; // Sort in descending order by the real eigenvalue
              });

    // Reorder eigenvalues and eigenvectors based on sorted indices
    std::vector<double> sorted_wr(N), sorted_vr(N * N);
    for (int i = 0; i < N; ++i)
    {
        sorted_wr[i] = wr[indices[i]];
        for (int j = 0; j < N; ++j)
        {
            sorted_vr[i * N + j] = vr[indices[i] * N + j]; // Reorder the eigenvectors (column-major)
        }
    }

    // Copy the sorted values back into the original vectors
    wr = std::move(sorted_wr);
    vr = std::move(sorted_vr);
}

void save_eigenvectors(const std::vector<double> &vr, int N, const std::string &filename)
{
    std::ofstream out_file(filename);
    if (!out_file)
    {
        std::cerr << "Error opening file for saving eigenvectors!" << std::endl;
        return;
    }

    // Write eigenvectors column by column (as LAPACK stores them in column-major order)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            out_file << vr[j * N + i] << " "; // Column-major indexing
        }
        out_file << "\n";
    }

    out_file.close();
    std::cout << "Eigenvectors saved to " << filename << std::endl;
}

int main(int argc, char **argv)
{
    // Check if image filename is provided
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <image_file>" << std::endl;
        return -1;
    }

    // Load the image in grayscale
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Convert to binary (black & white) using a threshold
    cv::Mat binary_image;
    // This takes the pixel values of the image matrix and sorts them to a binary matrix
    cv::threshold(image, binary_image, 128, 255, cv::THRESH_BINARY);

    cv::Mat binary_photo_matrix = cv::Mat::zeros(binary_image.size(), CV_8UC1);
    for (int i = 0; i < binary_image.rows; ++i)
    {
        for (int j = 0; j < binary_image.cols; ++j)
        {
            if (binary_image.at<uchar>(i, j) == 255)
            {                                            // White pixel
                binary_photo_matrix.at<uchar>(i, j) = 0; // Assign 0
            }
            else
            {                                            // Black pixel
                binary_photo_matrix.at<uchar>(i, j) = 1; // Assign 1
            }
        }
    }
    // Create the binary photo matrix and flatten it to position_mesh
    int No_points = binary_image.rows * binary_image.cols;
    std::vector<int> position_mesh;

    // Flatten the binary image to a 1D array (position_mesh)
    // We treat all black pixels (0) as 1 in the mesh and white pixels (255) as 0
    position_mesh.resize(No_points);
    for (int i = 0; i < binary_image.rows; ++i)
    {
        for (int j = 0; j < binary_image.cols; ++j)
        {
            // Flattening process
            int index = TO1D(i, j, binary_image.cols);
            if (binary_image.at<uchar>(i, j) == 0)
            {                             // Black pixel
                position_mesh[index] = 1; // Mark as occupied
            }
            else
            {                             // White pixel
                position_mesh[index] = 0; // Mark as empty
            }
        }
    }

    std::vector<double> Hamiltonian(No_points * No_points, 0.0);

    double increment = 1.0 / (binary_image.cols - 1);
    int N = binary_image.cols;

    // iterate through rows
    for (int j = 0; j < No_points; j++)
    {
        if (position_mesh[j] != 0)
        {
            if (j < position_mesh.size())
            {
                // iterate through columns
                for (int i = 0; i < No_points; i++)
                {
                    int ham_index = TO1D(j, i, No_points);
                    double denom = increment * increment;
                    if (i == j)
                    {
                        Hamiltonian[ham_index] = 4 / denom;
                    }

                    if (i == j + 1 && i % N != 0 && position_mesh[j + 1] != 0)
                    {
                        Hamiltonian[ham_index] = -1 / denom;
                    }

                    if (i == j - 1 && j % N != 0 && position_mesh[j - 1] != 0)
                    {
                        Hamiltonian[ham_index] = -1 / denom;
                    }

                    if (i == j - N && position_mesh[j - N] != 0)
                    {
                        Hamiltonian[ham_index] = -1 / denom;
                    }
                    if (i == j + N && position_mesh[j + N] != 0)
                    {
                        Hamiltonian[ham_index] = -1 / denom;
                    }
                }
            }
        }
    }

#ifdef DEBUG
    cv::Mat edge_image = cv::Mat::ones(binary_image.size(), CV_8UC1) * 255; // Initialize edge image with white (255)
    for (int i = 1; i < binary_image.rows - 1; ++i)
    {
        for (int j = 1; j < binary_image.cols - 1; ++j)
        {
            if (binary_image.at<uchar>(i, j) == 0) // Black pixel
            {
                // Check if any of the 4 neighboring pixels are white (i.e., edge of a black region)
                if (binary_image.at<uchar>(i - 1, j) == 255 || // Top
                    binary_image.at<uchar>(i + 1, j) == 255 || // Bottom
                    binary_image.at<uchar>(i, j - 1) == 255 || // Left
                    binary_image.at<uchar>(i, j + 1) == 255)   // Right
                {
                    edge_image.at<uchar>(i, j) = 0; // Mark edge pixels as black (0)
                }
            }
        }
    }

    cv::imwrite("./debug/" + std::string(argv[1]) + "_edge_image.png", edge_image);
    // Output the binary matrix (optional: for debugging or display purposes)
    std::string file_name = "./debug/" + std::string(argv[1]) + ".mat";
    std::ofstream log_file(file_name);
    log_file << "Binary Matrix:" << std::endl;
    for (int i = 0; i < binary_photo_matrix.rows; ++i)
    {
        for (int j = 0; j < binary_photo_matrix.cols; ++j)
        {
            log_file << (int)binary_photo_matrix.at<uchar>(i, j) << " ";
        }
        log_file << std::endl;
    }

    log_file.close();

    file_name = "./debug/" + std::string(argv[1]) + ".mesh";
    std::ofstream file(file_name);
    file << "Position Mesh:" << std::endl;
    for (int i = 0; i < position_mesh.size(); ++i)
    {
        file << position_mesh[i] << " ";
        // if ((i + 1) % binary_image.cols == 0)
        // {
        //     file << std::endl;
        // }
    }
    file.close();

    file_name = "./debug/" + std::string(argv[1]) + ".ham";
    std::ofstream filee(file_name);
    filee << "Ham Matrix:" << std::endl;
    for (int j = 0; j < No_points; j++)
    {
        filee << "----------------------" << std::endl;
        for (int i = 0; i < No_points; i++)
        {
            filee << Hamiltonian[j][i] << ", ";
        }
        filee << std::endl;
    }
    filee.close();
#endif

    // Arrays to store eigenvalues (real and imaginary parts) and eigenvectors
    std::vector<double> wr(No_points, 0.0);             // Real parts of eigenvalues
    std::vector<double> wi(No_points, 0.0);             // Imaginary parts of eigenvalues
    std::vector<double> vr(No_points * No_points, 0.0); // Right eigenvectors (column-major order)

    // LAPACKE_dgeev requires column-major input, so use LAPACK_ROW_MAJOR
    int info = LAPACKE_dgeev(LAPACK_COL_MAJOR,   // Row-major order for the matrix
                             'N',                // Do not compute left eigenvectors
                             'V',                // Compute right eigenvectors
                             No_points,          // Matrix size (N x N)
                             Hamiltonian.data(), // Matrix (input)
                             No_points,          // Leading dimension of A
                             wr.data(),          // Real part of eigenvalues
                             wi.data(),          // Imaginary part of eigenvalues
                             nullptr,            // Left eigenvectors (not needed)
                             No_points,          // Leading dimension of left eigenvectors (unused)
                             vr.data(),          // Right eigenvectors (output)
                             No_points);         // Leading dimension of right eigenvectors

    sort_eigenvalues_and_vectors(wr, vr, No_points);

    save_eigenvectors(vr, No_points, "eigenvectors.txt");

    return 0;
}
