using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

var folderPath = "C:\\Users\\Connor\\OneDrive\\Pictures\\Software - Sample of images with issues";

try
{
    var imageFiles = Directory.GetFiles(folderPath, "*.jpg");

    foreach (var imagePath in imageFiles)
    {
        try
        {
            var numOfFaults = 0;

            if (IsMostlyOneColour(imagePath))
            {
                //Console.WriteLine($"Image {Path.GetFileName(imagePath)} is mostly one colour.");
                numOfFaults++;
            }
            if (IsHeavilyNoisy(imagePath))
            {
                //Console.WriteLine($"Image {Path.GetFileName(imagePath)} is heavily noisy.");
                numOfFaults++;
            }
            if (IsTinted(imagePath))
            {
                //Console.WriteLine($"Image {Path.GetFileName(imagePath)} is tinted.");
                numOfFaults++;
            }
            if (GetLaplacianVariance(imagePath) < 100)
            {
                //Console.WriteLine($"Image {Path.GetFileName(imagePath)} blurry: Variance -> {variance}");
                numOfFaults++;
            }

            Console.WriteLine($"Image {Path.GetFileName(imagePath)} has {numOfFaults} faults.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred while loading {Path.GetFileName(imagePath)}: {ex.Message}");
        }
    }
}
catch (Exception ex)
{
    Console.WriteLine($"An error occurred: {ex.Message}");
}

/// <summary>
/// Blur detection
/// </summary>
static double GetLaplacianVariance(string imagePath)
{
    try
    {
        // Load the image
        using var image = CvInvoke.Imread(imagePath, ImreadModes.Grayscale);
        using var laplacian = new Mat();

        // Calculate the Laplacian
        CvInvoke.Laplacian(image, laplacian, DepthType.Cv64F);

        // Initialize mean and stddev
        var mean = new MCvScalar();
        var stddev = new MCvScalar();

        // Calculate the variance of the Laplacian
        CvInvoke.MeanStdDev(laplacian, ref mean, ref stddev);
        var variance = stddev.V0 * stddev.V0;

        return variance;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return -1;
    }
}

/// <summary>
/// Determines if an image is mostly one block colour
/// </summary>
static bool IsMostlyOneColour(string imagePath)
{
    try
    {
        // Load the image
        using var image = CvInvoke.Imread(imagePath, ImreadModes.Color);

        // Initialize mean and stddev
        var mean = new MCvScalar();
        var stddev = new MCvScalar();

        // Calculate the mean and standard deviation of the image
        CvInvoke.MeanStdDev(image, ref mean, ref stddev);

        // If the standard deviation is very low, the image is mostly one colour
        var threshold = 20.0; // You can adjust this threshold based on your needs
        var isMostlyOneColour = stddev.V0 < threshold && stddev.V1 < threshold && stddev.V2 < threshold;

        return isMostlyOneColour;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return false;
    }
}

/// <summary>
/// Determines if an image is very heavily noisy
/// </summary>
static bool IsHeavilyNoisy(string imagePath)
{
    try
    {
        // Load the image
        using var image = CvInvoke.Imread(imagePath, ImreadModes.Grayscale);

        // Initialize mean and stddev
        var mean = new MCvScalar();
        var stddev = new MCvScalar();

        // Calculate the mean and standard deviation of the image
        CvInvoke.MeanStdDev(image, ref mean, ref stddev);

        // If the standard deviation is very high, the image is heavily noisy
        var noiseThreshold = 44.0; // You can adjust this threshold based on your needs
        var isHeavilyNoisy = stddev.V0 > noiseThreshold;

        return isHeavilyNoisy;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return false;
    }
}

/// <summary>
/// Determines if an image is tinted one colour, considering non-primary colours
/// </summary>
static bool IsTinted(string imagePath)
{
    try
    {
        // Load the image
        using var image = CvInvoke.Imread(imagePath, ImreadModes.Color);

        // Split the image into its colour channels
        var channels = image.Split();

        // Calculate the mean and standard deviation of each channel
        var meanBlue = new MCvScalar();
        var stddevBlue = new MCvScalar();
        CvInvoke.MeanStdDev(channels[0], ref meanBlue, ref stddevBlue);

        var meanGreen = new MCvScalar();
        var stddevGreen = new MCvScalar();
        CvInvoke.MeanStdDev(channels[1], ref meanGreen, ref stddevGreen);

        var meanRed = new MCvScalar();
        var stddevRed = new MCvScalar();
        CvInvoke.MeanStdDev(channels[2], ref meanRed, ref stddevRed);

        // Calculate the differences between the means of the channels
        var diffBlueGreen = Math.Abs(meanBlue.V0 - meanGreen.V0);
        var diffBlueRed = Math.Abs(meanBlue.V0 - meanRed.V0);
        var diffGreenRed = Math.Abs(meanGreen.V0 - meanRed.V0);

        // Calculate the differences between the standard deviations of the channels
        var stddevDiffBlueGreen = Math.Abs(stddevBlue.V0 - stddevGreen.V0);
        var stddevDiffBlueRed = Math.Abs(stddevBlue.V0 - stddevRed.V0);
        var stddevDiffGreenRed = Math.Abs(stddevGreen.V0 - stddevRed.V0);

        // If one channel's mean or standard deviation is significantly different from the others, the image is tinted
        var tintThresholdMean = 50.0; // You can adjust this threshold based on your needs
        var tintThresholdStddev = 20.0; // You can adjust this threshold based on your needs
        var isTinted = diffBlueGreen > tintThresholdMean || diffBlueRed > tintThresholdMean || diffGreenRed > tintThresholdMean ||
                        stddevDiffBlueGreen > tintThresholdStddev || stddevDiffBlueRed > tintThresholdStddev || stddevDiffGreenRed > tintThresholdStddev;

        return isTinted;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"An error occurred while processing {Path.GetFileName(imagePath)}: {ex.Message}");
        return false;
    }
}
